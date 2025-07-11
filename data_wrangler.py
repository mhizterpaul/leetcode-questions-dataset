# main.py

import os
import subprocess
import pandas as pd
import torch
from datasets import Dataset
from huggingface_hub import login, snapshot_download
from tqdm import tqdm
from transformers import (
    AutoTokenizer, 
    AutoModelForCausalLM, 
    pipeline, 
    BitsAndBytesConfig
)

# --- Constants ---
HF_TOKEN = "YOUR_TOKEN"
MODEL_NAME = "microsoft/Phi-3-mini-4k-instruct"
LOCAL_MODEL_DIR = "/kaggle/working/Phi-3-mini-4k"
INPUT_CSV = "data/interim/filtered_dataset.csv"
OUTPUT_CSV = "data/dist/combined_questions_with_llm_features.csv"
BATCH_SIZE = 6
MAX_NEW_TOKENS = 256
TEMPERATURE = 0.7
TOP_P = 0.9

# --- Setup Paths ---
os.makedirs("data/dist", exist_ok=True)
os.makedirs("data/interim", exist_ok=True)

# --- Git Clone Datasets ---
def clone_and_prepare():
    subprocess.run(["git", "clone", "https://github.com/mhizterpaul/leetcode-questions-dataset.git"])
    subprocess.run(["mv", "leetcode-questions-dataset/data", "data"])
    print("✅ Dataset cloned and moved")

# --- Dataset Preprocessing ---
def preprocess_datasets():
    leetcode = pd.read_csv("data/leetcode_dataset/all_questions_details_cumulative.csv")
    interview = pd.read_csv("data/interview_dataset/SoftwareQuestions.csv", encoding='latin1')
    hack_algo = pd.read_csv("data/hackerrank_dataset/algorithms_questions.csv")
    hack_ds = pd.read_csv("data/hackerrank_dataset/data_structures_questions.csv")

    leetcode = leetcode[[
        'Question ID', 'Question Title', 'Question Text', 'Topic Tagged text', 'Difficulty Level'
    ]].rename(columns={
        'Question ID': 'id',
        'Question Title': 'question_title',
        'Question Text': 'question_text',
        'Topic Tagged text': 'tags',
        'Difficulty Level': 'difficulty'
    })
    leetcode['question'] = leetcode['question_title'] + "\n" + leetcode['question_text']
    leetcode.drop(columns=['question_title', 'question_text'], inplace=True)

    interview = interview[[
        'Question Number', 'Question', 'Answer', 'Category', 'Difficulty'
    ]].rename(columns={
        'Question Number': 'id',
        'Question': 'question',
        'Answer': 'correct',
        'Category': 'category',
        'Difficulty': 'difficulty'
    })
    interview = interview[interview['category'].isin([
        'Algorithms', 'Data Structures', 'Database Systems', 'System Design', 'Security'
    ])]
    for col in ['tags', 'a', 'b', 'c', 'd']:
        interview[col] = None

    def prepare_hack(df, category):
        df = df[['url', 'question_text']].rename(columns={'url': 'id', 'question_text': 'question'})
        for col in ['tags', 'a', 'b', 'c', 'd', 'correct']:
            df[col] = None
        df['category'] = category
        return df

    hack_algo = prepare_hack(hack_algo, "Algorithms")
    hack_ds = prepare_hack(hack_ds, "Data Structures")

    combined = pd.concat([leetcode, interview, hack_algo, hack_ds], ignore_index=True)
    combined = combined.dropna(subset=['question']).drop_duplicates(subset=['question'])

    combined[['id', 'question']].to_csv(INPUT_CSV, index=False)
    print(f"✅ Processed and saved combined dataset to {INPUT_CSV}")

# --- Model Setup ---
def load_model():
    login(token=HF_TOKEN)
    if not os.path.exists(LOCAL_MODEL_DIR):
        print(f"⬇️ Downloading model to {LOCAL_MODEL_DIR}")
        snapshot_download(
            repo_id=MODEL_NAME,
            local_dir=LOCAL_MODEL_DIR,
            local_dir_use_symlinks=False
        )
    else:
        print(f"✅ Using cached model from {LOCAL_MODEL_DIR}")

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16
    )

    tokenizer = AutoTokenizer.from_pretrained(LOCAL_MODEL_DIR, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        LOCAL_MODEL_DIR,
        device_map="auto",
        trust_remote_code=True,
        quantization_config=bnb_config
    )

    llm_pipeline = pipeline("text-generation", model=model, tokenizer=tokenizer, padding=True)
    return llm_pipeline


# --- Step 3: Prompt Template ---
def build_prompt(question: str) -> str:
    return tokenizer.apply_chat_template([
    {
        "role": "system",
        "content": (
            "You are an expert tutor generating multiple-choice questions (MCQs) for technical interviews.\n"
            "Each MCQ must include:\n"
            "- Four **plausible but incorrect** options: 'a', 'b', 'c', 'd'\n"
            "- One **correct** answer labeled as 'correct'\n\n"
            "Constraints:\n"
            "- 'correct' must NOT appear in a–d\n"
            "- All five options must be **distinct**, **relevant**, and **technically sound**\n"
            "- 'correct' must be **difficult to infer** from the others\n"
            "- Infer 1–3 technical **tags** from question context (e.g., 'linked list', 'recursion')\n"
            "- Use one category from: [algorithms, data structures, database systems, system design, security]\n\n"
            "Respond ONLY with a valid JSON object as specified below. No commentary or formatting beyond that."
        )
    },
    {
        "role": "user",
        "content": f"""
Question:
\"\"\"{question.strip()}\"\"\"

Example output:
{{
  "a": "Iterate both lists using a for loop and append sums directly to the end",
  "b": "Convert the linked lists to integers and sum them using Python's built-in arithmetic",
  "c": "Use stacks to reverse the digits and then perform manual addition with carry-over",
  "d": "Merge the two lists and sort them before summing",
  "correct": "Traverse both lists node-by-node, adding digits with carry, and build a new list for the result",
  "tags": ["linked list", "carry", "addition"],
  "category": "algorithms"
}}

Note:
- Do NOT duplicate options.
- Ensure responses are clean, precise, and valid JSON.
"""
    },
], tokenize=False, add_generation_prompt=True)


# --- Custom Workflow Preprocessing Function ---
def clean_json_string_custom_workflow(json_str_raw):
    """
    Attempts to clean a raw string based on the user's specified workflow:
    remove {}, split by ,, split by :, validate key (as string), remove newlines in value, merge valid, add {}.
    This is a highly experimental and likely fragile heuristic.
    Trims whitespace from keys and values.
    Does NOT include explicit delimiter balance check.
    """
    # 1. Remove outer {}
    # Be more robust to whitespace around braces
    cleaned_raw = json_str_raw.strip()
    if cleaned_raw.startswith('{') and cleaned_raw.endswith('}'):
        content_without_braces = cleaned_raw[1:-1].strip()
    else:
        # If braces are missing or misplaced, this workflow won't work as intended.
        # Return the raw string or an empty string to indicate failure.
        print(f"Warning: Raw string does not start/end with {{}} as expected for custom workflow: {json_str_raw}")
        return "" # Cannot apply workflow if basic structure is missing

    # 2. Split by comma (carefully, to not split inside strings)
    segments = []
    current_segment = []
    in_string = False

    i = 0
    while i < len(content_without_braces):
        char = content_without_braces[i]

        if char == '"':
            current_segment.append(char)
            in_string = not in_string
        elif char == ',' and not in_string:
            # Found a comma outside a string - this is a segment boundary
            segments.append("".join(current_segment).strip())
            current_segment = [] # Start new segment
        else:
            current_segment.append(char)

        i += 1

    # Add the last segment after the loop
    segments.append("".join(current_segment).strip())

    valid_segments_processed = []

    # 3. & 4. Split by colon, Validate key, Trim key/value, Remove newlines in value
    for segment in segments:
        # Split by the first colon
        colon_index = segment.find(':')
        if colon_index != -1:
            key_part = segment[:colon_index].strip() # Trim key part
            value_part_raw = segment[colon_index + 1:] # Raw value part

            # Check if the key part is a valid JSON string (starts and ends with ")
            if key_part.startswith('"') and key_part.endswith('"'):
                 # This segment appears to have a valid string key.
                 # Now, process the value part to remove newlines within strings.
                 cleaned_value_chars = []
                 in_string_value = False
                 j = 0
                 while j < len(value_part_raw):
                      val_char = value_part_raw[j]
                      if in_string_value and val_char == '\n':
                           # Remove newline within string in value
                           pass
                      elif val_char == '"':
                            cleaned_value_chars.append(val_char)
                            in_string_value = not in_string_value
                      else:
                           # Keep other characters in value (including whitespace and structural chars outside quotes)
                           cleaned_value_chars.append(val_char)
                      j += 1

                 value_part_cleaned = "".join(cleaned_value_chars).strip() # Trim cleaned value
                 valid_segments_processed.append(key_part + ":" + value_part_cleaned) # Reconstruct with cleaned value
            else:
                 # Key is not a valid string. Discard this segment.
                 # print(f"Debug (Workflow): Discarding segment with invalid key: {segment}")
                 pass # Discard the segment
        else:
            # Segment does not contain a colon. This might be a structural element like an empty object/array.
            # Or it could be malformed. Let's check if it contains only whitespace or valid structural chars.
            # A simple check: if it contains only whitespace or valid structural chars, keep it.
            stripped_segment = segment.strip()
            if stripped_segment == '' or all(c in '{}[]:, \n\t' for c in stripped_segment):
                 valid_segments_processed.append(stripped_segment)
            else:
                 # print(f"Debug (Workflow): Discarding segment without colon and not structural: {segment}")
                 pass # Discard


    # 5. Merge valid segments
    content_without_braces_cleaned = ",".join(valid_segments_processed)

    # 6. Add back {}
    json_str_cleaned = "{" + content_without_braces_cleaned + "}"

    # No explicit delimiter balance check as requested.

    return json_str_cleaned

# Function to process a single question with retries
def process_question_with_retries(qid, question, llm_pipeline, max_retries=MAX_RETRIES):
    retries = 0
    success = False
    parsed_data = None
    raw_output_debug = None # Initialize debug field

    while retries < max_retries and not success:
        try:
            prompt = build_prompt(question)
            # Use the pipeline for a single prompt, results is a list of dicts
            results = llm_pipeline(prompt, max_new_tokens=MAX_TOKENS, temperature=TEMPERATURE, top_p=TOP_P, do_sample=True, use_cache=False)

            response_text = None
            if isinstance(results, list) and results and isinstance(results[0], dict) and 'generated_text' in results[0]:
                response_text = results[0]['generated_text']
            elif isinstance(results, dict) and 'generated_text' in results:
                response_text = results['generated_text']
            else:
                print(f"❌ Unexpected response format for ID {qid}, retry {retries + 1}. Raw result: {results}")
                raw_output_debug = f"Unexpected response format.\nRaw Result: {str(results)}"
                retries += 1
                continue # Continue to the next retry

            raw_output_debug = response_text # Store raw response text in debug

            # Extract text after the last <|assistant|> tag
            assistant_tag = "<|assistant|>"
            last_assistant_index = response_text.rfind(assistant_tag)

            extracted_text_after_assistant = ""
            if last_assistant_index != -1:
                extracted_text_after_assistant = response_text[last_assistant_index + len(assistant_tag):].strip()
            else:
                 print(f"❌ Could not find the last '{assistant_tag}' tag for ID {qid}, retry {retries + 1}.")
                 raw_output_debug = f"Could not find '{assistant_tag}' tag.\nRaw Response Text: {response_text}"
                 retries += 1
                 continue


            # Find the JSON block using ```json and ```
            json_block_start = extracted_text_after_assistant.find('```json')
            json_block_end = extracted_text_after_assistant.find('```', json_block_start + 7)

            json_str_raw = ""
            if json_block_start != -1 and json_block_end != -1:
                json_str_raw = extracted_text_after_assistant[json_block_start + 7 : json_block_end].strip()
            else:
                 print(f"❌ Could not find ```json block for ID {qid}, retry {retries + 1}. Extracted Text: {extracted_text_after_assistant}")
                 raw_output_debug = f"Could not find JSON block.\nExtracted Text: {extracted_text_after_assistant}"
                 retries += 1
                 continue

            json_str_to_parse = json_str_raw # Start with the extracted raw string

            # Attempt to parse the JSON
            try:
                parsed = json.loads(json_str_to_parse)
                success = True # If parsing is successful, set success to True
            except json.JSONDecodeError as e:
                print(f"JSON Decode Error for ID {qid}, retry {retries + 1}: {e}")
                print(f"--- Failed JSON string for ID {qid}, retry {retries + 1} ---\n{json_str_to_parse}\n--- End Failed JSON string ---")

                # Attempt cleanup if retries remain
                if retries < max_retries -1 :
                     print(f"Attempting JSON cleanup for ID {qid}, retry {retries + 1}.")
                     json_str_cleaned = clean_json_string_custom_workflow(json_str_raw)
                     json_str_to_parse = json_str_cleaned # Try parsing the cleaned string

                retries += 1 # Increment retry count
                continue # Continue to the next retry (either with cleaned string or failing)


            # If parsing was successful, validate the data
            if success:
                 if not isinstance(parsed, dict):
                     print(f"❌ Parsed JSON is not a dictionary for ID {qid}, retry {retries + 1}. Parsed object: {parsed}")
                     success = False # Mark as failed if not a dictionary
                 else:
                     # Try extracting data based on the parsed dictionary structure
                     try:
                        option_a = parsed.get('a') if 'a' in parsed else parsed.get('options', {}).get('a')
                        option_b = parsed.get('b') if 'b' in parsed else parsed.get('options', {}).get('b')
                        option_c = parsed.get('c') if 'c' in parsed else parsed.get('options', {}).get('c')
                        option_d = parsed.get('d') if 'd' in parsed else parsed.get('options', {}).get('d')
                        correct_answer = parsed.get('correct') if 'correct' in parsed else parsed.get('options', {}).get('correct')
                        tags_list = parsed.get('tags') if 'tags' in parsed else parsed.get('options', {}).get('tags', [])
                        category = parsed.get('category') if 'category' in parsed else parsed.get('options', {}).get('category')

                        # Ensure required keys exist and correct is not a single character option
                        if not (option_a is not None and option_b is not None and option_c is not None and option_d is not None and correct_answer is not None and category is not None):
                             print(f"❌ Parsed dictionary missing required keys for ID {qid}, retry {retries + 1}. Parsed dict: {parsed}")
                             success = False # Mark as failed if keys are missing
                        elif isinstance(correct_answer, str) and correct_answer.strip().lower() in ['a', 'b', 'c', 'd']:
                             print(f"❌ Correct answer is a character ('a', 'b', 'c', or 'd') for ID {qid}, retry {retries + 1}.")
                             success = False # Mark as failed
                        else:
                             # Case-insensitive and whitespace-stripped comparison for leak detection
                             correct_stripped = str(correct_answer).strip().lower()
                             leaked = False
                             for option_key in ['a', 'b', 'c', 'd']:
                                 option_text = parsed.get(option_key) if option_key in parsed else parsed.get('options', {}).get(option_key)
                                 if option_text is not None and isinstance(option_text, str) and option_text.strip().lower() == correct_stripped:
                                     leaked = True
                                     break

                             if leaked:
                                  print(f"⚠️ Correct leaked into options for ID {qid}, retry {retries + 1}. Skipping.")
                                  success = False # Mark as failed if leak detected

                             if success: # Only update entry if no leak, keys are present, AND correct is not a single character option
                                 parsed_data = { # Store parsed data if validation passes
                                     "id": qid,
                                     "question": question,
                                     "a": option_a,
                                     "b": option_b,
                                     "c": option_c,
                                     "d": option_d,
                                     "correct": correct_answer,
                                     "tags": ", ".join([str(tag) for tag in tags_list if tag is not None]),
                                     "category": category
                                 }

                     except Exception as extract_e:
                        print(f"❌ Error extracting data from parsed JSON for ID {qid}, retry {retries + 1}: {extract_e}. Parsed dict: {parsed}")
                        success = False # Mark as failed on extraction error

        except Exception as e:
            print(f"❌ An unexpected error occurred for ID {qid}, retry {retries + 1}: {e}")
            raw_output_debug = f"Unexpected error: {e}\nRaw Response Text (if available): {response_text if 'response_text' in locals() else 'N/A'}"
            retries += 1
            # No continue here, the while loop condition handles retries

    if not success:
        print(f"❌ Failed to process ID {qid} after {max_retries} retries.")
        # Return None or an indicator of failure
        return None
    else:
        print(f"✅ Successfully processed ID {qid} after {retries + 1} attempts.")
        return parsed_data


# --- Main Processing Logic ---
if __name__ == "__main__":
    clone_and_prepare()
    preprocess_datasets()
    llm = load_model()
    os.makedirs(os.path.dirname(OUTPUT_CSV), exist_ok=True)

    # Step 1: Load and Filter Data
    print("⏳ Loading and filtering data...")
    df = pd.read_csv(INPUT_CSV)[['id', 'question']]

    # Check for existing output and load successfully processed IDs
    processed_ids = set()
    if os.path.exists(OUTPUT_CSV):
        try:
            existing_df = pd.read_csv(OUTPUT_CSV)
            if not existing_df.empty and 'id' in existing_df.columns and 'category' in existing_df.columns:
                # Consider an ID processed only if at least 'a', 'b', 'c', 'd', 'correct', and 'category' are not None
                # Adjusted criteria to only check 'category' as a simple indicator for successfully processed rows written by THIS script
                # This is a simplification, a more robust check might look for all generated columns.
                # However, relying on the final columns written by this script for 'successful' state is reasonable for resume.
                successfully_processed_df = existing_df.dropna(subset=['a', 'b', 'c', 'd', 'correct', 'tags', 'category'])
                processed_ids = set(successfully_processed_df['id'].astype(str).tolist())
                print(f"Loaded {len(processed_ids)} successfully processed questions from {OUTPUT_CSV}.")
            else:
                print(f"Existing output file {OUTPUT_CSV} is empty or missing required columns. Starting from scratch.")
        except pd.errors.EmptyDataError:
            print(f"Existing output file {OUTPUT_CSV} is empty. Starting from scratch.")
        except Exception as e:
            print(f"Error loading existing output file {OUTPUT_CSV}: {e}. Starting from scratch.")

    # Filter out already successfully processed questions
    initial_rows = len(df)
    df_to_process = df[~df['id'].astype(str).isin(processed_ids)].copy()
    skipped_successfully_processed_rows = initial_rows - len(df_to_process)
    if skipped_successfully_processed_rows > 0:
        print(f"Skipping {skipped_successfully_processed_rows} questions that have already been successfully processed.")

    print(f"🚀 Starting processing for {len(df_to_process)} questions.")

    # Step 2: Prepare Data for Hugging Face Datasets (will be done per batch)

    # Lists to track failed questions across batches
    failed_items_across_batches = []

    # Step 3 & 4: Define Batch Processing Loop and Process Each Batch
    total_questions_to_process = len(df_to_process)
    batch_size_calc = (total_questions_to_process + NUM_BATCHES - 1) // NUM_BATCHES # Calculate batch size to distribute evenly
    if batch_size_calc == 0:
        batch_size_calc = 1 # Ensure batch size is at least 1 if few questions remain

    current_start_index = 0

    print(f"Dividing {total_questions_to_process} questions into {NUM_BATCHES} batches of approximately {batch_size_calc} questions.")

    for batch_num in range(NUM_BATCHES):
        print(f"\n--- Processing Batch {batch_num + 1}/{NUM_BATCHES} ---")

        # Get the slice for the current batch
        batch_end_index = min(current_start_index + batch_size_calc, total_questions_to_process)
        current_batch_df = df_to_process.iloc[current_start_index:batch_end_index].copy()

        # Add failed items from previous batches to the current batch (if any)
        if failed_items_across_batches:
            print(f"Adding {len(failed_items_across_batches)} failed/incomplete items from previous batches to Batch {batch_num + 1}.")
            failed_df = pd.DataFrame(failed_items_across_batches, columns=['id', 'question'])
            current_batch_df = pd.concat([current_batch_df, failed_df], ignore_index=True)
            failed_items_across_batches = [] # Clear the list after adding them to the current batch

        if current_batch_df.empty:
            print(f"Batch {batch_num + 1} is empty. Skipping.")
            current_start_index = batch_end_index # Move index even if batch is empty
            continue

        # Convert current batch DataFrame to Hugging Face Dataset
        batch_dataset = Dataset.from_pandas(current_batch_df)

        print(f"Processing {len(batch_dataset)} questions in Batch {batch_num + 1}.")

        # Function to map questions to prompts
        def generate_prompt_column(example):
            example["prompt"] = build_prompt(example["question"])
            return example

        # Apply the prompt generation
        batch_dataset = batch_dataset.map(generate_prompt_column, num_proc=os.cpu_count()) # Use multiple processes

        # Step 5: Process and Validate LLM Responses (using batched pipeline)
        # Prepare inputs for the pipeline - just the prompt column
        pipeline_inputs = batch_dataset['prompt']

        batch_successful_entries = []

        # Process the batch with the LLM pipeline
        try:
            # The pipeline processes a list of prompts and returns a list of results
            batch_results = llm(pipeline_inputs, max_new_tokens=MAX_TOKENS, temperature=TEMPERATURE, top_p=TOP_P, do_sample=True, use_cache=False)

            # Step 5 & 6 & 7: Process and Validate, Handle Failed, Append Successful
            for i, result in enumerate(batch_results):
                qid = batch_dataset[i]['id']
                question = batch_dataset[i]['question']
                response_text = None # Initialize

                # Extract generated_text from the result
                if isinstance(result, list) and result and isinstance(result[0], dict) and 'generated_text' in result[0]:
                    response_text = result[0]['generated_text']
                elif isinstance(result, dict) and 'generated_text' in result:
                    response_text = result['generated_text']
                else:
                    print(f"❌ Batch Processing: Unexpected response format for ID {qid}. Raw result: {result}")
                    failed_items_across_batches.append({'id': qid, 'question': question})
                    continue


                # --- Extract text after the last <|assistant|> tag ---
                assistant_tag = "<|assistant|>"
                last_assistant_index = response_text.rfind(assistant_tag)

                extracted_text_after_assistant = ""
                if last_assistant_index != -1:
                    extracted_text_after_assistant = response_text[last_assistant_index + len(assistant_tag):].strip()
                else:
                    print(f"❌ Batch Processing: Could not find the last '{assistant_tag}' tag for ID {qid}. Raw response text: {response_text}")
                    failed_items_across_batches.append({'id': qid, 'question': question})
                    continue


                # --- Now, find the JSON block using ```json and ``` within the extracted text ---
                json_block_start = extracted_text_after_assistant.find('```json')
                json_block_end = extracted_text_after_assistant.find('```', json_block_start + 7)

                json_str_raw = ""
                if json_block_start != -1 and json_block_end != -1:
                    json_str_raw = extracted_text_after_assistant[json_block_start + 7 : json_block_end].strip()
                else:
                    print(f"❌ Batch Processing: Could not find ```json block for ID {qid} within text after '{assistant_tag}'. Extracted Text: {extracted_text_after_assistant}")
                    failed_items_across_batches.append({'id': qid, 'question': question})
                    continue

                # --- Parsing and Validation (Single Attempt for now in Batch) ---
                # The retry logic is handled by adding to failed_items_across_batches and reprocessing in subsequent batches.
                parsed = None
                success = False
                json_str_to_parse = json_str_raw # Start with raw string for parsing attempts

                try:
                    parsed = json.loads(json_str_to_parse)
                    success = True # If parsing is successful, set success to True

                except json.JSONDecodeError as e:
                    print(f"❌ Batch Processing: JSON Decode Error for ID {qid}: {e}")
                    print(f"--- Failed JSON string for ID {qid} ---\n{json_str_to_parse}\n--- End Failed JSON string ---")

                    # Attempt cleanup if retries remain (this will happen in the retry phase)
                    # For now, just log the failure and add to failed items
                    print(f"Batch Processing: Attempting JSON cleanup via custom workflow for ID {qid}.")
                    json_str_cleaned = clean_json_string_custom_workflow(json_str_raw)
                    # Do NOT reassign json_str_to_parse here, as we are not retrying immediately in the batch loop.
                    # The cleaned string will be available if this item fails and enters the retry phase.
                    success = False # Ensure success is False on decode error


                if success:
                    # Check if the parsed output is a dictionary
                    if not isinstance(parsed, dict):
                        print(f"❌ Batch Processing: Parsed JSON is not a dictionary for ID {qid}. Parsed object: {parsed}")
                        success = False # Mark as failed if not a dictionary
                    else:
                        # Try extracting data based on the parsed dictionary structure
                        try:
                            option_a = parsed.get('a') if 'a' in parsed else parsed.get('options', {}).get('a')
                            option_b = parsed.get('b') if 'b' in parsed else parsed.get('options', {}).get('b')
                            option_c = parsed.get('c') if 'c' in parsed else parsed.get('options', {}).get('c')
                            option_d = parsed.get('d') if 'd' in parsed else parsed.get('options', {}).get('d')
                            correct_answer = parsed.get('correct') if 'correct' in parsed else parsed.get('options', {}).get('correct')
                            tags_list = parsed.get('tags') if 'tags' in parsed else parsed.get('options', {}).get('tags', [])
                            category = parsed.get('category') if 'category' in parsed else parsed.get('options', {}).get('category')


                            # Ensure required keys exist (including 'correct' now being the text)
                            if not (option_a is not None and option_b is not None and option_c is not None and option_d is not None and correct_answer is not None and category is not None):
                                print(f"❌ Batch Processing: Parsed dictionary missing required keys for ID {qid}. Parsed dict: {parsed}")
                                success = False # Mark as failed if keys are missing
                            elif isinstance(correct_answer, str) and correct_answer.strip().lower() in ['a', 'b', 'c', 'd']:
                                print(f"❌ Batch Processing: Correct answer is a character ('a', 'b', 'c', or 'd') for ID {qid}. Marking as failed for retry.")
                                success = False # Mark as failed
                            else:
                                # Case-insensitive and whitespace-stripped comparison for leak detection
                                correct_stripped = str(correct_answer).strip().lower()
                                leaked = False
                                for option_key in ['a', 'b', 'c', 'd']:
                                    option_text = parsed.get(option_key) if option_key in parsed else parsed.get('options', {}).get(option_key)
                                    if option_text is not None and isinstance(option_text, str) and option_text.strip().lower() == correct_stripped:
                                        leaked = True
                                        break

                                if leaked:
                                    print(f"⚠️ Batch Processing: Correct leaked into options for ID {qid}. Marking as failed for retry.")
                                    success = False # Mark as failed if leak detected

                                if success: # Only update entry if no leak, keys are present, AND correct is not a single character option
                                    batch_successful_entries.append({
                                        "id": qid,
                                        "question": question,
                                        "a": option_a,
                                        "b": option_b,
                                        "c": option_c,
                                        "d": option_d,
                                        "correct": correct_answer,
                                        "tags": ", ".join([str(tag) for tag in tags_list if tag is not None]),
                                        "category": category
                                    })

                        except Exception as extract_e:
                            print(f"❌ Batch Processing: Error extracting data from parsed JSON for ID {qid}: {extract_e}. Parsed dict: {parsed}")
                            success = False # Mark as failed on extraction error

                if not success:
                    print(f"❌ Batch Processing: Failed to process ID {qid}. Adding to failed list for retry.")
                    failed_items_across_batches.append({'id': qid, 'question': question})


        except Exception as e:
            print(f"❌ Batch {batch_num + 1} failed during pipeline execution: {e}")
            # Add all items from the current batch to failed_items_across_batches if the batch fails
            for _, row in current_batch_df.iterrows():
                failed_items_across_batches.append({'id': row['id'], 'question': row['question']})


        # Write the successfully processed entries for the current batch
        if batch_successful_entries:
            output_columns = ["id", "question", "a", "b", "c", "d", "correct", "tags", "category"]
            batch_df_to_save = pd.DataFrame(batch_successful_entries, columns=output_columns)
            batch_df_to_save.to_csv(OUTPUT_CSV, mode='a', index=False, header=not os.path.exists(OUTPUT_CSV))
            print(f"✅ Saved {len(batch_successful_entries)} successful entries from Batch {batch_num + 1} to {OUTPUT_CSV}")

        # Update the starting index for the next batch
        current_start_index = batch_end_index


    # Step 9 & 10: Retry Remaining Failed Items and Append Successful Retries
    print(f"\n--- Retrying Remaining Failed Items ---")
    if failed_items_across_batches:
        print(f"Attempting to retry {len(failed_items_across_batches)} failed/incomplete items.")
        final_failed_items = []
        retried_successful_entries = []

        # Process each remaining failed item individually with retries
        for item in tqdm(failed_items_across_batches, desc="Retrying failed items"):
            qid = item['id']
            question = item['question']
            processed_result = process_question_with_retries(qid, question, llm, max_retries=MAX_RETRIES)

            if processed_result:
                retried_successful_entries.append(processed_result)
            else:
                final_failed_items.append({'id': qid, 'question': question})

        # Write the successfully retried entries
        if retried_successful_entries:
            output_columns = ["id", "question", "a", "b", "c", "d", "correct", "tags", "category"]
            retried_df_to_save = pd.DataFrame(retried_successful_entries, columns=output_columns)
            retried_df_to_save.to_csv(OUTPUT_CSV, mode='a', index=False, header=False) # No header for appended data
            print(f"✅ Saved {len(retried_successful_entries)} successful entries from retries to {OUTPUT_CSV}")

        # Step 11: Final Check and Logging
        if final_failed_items:
            print(f"❌ {len(final_failed_items)} items still failed after all retries:")
            for item in final_failed_items:
                print(f"  ID: {item['id']}, Question (first 100 chars): {item['question'][:100]}...")
        else:
            print("✅ All remaining failed items were successfully processed during retries.")

    else:
        print("✅ No failed items remained after the initial batch processing.")

    # Step 12: Finish task
    print("\n--- Processing Complete ---")
    print(f"Final output saved to {OUTPUT_CSV}")
    # You can add a final display of the head of the output file if desired.
    # display(pd.read_csv(OUTPUT_CSV).head()) # Uncomment to display head of final output