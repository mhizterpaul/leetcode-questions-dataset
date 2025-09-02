import os
import subprocess
import pandas as pd
import google.generativeai as genai
import time
import json
from tqdm import tqdm

# --- Constants ---
MODEL_NAME = "gemini-2.0-flash"
INPUT_CSV = "data/interim/filtered_dataset.csv"
OUTPUT_CSV = "data/dist/mcq_dataset.csv"
MAX_RETRIES = 3
RETRY_DELAY = 2  # seconds

# --- Setup Paths ---
os.makedirs("data/dist", exist_ok=True)
os.makedirs("data/interim", exist_ok=True)

# --- Git Clone Datasets ---
def clone_and_prepare():
    if not os.path.exists("data/leetcode_dataset"):
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
    gemini_api_key = os.environ.get(GEMINI_API_KEY)
    if not gemini_api_key:
        raise ValueError("Gemini API key not found. Please set the GEMINI_API_KEY environment variable.")
    genai.configure(api_key=gemini_api_key)
    model = genai.GenerativeModel(MODEL_NAME)
    return model

# --- Prompt Template ---
def build_prompt(question: str) -> str:
    return f"""
You are an expert tutor generating multiple-choice questions (MCQs) for technical interviews.
Each MCQ must include:
- Four plausible but incorrect options: 'a', 'b', 'c', 'd'
- One correct answer labeled as 'correct'

Constraints:
- 'correct' must NOT appear in a–d
- All five options must be distinct, relevant, and technically sound
- 'correct' must be difficult to infer from the others
- Infer 1–3 technical tags from question context (e.g., 'linked list', 'recursion')
- Use one category from: [algorithms, data structures, database systems, system design, security]

Respond ONLY with a valid JSON object as specified below. No commentary or formatting beyond that.

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

# --- Utility: Deduplicate CSV by question ---
def deduplicate_csv_by_question(csv_path):
    if os.path.exists(csv_path):
        try:
            df = pd.read_csv(csv_path)
            if 'question' in df.columns:
                before = len(df)
                df = df.drop_duplicates(subset=['question'])
                after = len(df)
                if after < before:
                    df.to_csv(csv_path, index=False)
                    print(f"✅ Deduplicated {csv_path}: removed {before - after} duplicates, {after} unique questions remain.")
                else:
                    print(f"No duplicates found in {csv_path}.")
            else:
                print(f"No 'question' column in {csv_path}, skipping deduplication.")
        except Exception as e:
            print(f"Error deduplicating {csv_path}: {e}")
    else:
        print(f"File {csv_path} does not exist, skipping deduplication.")

# --- Main Processing Logic ---
if __name__ == "__main__":
    import sys
    # CLI: python data_wrangler.py dedup_only
    if len(sys.argv) > 1 and sys.argv[1] == "dedup_only":
        deduplicate_csv_by_question(OUTPUT_CSV)
        sys.exit(0)

    # Always deduplicate output before starting
    deduplicate_csv_by_question(OUTPUT_CSV)

    # --- Step 1: Check if filtered_dataset.csv has 3246+ questions ---
    skip_processing = False
    if os.path.exists(INPUT_CSV):
        try:
            interim_df = pd.read_csv(INPUT_CSV)
            # Drop duplicates by question in input
            interim_df = interim_df.drop_duplicates(subset=['question'])
            interim_df.to_csv(INPUT_CSV, index=False)
            if len(interim_df) >= 3246:
                print(f"{INPUT_CSV} already contains {len(interim_df)} questions. Skipping preprocessing and loading.")
                skip_processing = True
        except Exception as e:
            print(f"Error reading {INPUT_CSV}: {e}. Proceeding with preprocessing.")

    if not skip_processing:
        clone_and_prepare()
        preprocess_datasets()

    # --- Step 2: Load only unprocessed questions for processing ---
    print("⏳ Loading and filtering data...")
    df = pd.read_csv(INPUT_CSV)[['id', 'question']]
    df = df.drop_duplicates(subset=['question'])  # Ensure no duplicate questions in input

    processed_questions = set()
    if os.path.exists(OUTPUT_CSV):
        try:
            existing_df = pd.read_csv(OUTPUT_CSV)
            if not existing_df.empty and 'question' in existing_df.columns:
                processed_questions = set(existing_df['question'].astype(str).str.strip().tolist())
                print(f"Loaded {len(processed_questions)} successfully processed questions from {OUTPUT_CSV}.")
        except pd.errors.EmptyDataError:
            print(f"Existing output file {OUTPUT_CSV} is empty. Starting from scratch.")
        except Exception as e:
            print(f"Error loading existing output file {OUTPUT_CSV}: {e}. Starting from scratch.")

    # Only load questions not already processed (by question text)
    df['question'] = df['question'].astype(str).str.strip()
    df_to_process = df[~df['question'].isin(processed_questions)].copy()
    skipped_rows = len(df) - len(df_to_process)
    if skipped_rows > 0:
        print(f"Skipping {skipped_rows} questions that have already been processed.")

    print(f"🚀 Starting processing for {len(df_to_process)} questions.")

    # --- Step 3: Model Setup and Rate Limit Handling ---
    API_KEYS = []
    current_key_index = 0
    def load_model_with_key(api_key):
        genai.configure(api_key=api_key)
        return genai.GenerativeModel(MODEL_NAME)
    model = load_model_with_key(API_KEYS[current_key_index])

    results = []
    batch_size = 5
    for i in range(0, len(df_to_process), batch_size):
        batch = df_to_process.iloc[i:i+batch_size]
        for index, row in batch.iterrows():
            question_id = row['id']
            question_text = row['question']  # Always use full question text
            prompt = build_prompt(question_text)

            rate_limit_retries = 0
            normal_retries = 0
            while normal_retries < MAX_RETRIES:
                try:
                    response = model.generate_content(prompt)
                    response_text = response.text

                    # Extract JSON from the response
                    json_start = response_text.find('{')
                    json_end = response_text.rfind('}') + 1
                    json_str = response_text[json_start:json_end]

                    data = json.loads(json_str)
                    data['id'] = question_id
                    data['question'] = question_text  # Ensure full question text is saved
                    results.append(data)
                    break  # Success, exit retry loop
                except Exception as e:
                    error_str = str(e).lower()
                    print(f"Error processing question {question_id} on attempt {normal_retries + 1}: {e}")
                    if ("rate limit" in error_str or "quota" in error_str or "429" in error_str):
                        rate_limit_retries += 1
                        if rate_limit_retries < 3:
                            print("Rate limit detected. Waiting 2 minutes before retrying...")
                            time.sleep(RETRY_DELAY*60)  # Wait 2 minutes
                        else:
                            print(f"Rate limit persists after 3 retries. Switching API key.")
                            current_key_index = (current_key_index + 1) % len(API_KEYS)
                            print(f"Switching to API key index {current_key_index}.")
                            model = load_model_with_key(API_KEYS[current_key_index])
                            rate_limit_retries = 0  # Reset rate limit retry count after switching key
                        continue  # Do not increment normal_retries on rate limit
                    else:
                        if normal_retries < MAX_RETRIES - 1:
                            print(f"Retrying in {RETRY_DELAY} seconds...")
                            time.sleep(RETRY_DELAY)
                        normal_retries += 1
                        if normal_retries == MAX_RETRIES:
                            print(f"Failed to process question {question_id} after {MAX_RETRIES} attempts.")

            time.sleep(0.5) # Add a delay to avoid hitting rate limits

        if results:
            # Filter out any results whose questions are already in the output CSV
            if os.path.exists(OUTPUT_CSV):
                try:
                    existing_df = pd.read_csv(OUTPUT_CSV)
                    existing_questions = set(existing_df['question'].astype(str).str.strip().tolist()) if not existing_df.empty and 'question' in existing_df.columns else set()
                except Exception as e:
                    print(f"Error reading {OUTPUT_CSV} for deduplication: {e}")
                    existing_questions = set()
            else:
                existing_questions = set()
            new_results = [r for r in results if str(r['question']).strip() not in existing_questions]
            if new_results:
                output_df = pd.DataFrame(new_results)
                write_header = not os.path.exists(OUTPUT_CSV) or os.stat(OUTPUT_CSV).st_size == 0
                output_df.to_csv(OUTPUT_CSV, mode='a', index=False, header=write_header)
                print(f"✅ Saved {len(new_results)} new entries to {OUTPUT_CSV}")
                # Deduplicate output after each batch
                deduplicate_csv_by_question(OUTPUT_CSV)
            else:
                print("No new unique entries to save in this batch.")
            results = []

    print("\n--- Processing Complete ---")
    print(f"Final output saved to {OUTPUT_CSV}")
    # Final deduplication
    deduplicate_csv_by_question(OUTPUT_CSV)