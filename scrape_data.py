import os
import json
import pandas as pd
import time

from utils.leetcode_api import get_problemset_api_query, get_problem_details_api_query, call_api
from utils.parser import parse_html_text, parse_html_hints

# --- Constants ---
QUESTIONS_LIST_PATH = 'data/questions_list.csv'
ALL_QUESTIONS_DETAILS_CUMULATIVE_PATH = 'data/all_questions_details_cumulative.csv' # New Path
# ALL_QUESTIONS_DATA_DIR = 'data/all_questions_data' # No longer primary for detailed data output
TARGET_TOTAL_QUESTIONS = 10000 # This remains the ideal target for the list
QUESTIONS_LIST_CHUNK_SIZE = 500
MAX_DETAILS_TO_FETCH_PER_RUN = 100 # Limit for detailed data fetching per run

# Global flag declaration
API_EXHAUSTED_FLAG = False # Will be properly managed in main and other functions via 'global'

def create_questions_list(output_path, offset, limit, master_processed_ids_set):
    """
    Fetches a chunk of question summaries from the API and appends only new, unique questions to the output_path.
    Args:
        output_path (str): Path to the CSV file to append to.
        offset (int): The API skip parameter.
        limit (int): The API limit parameter.
        master_processed_ids_set (set): A set of already processed 'Question ID' strings.
                                        This function will check against this set and also update it
                                        with newly added question IDs.
    Returns:
        int: The number of newly added unique questions to the CSV.
    """
    print(f"Fetching question list chunk: API offset={offset}, limit={limit}")
    problemset_query = get_problemset_api_query(skip=offset, limit=limit)
    problemset_data = call_api(problemset_query)

    if not problemset_data or 'problemsetQuestionList' not in problemset_data or not problemset_data['problemsetQuestionList']['questions']:
        print("No more questions found or error in API response for problemsetQuestionList.")
        return 0

    problemset = problemset_data['problemsetQuestionList']['questions']
    questions_in_chunk = len(problemset)
    print(f"Received {questions_in_chunk} question summaries in this chunk.")

    if questions_in_chunk == 0:
        return 0

    file_exists = os.path.exists(output_path)
    is_empty_file = file_exists and os.path.getsize(output_path) == 0

    newly_added_questions_in_chunk = []
    newly_added_count = 0

    for problem in problemset:
        question_id_str = str(problem.get('frontendQuestionId')) # Ensure ID is string
        if question_id_str not in master_processed_ids_set:
            prob_dict = {}
            prob_dict['Question ID'] = question_id_str
            prob_dict['Question Title'] = problem.get('title')
            prob_dict['Question Slug'] = problem.get('titleSlug')
            tags = problem.get('topicTags', [])
            prob_dict['Topic Tagged text'] = ",".join([tag.get('name', '') for tag in tags])
            prob_dict['Topic Tagged ID'] = ",".join([tag.get('id', '') for tag in tags])
            newly_added_questions_in_chunk.append(prob_dict)
            master_processed_ids_set.add(question_id_str) # Update the master set
            newly_added_count += 1
        # else:
            # print(f"Skipping already processed Question ID: {question_id_str}") # Optional: for verbose logging

    if not newly_added_questions_in_chunk:
        print(f"No new unique questions to add from this API chunk (total received: {questions_in_chunk}).")
        # Return total questions *received from API* so generate_question_list_resumable can decide if API is exhausted
        return 0, questions_in_chunk # (newly_added_count, total_received_from_api_for_this_chunk)


    chunk_df = pd.DataFrame(newly_added_questions_in_chunk)

    try:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        write_header = not file_exists or is_empty_file
        with open(output_path, 'a', newline='', encoding='utf-8') as f:
            chunk_df.to_csv(f, header=write_header, index=False)
        print(f"Successfully appended {len(chunk_df)} new unique rows to {output_path}")
    except Exception as e:
        print(f"Error writing to CSV {output_path}: {e}")
        # If write fails, we should ideally revert additions to master_processed_ids_set for this chunk,
        # but that's complex. For now, we'll assume write is usually successful if DataFrame is formed.
        # Returning 0 new indicates failure to persist.
        return 0, questions_in_chunk

    # Return count of NEWLY ADDED questions, and total received from API for exhaustion check
    return newly_added_count, questions_in_chunk

def generate_question_list_resumable():
    # This function remains largely unchanged
    global API_EXHAUSTED_FLAG
    if not os.path.exists('data'):
        os.makedirs('data')
        print("Created directory: data")

    master_processed_ids = set()
    current_unique_question_count = 0 # Tracks unique questions based on IDs in the CSV

    if os.path.exists(QUESTIONS_LIST_PATH) and os.path.getsize(QUESTIONS_LIST_PATH) > 0:
        try:
            df_existing = pd.read_csv(QUESTIONS_LIST_PATH, dtype={'Question ID': str})
            if not df_existing.empty and 'Question ID' in df_existing.columns:
                master_processed_ids = set(df_existing['Question ID'].astype(str).tolist())
                current_unique_question_count = len(master_processed_ids)
                print(f"Resuming: Found {current_unique_question_count} unique question IDs in {QUESTIONS_LIST_PATH}.")
            else: # File exists but is empty or malformed
                print(f"{QUESTIONS_LIST_PATH} is empty or malformed. Starting fresh for list generation logic.")
        except Exception as e:
            print(f"Error reading {QUESTIONS_LIST_PATH} for initial ID set: {e}. Starting fresh.")
            # If read fails, ensure QUESTIONS_LIST_PATH is treated as new for header writing in create_questions_list
            if os.path.exists(QUESTIONS_LIST_PATH): os.remove(QUESTIONS_LIST_PATH) # Risky, but might be needed if file is corrupt

    api_skip_offset = current_unique_question_count # API skip should be based on unique items we think we have

    print(f"Target unique questions for list: {TARGET_TOTAL_QUESTIONS}")

    if current_unique_question_count < TARGET_TOTAL_QUESTIONS:
        print(f"\nCurrent unique list progress: {current_unique_question_count}/{TARGET_TOTAL_QUESTIONS}.")

        # Determine how many to fetch in this run for the API call
        # This is an attempt to reach TARGET_TOTAL_QUESTIONS unique entries.
        # The API might return duplicates or fewer than requested.
        # We use QUESTIONS_LIST_CHUNK_SIZE as the *API limit*, not necessarily how many new ones we'll get.

        # The loop condition should be based on current_unique_question_count vs TARGET_TOTAL_QUESTIONS
        # For a single run, we'll just do one chunk fetch.

        remaining_to_target_approx = TARGET_TOTAL_QUESTIONS - current_unique_question_count
        api_limit_for_this_call = min(QUESTIONS_LIST_CHUNK_SIZE, remaining_to_target_approx if remaining_to_target_approx > 0 else QUESTIONS_LIST_CHUNK_SIZE)


        if api_limit_for_this_call > 0 : # Only proceed if we intend to fetch
            print(f"Attempting to fetch up to {api_limit_for_this_call} question summaries, API skip offset {api_skip_offset}.")

            newly_added_this_call, api_returned_count = create_questions_list(
                output_path=QUESTIONS_LIST_PATH,
                offset=api_skip_offset, # Use the count of unique IDs as the basis for API skip
                limit=api_limit_for_this_call,
                master_processed_ids_set=master_processed_ids # Pass the live set
            )

            current_unique_question_count += newly_added_this_call # Update count with actually new items
            # api_skip_offset += api_returned_count # The API skip for *next* time should advance by what API returned
                                                # This is now implicitly handled by re-calculating current_unique_question_count
                                                # and using that as api_skip_offset at the start of the *next run*.

            print(f"After this call: newly added unique: {newly_added_this_call}, API returned: {api_returned_count}. Total unique in list now: {current_unique_question_count}")

            if api_returned_count == 0:
                print("API returned 0 items. Definitively exhausted for list summaries. Setting API_EXHAUSTED_FLAG.")
                API_EXHAUSTED_FLAG = True
            elif newly_added_this_call == 0 and api_returned_count > 0:
                print(f"API returned {api_returned_count} items, but none were new. Temporarily setting API_EXHAUSTED_FLAG to test Part 2.")
                API_EXHAUSTED_FLAG = True # TEMPORARY for testing Part 2
            elif api_returned_count < api_limit_for_this_call:
                 print("API returned fewer items than requested. Assuming end of list for summaries. Setting API_EXHAUSTED_FLAG.")
                 API_EXHAUSTED_FLAG = True
        else:
            print("Calculated API limit is 0 or negative. Not fetching summaries.")
            if current_unique_question_count >= TARGET_TOTAL_QUESTIONS:
                print("Target for question list already met or exceeded.")
            else: # Should not happen if remaining_to_target_approx was positive
                print("Logic error in chunk size calculation, or target met.")

    else: # current_unique_question_count >= TARGET_TOTAL_QUESTIONS
        print(f"Question list (unique IDs: {current_unique_question_count}) already meets or exceeds target {TARGET_TOTAL_QUESTIONS}.")

    final_list_rows = 0
    final_list_unique_ids = 0
    if os.path.exists(QUESTIONS_LIST_PATH) and os.path.getsize(QUESTIONS_LIST_PATH) > 0:
        try:
            df_qlist_final = pd.read_csv(QUESTIONS_LIST_PATH, dtype={'Question ID': str})
            final_list_rows = len(df_qlist_final)
            if 'Question ID' in df_qlist_final.columns:
                final_list_unique_ids = df_qlist_final['Question ID'].nunique()
            else:
                final_list_unique_ids = final_list_rows
        except Exception as e:
            print(f"Error reading final {QUESTIONS_LIST_PATH} for stats: {e}")

    print(f"Finished current run of generating question summary list. Total rows in {QUESTIONS_LIST_PATH}: {final_list_rows}. Total unique IDs: {final_list_unique_ids}.")
    
    # Set API_EXHAUSTED_FLAG if we haven't reached target but didn't add new ones and API didn't give full chunk
    # This logic is now partially inside the if api_limit_for_this_call > 0 block.
    # Let's consolidate: if we tried to fetch and got nothing new or indication of end, set flag.
    if current_unique_question_count < TARGET_TOTAL_QUESTIONS and \
       'newly_added_this_call' in locals() and \
       newly_added_this_call == 0 and \
       ('api_returned_count' not in locals() or api_returned_count < QUESTIONS_LIST_CHUNK_SIZE) : # Check if api_returned_count exists
        # If we didn't add new items AND (either api_returned_count wasn't set (error) or it was less than a full chunk)
        # This condition is a bit complex, the one inside the loop is more direct.
        # API_EXHAUSTED_FLAG might already be True from within the loop.
        if not API_EXHAUSTED_FLAG: # Avoid redundant message if already set
            print("After list generation run, target not met and no new items added / partial API response. Considering API exhausted for summaries.")
            API_EXHAUSTED_FLAG = True


def create_all_questions_data_single(question_row, ques_df_for_similar):
    # This function (fetching details for ONE question) remains largely unchanged,
    # including the fix for the NoneType error.
    q_id_str = str(question_row['Question ID']).strip()
    q_slug = str(question_row['Question Slug']).strip()
    # print(f"Loading full details for... ID: {q_id_str} Slug: {q_slug}") # Making this less verbose for cumulative runs

    q_dict = {}
    # Basic info from questions_list.csv
    q_dict['Question ID'] = q_id_str # Keep as string for now, convert to int in final combine if needed
    q_dict['Question Title'] = question_row['Question Title']
    q_dict['Question Slug'] = q_slug
    q_dict['Topic Tagged text'] = question_row.get('Topic Tagged text', '')
    q_dict['Topic Tagged ID'] = question_row.get('Topic Tagged ID', '')

    query = get_problem_details_api_query(q_slug)
    time.sleep(0.2)
    data_api_response = call_api(query)

    if not data_api_response or 'question' not in data_api_response:
        print(f"Could not fetch details for {q_slug} (no 'question' field in response or empty response). Skipping.")
        return None

    data = data_api_response['question']
    if data is None:
        print(f"Details for {q_slug} contained a null 'question' object. Skipping.")
        return None

    q_dict['Question Text'] = parse_html_text(data.get('content'))
    q_dict['Difficulty Level'] = data.get('difficulty', 'Unknown')
    stats_str = data.get('stats', '{}')
    try:
        stats = json.loads(stats_str)
        raw_ac_rate = stats.get('acRate', '0%')
        q_dict['Success Rate'] = float(raw_ac_rate.replace('%', '')) if '%' in raw_ac_rate else float(raw_ac_rate)
        q_dict['total submission'] = int(stats.get('totalSubmissionRaw', 0))
        q_dict['total accepted'] = int(stats.get('totalAcceptedRaw', 0))
    except (json.JSONDecodeError, ValueError) as e:
        print(f"Error parsing stats for {q_slug}: {e}")
        q_dict['Success Rate'] = 0.0; q_dict['total submission'] = 0; q_dict['total accepted'] = 0

    q_dict['company tag'] = data.get('companyTagStats', "None")
    if q_dict['company tag'] is None: q_dict['company tag'] = "None"
    q_dict['Likes'] = data.get('likes', 0)
    q_dict['Dislikes'] = data.get('dislikes', 0)
    q_dict['Hints'] = parse_html_hints(data.get('hints', []))

    def get_similar_questions_details(all_ques_df, similar_ques_json_str, current_slug):
        # This helper remains as is
        que_text_list = []
        que_ids = []
        if not similar_ques_json_str: return "", ""
        try: similar_ques_list = json.loads(similar_ques_json_str)
        except json.JSONDecodeError: print(f"Error decoding similar questions JSON for {current_slug}"); return "", ""
        for sq in similar_ques_list:
            sq_slug = sq.get('titleSlug')
            if not sq_slug: continue
            if 'Question Slug' not in all_ques_df.columns: continue
            related_que = all_ques_df[all_ques_df['Question Slug'].astype(str) == str(sq_slug)]
            if not related_que.empty:
                if 'Question ID' in related_que.columns: que_ids.append(str(related_que['Question ID'].iloc[0]))
                if 'Question Title' in related_que.columns: que_text_list.append(str(related_que['Question Title'].iloc[0]))
        return ",".join(que_ids), ",".join(que_text_list)

    similar_ques_str = data.get('similarQuestions', '[]')
    q_dict['Similar Questions ID'], q_dict['Similar Questions Text'] = get_similar_questions_details(ques_df_for_similar, similar_ques_str, q_slug)

    return q_dict

def generate_all_question_data_resumable():
    """
    Generates detailed data for questions from questions_list.csv and appends
    them to a single cumulative CSV file: ALL_QUESTIONS_DETAILS_CUMULATIVE_PATH.
    Skips questions if their details are already in the cumulative CSV.
    Processes a limited number of questions per run.
    """
    if not os.path.exists(QUESTIONS_LIST_PATH) or os.path.getsize(QUESTIONS_LIST_PATH) == 0:
        print(f"{QUESTIONS_LIST_PATH} not found or is empty. Please generate the question list first.")
        return

    try:
        ques_df_all = pd.read_csv(QUESTIONS_LIST_PATH, dtype={'Question ID': str})
    except Exception as e:
        print(f"Error reading {QUESTIONS_LIST_PATH}: {e}. Cannot proceed.")
        return

    if ques_df_all.empty:
        print(f"{QUESTIONS_LIST_PATH} is empty. Nothing to process for detailed data.")
        return

    processed_ids = set()
    cumulative_file_exists = os.path.exists(ALL_QUESTIONS_DETAILS_CUMULATIVE_PATH)
    is_empty_cumulative_file = cumulative_file_exists and os.path.getsize(ALL_QUESTIONS_DETAILS_CUMULATIVE_PATH) == 0

    if cumulative_file_exists and not is_empty_cumulative_file:
        try:
            df_cumulative_details = pd.read_csv(ALL_QUESTIONS_DETAILS_CUMULATIVE_PATH, dtype={'Question ID': str})
            if 'Question ID' in df_cumulative_details.columns:
                # Ensure IDs from CSV are strings for consistent set comparison
                processed_ids = set(df_cumulative_details['Question ID'].astype(str).tolist())
            print(f"Resuming detailed data generation: Found {len(processed_ids)} already processed questions in {ALL_QUESTIONS_DETAILS_CUMULATIVE_PATH}.")
        except pd.errors.EmptyDataError:
             print(f"{ALL_QUESTIONS_DETAILS_CUMULATIVE_PATH} is empty. Starting detail fetch from scratch.")
             processed_ids = set()
             cumulative_file_exists = False # Will need header
             is_empty_cumulative_file = True
        except Exception as e:
            print(f"Error reading {ALL_QUESTIONS_DETAILS_CUMULATIVE_PATH} for processed IDs: {e}. Treating as new/empty.")
            processed_ids = set()
            cumulative_file_exists = False # Will need header
            is_empty_cumulative_file = True
    else:
        print(f"{ALL_QUESTIONS_DETAILS_CUMULATIVE_PATH} does not exist or is empty. Starting fresh.")
        cumulative_file_exists = False # Will need header
        is_empty_cumulative_file = True


    print(f"Starting generation of detailed question data, appending to {ALL_QUESTIONS_DETAILS_CUMULATIVE_PATH}...")

    chunk_details_list = []
    newly_fetched_this_run = 0
    skipped_due_to_already_processed = 0
    iterated_in_list = 0
    questions_to_process_count = 0 # Count of questions not in processed_ids

    # First pass to count how many questions we actually plan to process in this run
    for _, question_row in ques_df_all.iterrows():
        question_id_str = str(question_row['Question ID']).strip()
        if question_id_str not in processed_ids:
            questions_to_process_count +=1

    print(f"Total questions in list: {len(ques_df_all)}. Already processed: {len(processed_ids)}. Remaining to process: {questions_to_process_count}.")


    for index, question_row in ques_df_all.iterrows():
        iterated_in_list += 1
        question_id_str = str(question_row['Question ID']).strip()

        if not question_id_str or not question_row.get('Question Slug'):
            print(f"Skipping row {index} from questions_list.csv due to missing Question ID or Slug.")
            continue

        if question_id_str in processed_ids:
            skipped_due_to_already_processed += 1
            continue

        if newly_fetched_this_run >= MAX_DETAILS_TO_FETCH_PER_RUN:
            print(f"Reached fetch limit for this run ({MAX_DETAILS_TO_FETCH_PER_RUN}). Will save current chunk and continue in the next run.")
            break

        if (newly_fetched_this_run + 1) % 20 == 0 or newly_fetched_this_run == 0 : # Log more frequently
             print(f"Fetching details for {question_id_str}... ({newly_fetched_this_run + 1}/{min(MAX_DETAILS_TO_FETCH_PER_RUN, questions_to_process_count - skipped_due_to_already_processed)} in this batch)")

        detailed_data = create_all_questions_data_single(question_row, ques_df_all)
        if detailed_data:
            chunk_details_list.append(detailed_data)
            newly_fetched_this_run += 1

    if chunk_details_list:
        print(f"Attempting to append {len(chunk_details_list)} new detailed question data rows to {ALL_QUESTIONS_DETAILS_CUMULATIVE_PATH}")
        try:
            details_df_chunk = pd.DataFrame(chunk_details_list)
            os.makedirs(os.path.dirname(ALL_QUESTIONS_DETAILS_CUMULATIVE_PATH), exist_ok=True)

            write_header_cumulative = not cumulative_file_exists or is_empty_cumulative_file

            # Ensure consistent column order based on a sample dict or predefined list
            if not details_df_chunk.empty:
                # Define expected columns based on q_dict structure in create_all_questions_data_single
                # This ensures consistent order and handles missing keys by filling with NaN (which to_csv handles)
                expected_cols = [
                    'Question ID', 'Question Title', 'Question Slug', 'Topic Tagged text', 'Topic Tagged ID',
                    'Question Text', 'Difficulty Level', 'Success Rate', 'total submission', 'total accepted',
                    'company tag', 'Likes', 'Dislikes', 'Hints', 'Similar Questions ID', 'Similar Questions Text'
                ]
                details_df_chunk = details_df_chunk.reindex(columns=expected_cols)

            with open(ALL_QUESTIONS_DETAILS_CUMULATIVE_PATH, 'a', newline='', encoding='utf-8') as f:
                details_df_chunk.to_csv(f, header=write_header_cumulative, index=False)
            print(f"Successfully appended {len(details_df_chunk)} rows to {ALL_QUESTIONS_DETAILS_CUMULATIVE_PATH}")
        except Exception as e:
            print(f"Error writing chunk to {ALL_QUESTIONS_DETAILS_CUMULATIVE_PATH}: {e}")
    else:
        print("No new details fetched in this run to append to cumulative CSV.")

    total_in_cumulative = 0
    if os.path.exists(ALL_QUESTIONS_DETAILS_CUMULATIVE_PATH) and os.path.getsize(ALL_QUESTIONS_DETAILS_CUMULATIVE_PATH) > 0:
        try:
            df_final_cumulative = pd.read_csv(ALL_QUESTIONS_DETAILS_CUMULATIVE_PATH)
            total_in_cumulative = len(df_final_cumulative)
        except: pass # Ignore if read fails, just won't have the count

    print(f"Finished current run of generating detailed question data. Iterated in list: {iterated_in_list}. Newly fetched for CSV: {newly_fetched_this_run}. Skipped (already in CSV): {skipped_due_to_already_processed}. Total in cumulative CSV: {total_in_cumulative}")


def main():
    print("Starting data scraping process...")
    global API_EXHAUSTED_FLAG
    API_EXHAUSTED_FLAG = False  # Initialize after global declaration for this run's context

    # --- De-duplicate questions_list.csv if it exists before any other operations ---
    if os.path.exists(QUESTIONS_LIST_PATH) and os.path.getsize(QUESTIONS_LIST_PATH) > 0:
        print(f"Attempting to de-duplicate {QUESTIONS_LIST_PATH} by 'Question ID' and 'Question Slug'...")
        try:
            # Use low_memory=False if there are mixed types issues, though Question ID should be consistent.
            df_qlist = pd.read_csv(QUESTIONS_LIST_PATH, dtype={'Question ID': str, 'Question Slug': str})
            initial_count = len(df_qlist)

            # Ensure columns are definitely strings for robust de-duplication
            df_qlist['Question ID'] = df_qlist['Question ID'].astype(str)
            df_qlist['Question Slug'] = df_qlist['Question Slug'].astype(str) # Added for robustness

            # Drop duplicates based on 'Question ID', keeping the first occurrence.
            # Consider 'Question Slug' as well if IDs might not be globally unique (though for LeetCode, ID should be).
            df_qlist.drop_duplicates(subset=['Question ID', 'Question Slug'], keep='first', inplace=True)

            deduplicated_count = len(df_qlist)
            if initial_count > deduplicated_count:
                df_qlist.to_csv(QUESTIONS_LIST_PATH, index=False)
                print(f"De-duplication of {QUESTIONS_LIST_PATH} complete. Original rows: {initial_count}, Unique rows: {deduplicated_count}. File overwritten.")
            else:
                print(f"No duplicates found in {QUESTIONS_LIST_PATH} based on 'Question ID' and 'Question Slug'. Original rows: {initial_count}.")
        except Exception as e:
            print(f"Error during initial de-duplication of {QUESTIONS_LIST_PATH}: {e}")
    else:
        print(f"{QUESTIONS_LIST_PATH} does not exist or is empty. Skipping initial de-duplication.")

    print("\n--- Part 1: Generating Question List (resumable) ---")
    generate_question_list_resumable() # This function should now work with a cleaner list if it existed

    proceed_to_details = False
    num_questions_in_list = 0
    num_unique_questions_in_list = 0

    if os.path.exists(QUESTIONS_LIST_PATH) and os.path.getsize(QUESTIONS_LIST_PATH) > 0:
        try:
            df_qlist = pd.read_csv(QUESTIONS_LIST_PATH, dtype={'Question ID': str})
            num_questions_in_list = len(df_qlist) # Total rows
            if 'Question ID' in df_qlist.columns:
                 num_unique_questions_in_list = df_qlist['Question ID'].nunique()
            else:
                 num_unique_questions_in_list = num_questions_in_list # Fallback if no ID column

            print(f"\nAfter Part 1, {QUESTIONS_LIST_PATH} has {num_questions_in_list} total rows and {num_unique_questions_in_list} unique Question IDs.")

            if num_unique_questions_in_list >= TARGET_TOTAL_QUESTIONS:
                 print(f"Question list (unique IDs: {num_unique_questions_in_list}) has reached the target {TARGET_TOTAL_QUESTIONS}.")
                 proceed_to_details = True
            elif API_EXHAUSTED_FLAG:
                 print(f"Question list (unique IDs: {num_unique_questions_in_list}) - API reported no more new questions for the list.")
                 proceed_to_details = True
            else:
                 print(f"Question list (unique IDs: {num_unique_questions_in_list}) is not yet complete (target: {TARGET_TOTAL_QUESTIONS}) and API might still have questions for the list.")
                 print("Focus on completing Part 1 by re-running the script.")
        except Exception as e:
            print(f"Could not properly assess {QUESTIONS_LIST_PATH} after Part 1 due to {e}. Will not proceed to details generation.")
    else:
        print(f"\n{QUESTIONS_LIST_PATH} does not exist or is empty after Part 1. Focus on Part 1 first.")

    if proceed_to_details:
        print("\n--- Part 2: Generating Detailed Question Data (resumable, cumulative CSV) ---")
        generate_all_question_data_resumable()
    else:
        print("\nSkipping Part 2 (Detailed Question Data) for this run.")

    print("\nScript execution finished for this run.")

if __name__ == "__main__":
    main()
