import os
import pandas as pd
import json
import time

# --- Constants ---
DATA_DIR_ROOT = "data" # Root data directory
QUESTIONS_LIST_PATH = os.path.join(DATA_DIR_ROOT, 'questions_list.csv')
ALL_QUESTIONS_DETAILS_CUMULATIVE_PATH = os.path.join(DATA_DIR_ROOT, 'all_questions_details_cumulative.csv')
FINAL_OUTPUT_CSV_PATH = os.path.join(DATA_DIR_ROOT, 'all_questions.csv')
LOG_FILE = os.path.join(DATA_DIR_ROOT, 'combine_data_log.txt')

# --- Logging ---
def log_message(message):
    """Appends a message to the log file and prints it."""
    timestamp = time.strftime('%Y-%m-%d %H:%M:%S')
    full_message = f"{timestamp} - {message}"
    with open(LOG_FILE, "a", encoding='utf-8') as f:
        f.write(full_message + "\n")
    print(full_message)

def main():
    log_message("Starting data combination process...")

    # --- 1. Load Data ---
    log_message(f"Loading question list from: {QUESTIONS_LIST_PATH}")
    if not os.path.exists(QUESTIONS_LIST_PATH):
        log_message(f"ERROR: {QUESTIONS_LIST_PATH} not found. Exiting.")
        return
    try:
        df_qlist = pd.read_csv(QUESTIONS_LIST_PATH, dtype={'Question ID': str, 'Question Slug': str})
        log_message(f"Loaded {len(df_qlist)} rows from {QUESTIONS_LIST_PATH}.")
        # Ensure critical columns exist
        if 'Question ID' not in df_qlist.columns or 'Question Slug' not in df_qlist.columns:
            log_message("ERROR: 'Question ID' or 'Question Slug' missing from questions_list.csv. Exiting.")
            return
    except Exception as e:
        log_message(f"ERROR: Could not load {QUESTIONS_LIST_PATH}: {e}. Exiting.")
        return

    log_message(f"Loading detailed questions data from: {ALL_QUESTIONS_DETAILS_CUMULATIVE_PATH}")
    if not os.path.exists(ALL_QUESTIONS_DETAILS_CUMULATIVE_PATH):
        log_message(f"ERROR: {ALL_QUESTIONS_DETAILS_CUMULATIVE_PATH} not found. Exiting.")
        return
    try:
        # df_details columns are: 'Question ID', 'Question Title', 'Question Slug', ...
        df_details = pd.read_csv(ALL_QUESTIONS_DETAILS_CUMULATIVE_PATH, keep_default_na=False, na_values=[''])
        log_message(f"Loaded {len(df_details)} rows from {ALL_QUESTIONS_DETAILS_CUMULATIVE_PATH}.")

        # Standardize column names for df_details immediately after loading
        # The cumulative CSV uses "Question ID" and "Question Slug"
        df_details_rename_map = {}
        if 'Question ID' in df_details.columns:
            df_details_rename_map['Question ID'] = 'details_question_id'
        if 'Question Slug' in df_details.columns:
            df_details_rename_map['Question Slug'] = 'details_question_slug'
        # Add other detail column renames as needed, e.g.:
        if 'Question Text' in df_details.columns: # This is the main content field from cumulative
            df_details_rename_map['Question Text'] = 'details_content_text'
        if 'Difficulty Level' in df_details.columns:
            df_details_rename_map['Difficulty Level'] = 'details_difficulty'
        if 'Likes' in df_details.columns:
            df_details_rename_map['Likes'] = 'details_likes'
        if 'Dislikes' in df_details.columns:
            df_details_rename_map['Dislikes'] = 'details_dislikes'
        if 'Hints' in df_details.columns:
            df_details_rename_map['Hints'] = 'details_hints_text' # Assuming it's already parsed text
        # ... and so on for other columns from all_questions_details_cumulative.csv that you want to use directly

        if df_details_rename_map:
            df_details.rename(columns=df_details_rename_map, inplace=True)
            log_message(f"Renamed columns in df_details for clarity: {df_details.columns.tolist()}")

        if 'details_question_id' not in df_details.columns or 'details_question_slug' not in df_details.columns:
            log_message("ERROR: 'details_question_id' or 'details_question_slug' (renamed from 'Question ID'/'Question Slug') missing from all_questions_details_cumulative.csv. Exiting.")
            return

    except Exception as e:
        log_message(f"ERROR: Could not load or process {ALL_QUESTIONS_DETAILS_CUMULATIVE_PATH}: {e}. Exiting.")
        return

    # --- 2. Data Type Consistency for Merge ---
    # df_qlist 'Question ID' is already str. df_details 'details_question_id' needs to be str.
    df_details['details_question_id'] = df_details['details_question_id'].astype(str)


    # --- 3. Merging DataFrames ---
    log_message(f"Merging DataFrames. Left table: df_qlist ({len(df_qlist)} rows), Right table: df_details ({len(df_details)} rows).")

    # De-duplicate df_details on 'details_question_id' before merge, keeping the first.
    df_details.drop_duplicates(subset=['details_question_id'], keep='first', inplace=True)
    log_message(f"df_details de-duplicated to {len(df_details)} rows on 'details_question_id' before merging.")

    log_message(f"df_qlist columns before merge: {df_qlist.columns.tolist()}")
    log_message(f"df_details columns before merge: {df_details.columns.tolist()}")

    # Perform the merge using 'Question ID' from df_qlist and 'details_question_id' from df_details
    df_merged = pd.merge(df_qlist, df_details,
                         left_on='Question ID', right_on='details_question_id',
                         how='left', suffixes=('_qlist', '_details')) # _qlist for original df_qlist cols, _details for df_details cols

    log_message(f"Merged DataFrame shape: {df_merged.shape}")
    log_message(f"df_merged columns after merge: {df_merged.columns.tolist()}")

    # --- 4. Column Selection, Parsing, and Final Naming ---
    final_df = pd.DataFrame()
    final_df['Question ID'] = df_merged['Question ID'] # This is from df_qlist, which is the master ID

    # Check if 'Question Title' was suffixed or not
    if 'Question Title_qlist' in df_merged.columns:
        final_df['Question Title'] = df_merged['Question Title_qlist']
    elif 'Question Title' in df_merged.columns:
        final_df['Question Title'] = df_merged['Question Title']
    else:
        log_message("ERROR: 'Question Title' or 'Question Title_qlist' not found in df_merged. Exiting.")
        return

    final_df['Question Slug'] = df_merged['Question Slug']   # From df_qlist

    # Topic Tags from questions_list.csv
    final_df['Topic Tagged text'] = df_merged['Topic Tagged text_qlist'] # Explicitly use from qlist
    final_df['Topic Tagged ID'] = df_merged['Topic Tagged ID_qlist']   # Explicitly use from qlist

    # Details from all_questions_details_cumulative.csv (now with _details suffix or specific renames)
    final_df['Question Text'] = df_merged['details_content_text'].fillna("") # Use the renamed column
    final_df['Difficulty Level'] = df_merged['details_difficulty']
    final_df['Likes'] = pd.to_numeric(df_merged['details_likes'], errors='coerce').fillna(0).astype(int)
    final_df['Dislikes'] = pd.to_numeric(df_merged['details_dislikes'], errors='coerce').fillna(0).astype(int)

    # The cumulative CSV has these directly, no need to parse JSON for them
    final_df['Success Rate'] = pd.to_numeric(df_merged['Success Rate'], errors='coerce').fillna(0.0)
    final_df['Total Submissions'] = pd.to_numeric(df_merged['total submission'], errors='coerce').fillna(0).astype(int) # column name from header
    final_df['Total Accepted'] = pd.to_numeric(df_merged['total accepted'], errors='coerce').fillna(0).astype(int) # column name from header

    final_df['Company Tags'] = df_merged['company tag'].fillna("None") # column name from header
    final_df['Hints'] = df_merged['details_hints_text'].fillna("") # Use the renamed column

    # Similar Questions are directly in the cumulative CSV
    # These columns come from df_details and should not be suffixed if not present in df_qlist
    if 'Similar Questions ID' in df_merged.columns:
        final_df['Similar Questions ID'] = df_merged['Similar Questions ID'].fillna("")
    elif 'Similar Questions ID_details' in df_merged.columns: # Fallback if it was suffixed
        final_df['Similar Questions ID'] = df_merged['Similar Questions ID_details'].fillna("")
    else:
        final_df['Similar Questions ID'] = "" # Add as empty if missing
        log_message("Warning: 'Similar Questions ID' column missing from merged_df.")

    if 'Similar Questions Text' in df_merged.columns:
        final_df['Similar Questions Text'] = df_merged['Similar Questions Text'].fillna("")
    elif 'Similar Questions Text_details' in df_merged.columns: # Fallback if it was suffixed
        final_df['Similar Questions Text'] = df_merged['Similar Questions Text_details'].fillna("")
    else:
        final_df['Similar Questions Text'] = "" # Add as empty if missing
        log_message("Warning: 'Similar Questions Text' column missing from merged_df.")


    # Code Snippets - Assuming cumulative CSV does not have complex JSON for this. If it does, parsing needed.
    # For now, let's assume it might be missing or simple.
    if 'Code Snippets (Python3 or All)' in df_merged.columns: # Check if this column was carried over or created
         final_df['Code Snippets (Python3 or All)'] = df_merged['Code Snippets (Python3 or All)'].fillna("")
    else: # If not, means the detailed fetch for this via API (and subsequent JSON parsing) is not in all_questions_details_cumulative.csv
         final_df['Code Snippets (Python3 or All)'] = "" # Add as empty

    log_message(f"Final DataFrame columns: {final_df.columns.tolist()}")
    log_message(f"Final DataFrame shape before final de-duplication: {final_df.shape}")

    # --- 5. Final De-duplication and Sorting ---
    # Convert 'Question ID' to numeric for sorting, handling errors
    final_df['Question ID Numeric'] = pd.to_numeric(final_df['Question ID'], errors='coerce')
    final_df.dropna(subset=['Question ID Numeric'], inplace=True) # Remove rows where ID couldn't be numeric
    final_df['Question ID Numeric'] = final_df['Question ID Numeric'].astype(int)

    # De-duplicate based on the numeric Question ID
    initial_count_before_final_dedup = len(final_df)
    final_df.drop_duplicates(subset=['Question ID Numeric'], keep='first', inplace=True)
    deduplicated_count_final = len(final_df)
    if initial_count_before_final_dedup > deduplicated_count_final:
        log_message(f"Removed {initial_count_before_final_dedup - deduplicated_count_final} duplicates based on 'Question ID Numeric'.")

    # Sort by 'Question ID Numeric'
    final_df.sort_values(by='Question ID Numeric', inplace=True)

    # Select and order final columns (dropping the temporary numeric ID)
    final_columns_ordered = [
        'Question ID', 'Question Title', 'Question Slug',
        'Difficulty Level', 'Success Rate', 'Total Submissions', 'Total Accepted',
        'Likes', 'Dislikes',
        'Topic Tagged text', 'Topic Tagged ID',
        'Company Tags', 'Hints',
        'Similar Questions ID', 'Similar Questions Text',
        'Code Snippets (Python3 or All)', # Added this
        'Question Text' # Usually long, so kept at end
    ]
    # Ensure all selected columns exist, add any missing ones as empty if necessary before reindexing
    for col in final_columns_ordered:
        if col not in final_df.columns:
            final_df[col] = ""
            log_message(f"Warning: Column '{col}' was missing, added as empty.")

    final_df_output = final_df[final_columns_ordered]

    # --- 6. Saving Output ---
    try:
        os.makedirs(DATA_DIR_ROOT, exist_ok=True)
        final_df_output.to_csv(FINAL_OUTPUT_CSV_PATH, index=False, encoding='utf-8')
        log_message(f"Successfully combined data and saved to {FINAL_OUTPUT_CSV_PATH}. Final shape: {final_df_output.shape}")
    except Exception as e:
        log_message(f"ERROR: Could not save final CSV to {FINAL_OUTPUT_CSV_PATH}: {e}")

    log_message("Data combination process finished.")


# Helper for HTML parsing (can be moved to a utils file later if not already there)
def parse_html_text(html_content):
    if pd.isna(html_content) or not html_content:
        return ""
    from bs4 import BeautifulSoup # Local import if not already global
    soup = BeautifulSoup(html_content, 'html.parser')
    return soup.get_text()

def parse_html_hints(hints_list):
    if pd.isna(hints_list) or not hints_list:
        return ""
    parsed_hints = []
    for hint_html in hints_list:
        if hint_html:
             from bs4 import BeautifulSoup # Local import
             soup = BeautifulSoup(hint_html, 'html.parser')
             parsed_hints.append(soup.get_text())
    return "\n---\n".join(parsed_hints)


if __name__ == "__main__":
    # Clear log file for new run
    if os.path.exists(LOG_FILE):
        os.remove(LOG_FILE)
    main()
