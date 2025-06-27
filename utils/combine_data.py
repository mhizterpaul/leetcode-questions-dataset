""" This file is used to combine all individual questions data file.
    It creates one main file 'all_questions.csv' after removing duplicates.
"""

import os
import sys
import pandas as pd

def main():
    # Construct ROOT_DIR relative to this script's location
    # __file__ is 'utils/combine_data.py'
    # os.path.dirname(__file__) is 'utils'
    # os.path.dirname(os.path.dirname(__file__)) is the ROOT_DIR
    script_dir = os.path.dirname(os.path.abspath(__file__))
    ROOT_DIR = os.path.dirname(script_dir)

    print(f"Project Root Directory: {ROOT_DIR}")
    # No need to append to sys.path if imports are relative or handled by PYTHONPATH

    DATA_DIR = os.path.join(ROOT_DIR, 'data', 'all_questions_data')
    OUTPUT_FILENAME = os.path.join(ROOT_DIR, 'data', 'all_questions.csv') # Changed output filename

    if not os.path.exists(DATA_DIR):
        print(f"Data directory {DATA_DIR} does not exist. Nothing to combine.")
        return

    all_files = [f for f in os.listdir(DATA_DIR) if f.startswith('all_ques_data_') and f.endswith('.csv')]

    if not all_files:
        print(f"No chunked CSV files found in {DATA_DIR}. Nothing to combine.")
        return

    all_data_list = []
    print("Reading chunked data files...")
    for file_ in all_files:
        file_path = os.path.join(DATA_DIR, file_)
        try:
            df = pd.read_csv(file_path)
            all_data_list.append(df)
            print(f"Read {file_path}, found {len(df)} rows.")
        except pd.errors.EmptyDataError:
            print(f"Warning: {file_path} is empty and will be skipped.")
        except Exception as e:
            print(f"Error reading {file_path}: {e}. Skipping this file.")

    if not all_data_list:
        print("No data loaded from chunk files. Output file will not be created.")
        return

    all_data = pd.concat(all_data_list, ignore_index=True)
    print(f"Total rows before duplicate removal: {len(all_data)}")

    # Remove duplicates based on 'Question ID'
    # Keep the first occurrence in case of duplicates
    if 'Question ID' in all_data.columns:
        all_data.drop_duplicates(subset=['Question ID'], keep='first', inplace=True)
        print(f"Total rows after duplicate removal (on 'Question ID'): {len(all_data)}")
    else:
        print("Warning: 'Question ID' column not found. Cannot remove duplicates.")

    # Sort data by 'Question ID' to maintain sequence as much as possible.
    # This assumes 'Question ID' can be reasonably sorted numerically.
    if 'Question ID' in all_data.columns:
        all_data['Question ID'] = pd.to_numeric(all_data['Question ID'], errors='coerce')
        all_data.sort_values(by='Question ID', inplace=True)
        print("Data sorted by 'Question ID'.")

    all_data.to_csv(OUTPUT_FILENAME, index=False)
    print(f"Successfully combined data into {OUTPUT_FILENAME}")
    print(f"Final dataset contains {len(all_data)} unique questions.")

if __name__ == '__main__':
    main()
