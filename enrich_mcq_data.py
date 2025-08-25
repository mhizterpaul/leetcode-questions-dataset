import os
import pandas as pd

# --- Constants ---
MCQ_CSV = "data/mcq_dataset.csv"
LEETCODE_CSV = "data/leetcode_dataset/all_questions_details_cumulative.csv"
INTERVIEW_CSV = "data/interview_dataset/SoftwareQuestions.csv"
OUTPUT_CSV = "data/mcq_dataset_enriched.csv"


def normalize_question(s):
    if isinstance(s, str):
        return ' '.join(s.lower().split())
    return ''


def main():
    # Load the datasets
    try:
        mcq_df = pd.read_csv(MCQ_CSV)
        leetcode_df = pd.read_csv(LEETCODE_CSV)
        interview_df = pd.read_csv(INTERVIEW_CSV, encoding='latin1')
    except FileNotFoundError as e:
        print(f"Error loading data: {e}. Make sure all source CSV files are present.")
        return

    # --- Prepare LeetCode data ---
    leetcode_df = leetcode_df[['Question Title', 'Question Text', 'Topic Tagged text', 'Difficulty Level']].rename(columns={
        'Question Title': 'title',
        'Question Text': 'body',
        'Topic Tagged text': 'tags',
        'Difficulty Level': 'difficulty'
    })
    leetcode_df['question'] = leetcode_df['title'] + "\n" + leetcode_df['body']
    leetcode_df['norm_question'] = leetcode_df['question'].apply(normalize_question)

    # Let's keep only the columns we need to avoid confusion
    leetcode_df = leetcode_df[['norm_question', 'difficulty', 'tags']]
    # Rename columns to avoid conflicts after merge
    leetcode_df.rename(columns={'difficulty': 'difficulty_leetcode', 'tags': 'tags_leetcode'}, inplace=True)


    # --- Prepare Interview data ---
    interview_df = interview_df[['Question', 'Category', 'Difficulty']].rename(columns={
        'Question': 'question',
        'Category': 'category',
        'Difficulty': 'difficulty'
    })
    interview_df['norm_question'] = interview_df['question'].apply(normalize_question)
    # Let's keep only the columns we need
    interview_df = interview_df[['norm_question', 'difficulty', 'category']]
    # Rename columns to avoid conflicts
    interview_df.rename(columns={'difficulty': 'difficulty_interview', 'category': 'category_interview'}, inplace=True)


    # --- Prepare mcq_dataset data ---
    mcq_df['norm_question'] = mcq_df['question'].apply(normalize_question)

    # --- Merge data ---
    # Merge with LeetCode data
    merged_df = pd.merge(mcq_df, leetcode_df, on='norm_question', how='left')

    # Merge with Interview data
    merged_df = pd.merge(merged_df, interview_df, on='norm_question', how='left')

    # --- Consolidate columns ---
    # Consolidate difficulty
    merged_df['difficulty'] = merged_df['difficulty_leetcode'].fillna(merged_df['difficulty_interview'])
    merged_df['difficulty'] = merged_df['difficulty'].fillna(merged_df['difficulty'])


    # Consolidate tags
    merged_df['tags'] = merged_df['tags_leetcode'].fillna(merged_df['tags'])

    # Consolidate category
    merged_df['category'] = merged_df['category_interview'].fillna(merged_df['category'])

    # --- Final cleanup ---
    # Drop temporary and redundant columns
    merged_df.drop(columns=['norm_question', 'difficulty_leetcode', 'difficulty_interview',
                            'tags_leetcode', 'category_interview'], inplace=True, errors='ignore')

    # Reorder columns to be more logical
    final_cols = ['id', 'question', 'a', 'b', 'c', 'd', 'correct', 'difficulty', 'category', 'tags']
    # Add any missing columns with None
    for col in final_cols:
        if col not in merged_df.columns:
            merged_df[col] = None
    merged_df = merged_df[final_cols]

    # Save the enriched dataset
    merged_df.to_csv(OUTPUT_CSV, index=False)
    print(f"Enriched dataset saved to {OUTPUT_CSV}")


if __name__ == "__main__":
    main()
