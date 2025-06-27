""" Generate Data

    Two files will be generated:

    1. Questions List
    2. All Questions data (based on skip & limit)

"""
import os
import json
import pandas as pd
import csv # Added for row-by-row writing in create_questions_list

from utils.leetcode_api import get_problemset_api_query, get_problem_details_api_query, call_api
from utils.parser import parse_html_text, parse_html_hints

def create_questions_list(output_path):    
    problemset_query = get_problemset_api_query(skip=0, limit=10000)
    try:
        problemset_data = call_api(problemset_query)
        if not problemset_data or 'problemsetQuestionList' not in problemset_data or 'questions' not in problemset_data['problemsetQuestionList']:
            print("Error: Could not fetch problemset list or it's in an unexpected format.")
            return
        problemset = problemset_data['problemsetQuestionList']['questions']
    except Exception as e:
        print(f"Error calling API for problemset list: {e}")
        return # Cannot proceed without the initial list
    
    print(f'Generating.. question list. Found {len(problemset)} potential questions from API.')
    
    header = ['Question ID', 'Question Title', 'Question Slug', 'Topic Tagged text', 'Topic Tagged ID']
    
    file_exists = os.path.exists(output_path)
    processed_slugs = set()

    if file_exists:
        try:
            # Use low_memory=False if there are mixed types issues, though less likely for this specific CSV
            temp_df = pd.read_csv(output_path, on_bad_lines='skip')
            if not temp_df.empty and 'Question Slug' in temp_df.columns:
                processed_slugs.update(temp_df['Question Slug'].astype(str).tolist()) # Ensure slugs are strings
            print(f"Resuming questions_list.csv generation. Already have {len(processed_slugs)} questions.")
        except pd.errors.EmptyDataError:
            print(f"{output_path} is empty, starting from scratch or writing header.")
        except Exception as e:
            print(f"Error reading existing {output_path} for resume: {e}. Will attempt to write/append.")
            # If critical error, might need to os.remove(output_path) and start fresh,
            # but 'a' mode should handle some cases.

    # Counter for new questions added in this run
    new_questions_added_this_run = 0

    with open(output_path, 'a', newline='', encoding='utf-8') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=header)
        # Write header if file is new or was empty (os.path.getsize might be 0 for an empty file)
        if not file_exists or os.path.getsize(output_path) == 0:
            writer.writeheader()
            print("Written header to questions_list.csv")

        for i, problem in enumerate(problemset):
            if not problem or 'titleSlug' not in problem:
                print(f"Warning: Problem at index {i} is malformed or missing titleSlug. Skipping.")
                continue

            titleSlug = problem['titleSlug']
            if titleSlug in processed_slugs:
                continue

            print(f"Processing for list: {problem.get('frontendQuestionId', 'N/A')} {titleSlug} (API Index {i})")

            prob_dict = {}
            try:
                # Secondary API call for 'questionId'
                detail_query = get_problem_details_api_query(titleSlug)
                detailed_data = call_api(detail_query)

                if not detailed_data or 'question' not in detailed_data:
                    print(f"Warning: No detailed data or 'question' field for {titleSlug}. Skipping.")
                    continue
                question_details = detailed_data['question']
                if not question_details or 'questionId' not in question_details:
                    print(f"Warning: 'questionId' not in question_details for {titleSlug}. Skipping.")
                    continue

                prob_dict['Question ID'] = question_details['questionId']
                prob_dict['Question Title'] = problem.get('title', 'N/A')
                prob_dict['Question Slug'] = titleSlug

                tags = problem.get('topicTags', [])
                prob_dict['Topic Tagged text'] = ",".join([tag.get('name', '') for tag in tags])
                prob_dict['Topic Tagged ID'] = ",".join([tag.get('id', '') for tag in tags])

                writer.writerow(prob_dict)
                processed_slugs.add(titleSlug)
                new_questions_added_this_run += 1
            except Exception as e:
                print(f"Error processing details for {titleSlug} for questions_list: {e}. Skipping this question.")
                # Consider adding a small delay e.g. time.sleep(1) if suspecting rate limits
                continue
    
    print(f"Finished write/append pass for questions_list.csv.")
    print(f"Total unique questions in list now: {len(processed_slugs)}")
    print(f"New questions added in this specific run: {new_questions_added_this_run}")


def create_all_questions_data(skip, limit, output_path):

    ques_df = pd.read_csv('data/questions_list.csv')
    
    print(f'\n\nProcessing data from {skip} - {limit}')
    sl_range_df = ques_df[(ques_df.index >= skip) & (ques_df.index < limit)]

    main_df = pd.DataFrame()
    
    for _, question in sl_range_df.iterrows():

        print(f"Loading.. {question['Question ID']} {question['Question Slug']}")
        q_dict = {}
        q_dict['Question Title'] = question['Question Title']
        q_dict['Question Slug'] = question['Question Slug']
        q_dict['Question ID'] = int(question['Question ID'])

        # Call api
        query = get_problem_details_api_query(question['Question Slug'])
        data = call_api(query)
        data = data['question']

        # parse text
        q_dict['Question Text'] = parse_html_text(data['content'])

        # Topic Tagged Text, ID
        q_dict['Topic Tagged text'] = question['Topic Tagged text']
        q_dict['Topic Tagged ID'] = question['Topic Tagged ID']

        # stats
        q_dict['Difficulty Level'] = data['difficulty']
        stats = json.loads(data['stats'])
        q_dict['Success Rate'] = float(stats['acRate'][:-1])
        q_dict['total submission'] = int(stats['totalSubmissionRaw'])
        q_dict['total accepted'] = int(stats['totalAcceptedRaw'])

        # Company Tag
        q_dict['company tag'] = data['companyTagStats']

        # Likes & Dislikes
        q_dict['Likes'] = data['likes']
        q_dict['Dislikes'] = data['dislikes']

        # Parse hints
        q_dict['Hints'] = parse_html_hints(data['hints'])

        # get similar questions ids, text
        def get_similar_questions_details(ques_df, content):
            que_text_list = []
            que_ids = []
            
            for question in content:
                related_que = ques_df[ques_df['Question Slug'] == question['titleSlug']]

                if related_que.shape[0] == 0: # Uncommented this safety check
                    continue

                que_id = related_que['Question ID'].item()
                que_text = related_que['Question Title'].item()
                
                que_ids.append(str(que_id))
                que_text_list.append(que_text)

            q_ids = ",".join(que_ids)
            q_text = ",".join(que_text_list)
            

            return q_ids, q_text
                
        similar_ques = json.loads(data['similarQuestions'])
        q_dict['Similar Questions ID'], q_dict['Similar Questions Text'] = get_similar_questions_details(ques_df, similar_ques) 

        # create 1-D DataFrame
        q_df = pd.DataFrame.from_dict(q_dict, orient='index').transpose()
        
        main_df = main_df.append(q_df)
        
    main_df.to_csv(output_path, index=False)


def generate_question_list():

    DIR_PATH = 'data'
    QUE_LIST_PATH = os.path.join(os.getcwd(), DIR_PATH, f'questions_list.csv')

    if not os.path.exists(DIR_PATH):
        os.makedirs(DIR_PATH)
    
    if not os.path.exists(QUE_LIST_PATH):
        create_questions_list(QUE_LIST_PATH)

def generate_all_question_data(skip, limit):

    DIR_PATH = 'data/all_questions_data'
    ALL_DATA_PATH = os.path.join(os.getcwd(), DIR_PATH, f'all_ques_data_{skip}_{limit}.csv')

    if not os.path.exists(DIR_PATH):
        os.makedirs(DIR_PATH)
    
    if not os.path.exists(ALL_DATA_PATH): # Check if chunk file already exists
        create_all_questions_data(skip, limit, ALL_DATA_PATH)
    else:
        print(f"Chunk file {ALL_DATA_PATH} already exists. Skipping generation for this chunk.")

def main():

    # Part - 1
    # questions_list.csv will be generated by generate_question_list() if it doesn't exist.
    # We want to preserve it and the chunked data if the script timed out previously.
    print("Ensuring question list exists...")
    generate_question_list()
    print("Question list check complete.")

    # Part - 2
    questions_list_path = os.path.join(os.getcwd(), 'data', 'questions_list.csv')
    if not os.path.exists(questions_list_path):
        print(f"Error: {questions_list_path} not found. Please generate it first (should have happened above).")
        return

    try:
        total_questions_df = pd.read_csv(questions_list_path)
        total_questions = len(total_questions_df)
        print(f"Total questions found in questions_list.csv: {total_questions}")
    except pd.errors.EmptyDataError:
        print(f"Error: {questions_list_path} is empty. No questions to process.")
        return
    except Exception as e:
        print(f"Error reading {questions_list_path}: {e}")
        return

    if total_questions == 0:
        print("No questions to process from questions_list.csv.")
        return

    CHUNK_SIZE = 200 # Process 200 questions per chunk

    # Ensure the directory for chunked data exists, but do not clean it to allow resume.
    chunk_data_dir = os.path.join(os.getcwd(), 'data', 'all_questions_data')
    if not os.path.exists(chunk_data_dir):
        os.makedirs(chunk_data_dir)
        print(f"Created directory for chunked data: {chunk_data_dir}")

    for i in range(0, total_questions, CHUNK_SIZE):
        skip = i
        limit = min(i + CHUNK_SIZE, total_questions) # Ensure limit doesn't exceed total questions

        print(f"\nProcessing chunk: SKIP={skip}, LIMIT={limit}")
        # The generate_all_question_data function expects limit to be exclusive for slicing,
        # but its internal slicing ques_df[(ques_df.index >= skip) & (ques_df.index < limit)] is correct.
        generate_all_question_data(skip, limit)

    print("\nAll question data generation complete.")

main()





