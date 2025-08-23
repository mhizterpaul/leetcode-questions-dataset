import requests

def get_problemset_api_query(skip:int, limit:int) -> list:
    
    query = {"query":"\n    query problemsetQuestionList($categorySlug: String, $limit: Int, $skip: Int, $filters: QuestionListFilterInput) {\n  problemsetQuestionList: questionList(\n    categorySlug: $categorySlug\n    limit: $limit\n    skip: $skip\n    filters: $filters\n  ) {\n    total: totalNum\n    questions: data {\n      acRate\n      difficulty\n      freqBar\n      frontendQuestionId: questionFrontendId\n      isFavor\n      paidOnly: isPaidOnly\n      status\n      title\n      titleSlug\n      topicTags {\n        name\n        id\n        slug\n      }\n      hasSolution\n      hasVideoSolution\n    }\n  }\n}\n    ",
         "variables":{"categorySlug":"","skip":skip,"limit":limit,"filters":{}}}
    
    return query

def get_problem_details_api_query(titleSlug:str) -> dict:
    
    query = {"operationName":"questionData","variables":{"titleSlug":titleSlug},"query":"query questionData($titleSlug: String!) {\n  question(titleSlug: $titleSlug) {\n    questionId\n    questionFrontendId\n    boundTopicId\n    title\n    titleSlug\n    content\n    translatedTitle\n    translatedContent\n    isPaidOnly\n    difficulty\n    likes\n    dislikes\n    isLiked\n    similarQuestions\n    exampleTestcases\n    categoryTitle\n    contributors {\n      username\n      profileUrl\n      avatarUrl\n      __typename\n    }\n    topicTags {\n      name\n      slug\n      translatedName\n      __typename\n    }\n    companyTagStats\n    codeSnippets {\n      lang\n      langSlug\n      code\n      __typename\n    }\n    stats\n    hints\n    solution {\n      id\n      canSeeDetail\n      paidOnly\n      hasVideoSolution\n      paidOnlyVideo\n      __typename\n    }\n    status\n    sampleTestCase\n    metaData\n    judgerAvailable\n    judgeType\n    mysqlSchemas\n    enableRunCode\n    enableTestMode\n    enableDebugger\n    envInfo\n    libraryUrl\n    adminUrl\n    challengeQuestion {\n      id\n      date\n      incompleteChallengeCount\n      streakCount\n      type\n      __typename\n    }\n    __typename\n  }\n}\n"}

    return query

import json # Required for json.JSONDecodeError

def call_api(query:dict) -> dict:
    base_url = "https://leetcode.com/graphql/"
    try:
        response = requests.post(base_url, json=query, timeout=20) # Added timeout
        response.raise_for_status() # Raises HTTPError for bad responses (4XX or 5XX)

        # Check content type before trying to decode JSON, though LeetCode API should be consistent
        # content_type = response.headers.get("Content-Type", "")
        # if "application/json" not in content_type:
        #     print(f"API Error: Unexpected content type '{content_type}'. Response text: {response.text[:500]}")
        #     return None # Or handle as appropriate

        return response.json().get('data') # Safely get 'data' key

    except requests.exceptions.HTTPError as http_err:
        print(f"API HTTP error occurred: {http_err} - Status Code: {response.status_code} - Response: {response.text[:500]}")
        return None
    except requests.exceptions.ConnectionError as conn_err:
        print(f"API Connection error occurred: {conn_err}")
        return None
    except requests.exceptions.Timeout as timeout_err:
        print(f"API Timeout error occurred: {timeout_err}")
        return None
    except requests.exceptions.RequestException as req_err:
        print(f"API Ambiguous request error occurred: {req_err}")
        return None
    except json.JSONDecodeError as json_err:
        # This will catch cases where response.text is not valid JSON
        print(f"API JSONDecodeError: {json_err}. Response text: {response.text[:500]}") # Log part of response
        return None
    except Exception as e: # Catch any other unexpected errors
        print(f"An unexpected error occurred in call_api: {e}")
        return None