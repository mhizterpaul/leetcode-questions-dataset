from utils.scraper import collect_and_save_links, scrape_and_save_questions
import os

def main():
    data_dir = os.path.join(os.path.dirname(__file__), 'data')
    os.makedirs(data_dir, exist_ok=True)

    # Step 1: Collect links
    collect_and_save_links(data_dir)

    # Step 2: Scrape questions
    scrape_and_save_questions(data_dir)

if __name__ == "__main__":
    main()
