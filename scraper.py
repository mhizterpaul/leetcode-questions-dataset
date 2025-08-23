import asyncio
import pandas as pd
from playwright.async_api import async_playwright
from bs4 import BeautifulSoup
from urllib.parse import urljoin
import os
import traceback
import time

def get_algorithms_url():
    return "https://hackerrank.com/domains/algorithms?filters%5Bdifficulty%5D%5B%5D=medium&filters%5Bdifficulty%5D%5B%5D=hard"

def get_data_structures_url():
    return "https://hackerrank.com/domains/data-structures?filters%5Bdifficulty%5D%5B%5D=medium&filters%5Bdifficulty%5D%5B%5D=hard"

async def collect_links(domain_url: str, count: int = 2000) -> list:
    links = set()
    try:
        async with async_playwright() as p:
            browser = await p.chromium.launch(headless=False)
            page = await browser.new_page()
            # Use no timeout for page.goto so it waits indefinitely
            await page.goto(domain_url, timeout=0, wait_until="domcontentloaded")
            selector = "div.challenges-list a.challenge-list-item[href]"
            scroll_attempts = 0
            max_scrolls = 210  # 200 + 10 safety
            no_new_link_start = None
            while len(links) < count and scroll_attempts < max_scrolls:
                try:
                    # Wait for at least one link to appear
                    await page.wait_for_selector(selector, timeout=60000)
                    # Extract all current links
                    elements = await page.query_selector_all(selector)
                    before_count = len(links)
                    for el in elements:
                        href = await el.get_attribute("href")
                        if href and "/challenges/" in href:
                            full_url = "https://www.hackerrank.com" + href
                            links.add(full_url)
                    after_count = len(links)
                    print(f"[collect_links] Scroll {scroll_attempts}: {after_count} unique links collected.")
                    if after_count >= count:
                        break
                    # Scroll to bottom
                    await page.evaluate("window.scrollTo(0, document.body.scrollHeight)")
                    # Wait for DOM to update (wait for more <a> if possible)
                    new_links_found = False
                    for _ in range(20):  # up to 10 seconds
                        await page.wait_for_timeout(500)
                        new_elements = await page.query_selector_all(selector)
                        if len(new_elements) > len(elements):
                            new_links_found = True
                            break
                    if after_count == before_count:
                        if no_new_link_start is None:
                            no_new_link_start = time.time()
                        elif time.time() - no_new_link_start > 120:
                            print("[collect_links] No new links found for 3 minutes. Stopping.")
                            break
                    else:
                        no_new_link_start = None
                    scroll_attempts += 1
                except Exception as e:
                    print(f"[collect_links] Error during page interaction: {e}")
                    content = await page.content()
                    print("[collect_links] Page content snippet:", content[:1000])
                    traceback.print_exc()
                    break
            await browser.close()
    except Exception as e:
        print(f"[collect_links] Error: {e}")
        traceback.print_exc()
    return list(links)[:count]

def collect_and_save_links(data_dir: str):
    async def _main():
        try:
            algo_url = get_algorithms_url()
            ds_url = get_data_structures_url()
            algo_links = await collect_links(algo_url)
            ds_links = await collect_links(ds_url)
            algo_path = os.path.join(data_dir, "algorithms_links.csv")
            ds_path = os.path.join(data_dir, "data_structures_links.csv")
            # Remove files if they exist
            if os.path.exists(algo_path):
                os.remove(algo_path)
            if os.path.exists(ds_path):
                os.remove(ds_path)
            pd.DataFrame(algo_links, columns=["url"]).to_csv(algo_path, index=False)
            pd.DataFrame(ds_links, columns=["url"]).to_csv(ds_path, index=False)
        except Exception as e:
            print(f"[collect_and_save_links] Error: {e}")
            traceback.print_exc()
    try:
        asyncio.run(_main())
    except Exception as e:
        print(f"[collect_and_save_links] Error running asyncio: {e}")
        traceback.print_exc()

async def extract_plaintext_and_images(page, url: str) -> tuple[str, str]:
    max_retries = 2
    for attempt in range(max_retries):
        try:
            await page.goto(url, timeout=25000, wait_until="domcontentloaded")
            await page.wait_for_timeout(2000)
            # Check for auth modal/banner and close if present (alignment with scrape_questions)
            try:
                if await page.query_selector("div.auth-box"):
                    try:
                        await page.click("div.auth-modal-close button")
                        await page.wait_for_timeout(1000)
                    except Exception as e:
                        print(f"[extract_plaintext_and_images] Error closing auth modal: {e}")
                        traceback.print_exc()
            except Exception as e:
                print(f"[extract_plaintext_and_images] Error checking for auth box: {e}")
                traceback.print_exc()
            try:
                div = await page.query_selector("div.challenge-body-html")
                if div:
                    html = await div.inner_html()
                    soup = BeautifulSoup(html, "html.parser")
                    # Replace each <img> with its absolute URL in the text
                    for img in soup.find_all("img"):
                        img_url = urljoin(url, img.get("src")) if img.get("src") else ""
                        img.replace_with(img_url)
                    text = soup.get_text(separator="\n", strip=True)
                    images = [urljoin(url, img.get("src")) for img in soup.find_all("img") if img.get("src")]
                    print(f"[extract_plaintext_and_images] Success: {url} | Text length: {len(text)} | Images: {len(images)}")
                    return text, "|".join(images)
            except Exception as e:
                print(f"[extract_plaintext_and_images] Error extracting question: {e}")
                traceback.print_exc()
            break  # Success, exit retry loop
        except Exception as e:
            print(f"[extract_plaintext_and_images] Error navigating to {url} (attempt {attempt+1}): {e}")
            if attempt == max_retries - 1:
                traceback.print_exc()
                return "", ""
            await page.wait_for_timeout(5000)  # Wait 5 seconds before retry
    return "", ""

async def scrape_questions(input_csv: str, output_csv: str):
    try:
        df = pd.read_csv(input_csv)
        # Determine where to resume from
        scraped_urls = set()
        if os.path.exists(output_csv):
            try:
                out_df = pd.read_csv(output_csv)
                scraped_urls = set(out_df['url'].tolist())
            except Exception as e:
                print(f"[scrape_questions] Error reading output CSV for resume: {e}")
        # Open output file in append mode
        write_header = not os.path.exists(output_csv)
        async with async_playwright() as p:
            browser = await p.chromium.launch(headless=False)
            page = await browser.new_page()
            for _, row in df.iterrows():
                url = row["url"]
                if url in scraped_urls:
                    continue  # Skip already scraped
                try:
                    text, images = await extract_plaintext_and_images(page, url)
                    result = {"url": url, "question_text": text, "image_urls": images}
                    # Append to CSV one at a time
                    pd.DataFrame([result]).to_csv(output_csv, mode='a', header=write_header, index=False)
                    write_header = False  # Only write header for the first append
                except Exception as e:
                    print(f"[scrape_questions] Error scraping {url}: {e}")
                    traceback.print_exc()
                    result = {"url": url, "question_text": "", "image_urls": ""}
                    pd.DataFrame([result]).to_csv(output_csv, mode='a', header=write_header, index=False)
                    write_header = False
                await page.wait_for_timeout(1000)
            await browser.close()
    except Exception as e:
        print(f"[scrape_questions] Error: {e}")
        traceback.print_exc()

def scrape_and_save_single(input_csv: str, output_csv: str):
    asyncio.run(scrape_questions(input_csv, output_csv))

def scrape_and_save_questions(data_dir: str):
    try:
        algo_input = os.path.join(data_dir, "algorithms_links.csv")
        algo_output = os.path.join(data_dir, "algorithms_questions.csv")
        ds_input = os.path.join(data_dir, "data_structures_links.csv")
        ds_output = os.path.join(data_dir, "data_structures_questions.csv")
        scrape_and_save_single(algo_input, algo_output)
        scrape_and_save_single(ds_input, ds_output)
    except Exception as e:
        print(f"[scrape_and_save_questions] Error: {e}")
        traceback.print_exc()

if __name__ == "__main__":
    try:
        data_dir = "data/hackerrank_dataset"
        algo_path = os.path.join(data_dir, "algorithms_links.csv")
        ds_path = os.path.join(data_dir, "data_structures_links.csv")
        if not (os.path.exists(algo_path) and os.path.exists(ds_path)):
            collect_and_save_links(data_dir)
        else:
            scrape_and_save_questions(data_dir)
    except Exception as e:
        print(f"[main] Error: {e}")
        traceback.print_exc()