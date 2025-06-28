# HackerRank Scraper - Playwright + BeautifulSoup + pandas
# Goal: Collect and extract 2000 links + full question text + image URLs from two domains (algorithms, data-structures)

# ------------------------------------------------------------------------------------
# ✅ OBJECTIVE
# ------------------------------------------------------------------------------------
# Generate 4 CSV datasets:
#
# 1. algorithms_links.csv           --> 2000 links (Algorithms domain)
# 2. data_structures_links.csv      --> 2000 links (Data Structures domain)
# 3. algorithms_questions.csv       --> Full question text + image URLs from Algo links
# 4. data_structures_questions.csv  --> Full question text + image URLs from DS links

# ------------------------------------------------------------------------------------
# 📦 INSTALL DEPENDENCIES
# ------------------------------------------------------------------------------------
# pip install pandas beautifulsoup4 playwright
# playwright install webkit

# ------------------------------------------------------------------------------------
# 🧱 PART 1: Collect 2000 Links for Each Domain
# ------------------------------------------------------------------------------------

# File: collect_hackerrank_links.py

import asyncio
import pandas as pd
from playwright.async_api import async_playwright

async def collect_links(domain_url: str, count: int = 2000) -> list:
    links = set()

    async with async_playwright() as p:
        browser = await p.webkit.launch(headless=True)
        page = await browser.new_page()
        await page.goto(domain_url)
        prev_height = 0

        while len(links) < count:
            await page.wait_for_selector("div.challenge-list a[href]", timeout=10000)
            elements = await page.query_selector_all("div.challenge-list a[href]")

            for el in elements:
                href = await el.get_attribute("href")
                if href and "/challenges/" in href:
                    full_url = "https://www.hackerrank.com" + href.split("?")[0]
                    links.add(full_url)
                    if len(links) >= count:
                        break

            await page.evaluate("window.scrollTo(0, document.body.scrollHeight)")
            await page.wait_for_timeout(1500)

            current_height = await page.evaluate("document.body.scrollHeight")
            if current_height == prev_height:
                print("Reached bottom of page.")
                break
            prev_height = current_height

        await browser.close()
    return list(links)[:count]

async def main():
    algo_url = "https://hackerrank.com/domains/algorithms?filters%5Bdifficulty%5D%5B%5D=medium&filters%5Bdifficulty%5D%5B%5D=hard"
    ds_url = "https://hackerrank.com/domains/data-structures?filters%5Bdifficulty%5D%5B%5D=medium&filters%5Bdifficulty%5D%5B%5D=hard"

    print("Collecting Algorithms links...")
    algo_links = await collect_links(algo_url)
    pd.DataFrame(algo_links, columns=["url"]).to_csv("algorithms_links.csv", index=False)

    print("Collecting Data Structures links...")
    ds_links = await collect_links(ds_url)
    pd.DataFrame(ds_links, columns=["url"]).to_csv("data_structures_links.csv", index=False)

    print("Done.")

if __name__ == "__main__":
    asyncio.run(main())

# ------------------------------------------------------------------------------------
# 🧱 PART 2: Scrape Full Question Text + Image URLs
# ------------------------------------------------------------------------------------

# File: scrape_question_details.py

import asyncio
import pandas as pd
from bs4 import BeautifulSoup
from playwright.async_api import async_playwright
from urllib.parse import urljoin

async def extract_plaintext_and_images(page, url: str) -> tuple[str, str]:
    await page.goto(url)
    await page.wait_for_timeout(2000)

    # Handle authentication popup
    try:
        if await page.query_selector("div.auth-box"):
            print("Closing auth modal...")
            await page.click("div.auth-modal-close button")
            await page.wait_for_timeout(1000)
    except:
        pass

    try:
        div = await page.query_selector("div.challenge-body-html")
        if div:
            html = await div.inner_html()
            soup = BeautifulSoup(html, "html.parser")

            text = soup.get_text(separator="\n", strip=True)
            images = [urljoin(url, img.get("src")) for img in soup.find_all("img") if img.get("src")]
            return text, "|".join(images)
    except Exception as e:
        print(f"Error extracting from {url}: {e}")

    return "", ""

async def scrape_questions(input_csv: str, output_csv: str):
    df = pd.read_csv(input_csv)
    results = []

    async with async_playwright() as p:
        browser = await p.webkit.launch(headless=True)
        page = await browser.new_page()

        for i, row in df.iterrows():
            url = row["url"]
            print(f"[{i+1}/{len(df)}] Scraping {url}")

            try:
                text, images = await extract_plaintext_and_images(page, url)
                results.append({"url": url, "question_text": text, "image_urls": images})
            except Exception as e:
                print(f"Failed: {url} - {e}")

            await page.wait_for_timeout(1000)

        await browser.close()

    pd.DataFrame(results).to_csv(output_csv, index=False)
    print(f"Saved to {output_csv}")

async def main():
    await scrape_questions("algorithms_links.csv", "algorithms_questions.csv")
    await scrape_questions("data_structures_links.csv", "data_structures_questions.csv")

if __name__ == "__main__":
    asyncio.run(main())

# ------------------------------------------------------------------------------------
# 🗃 OUTPUT STRUCTURE SUMMARY
# ------------------------------------------------------------------------------------

# algorithms_links.csv          --> [ url ]
# data_structures_links.csv     --> [ url ]
# algorithms_questions.csv      --> [ url, question_text, image_urls ]
# data_structures_questions.csv --> [ url, question_text, image_urls ]

# image_urls is a '|' separated string of image URLs (absolute paths) or empty if none

# ------------------------------------------------------------------------------------
# 🔄 Proxy Support (optional)
# ------------------------------------------------------------------------------------
# browser = await p.webkit.launch(proxy={"server": "http://<ip>:<port>"})

# - Retry logic for failed URLs------------------------------------------------------------------------------------
# ------------------------------------------------------------------------------------


