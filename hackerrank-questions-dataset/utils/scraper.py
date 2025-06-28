import asyncio
import pandas as pd
from playwright.async_api import async_playwright
from bs4 import BeautifulSoup
from urllib.parse import urljoin
import os

def get_algorithms_url():
    return "https://hackerrank.com/domains/algorithms?filters%5Bdifficulty%5D%5B%5D=medium&filters%5Bdifficulty%5D%5B%5D=hard"

def get_data_structures_url():
    return "https://hackerrank.com/domains/data-structures?filters%5Bdifficulty%5D%5B%5D=medium&filters%5Bdifficulty%5D%5B%5D=hard"

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
                break
            prev_height = current_height
        await browser.close()
    return list(links)[:count]

def collect_and_save_links(data_dir: str):
    async def _main():
        algo_url = get_algorithms_url()
        ds_url = get_data_structures_url()
        algo_links = await collect_links(algo_url)
        ds_links = await collect_links(ds_url)
        pd.DataFrame(algo_links, columns=["url"]).to_csv(os.path.join(data_dir, "algorithms_links.csv"), index=False)
        pd.DataFrame(ds_links, columns=["url"]).to_csv(os.path.join(data_dir, "data_structures_links.csv"), index=False)
    asyncio.run(_main())

async def extract_plaintext_and_images(page, url: str) -> tuple[str, str]:
    await page.goto(url)
    await page.wait_for_timeout(2000)
    try:
        if await page.query_selector("div.auth-box"):
            try:
                await page.click("div.auth-modal-close button")
                await page.wait_for_timeout(1000)
            except:
                pass
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
    except Exception:
        pass
    return "", ""

async def scrape_questions(input_csv: str, output_csv: str):
    df = pd.read_csv(input_csv)
    results = []
    async with async_playwright() as p:
        browser = await p.webkit.launch(headless=True)
        page = await browser.new_page()
        for _, row in df.iterrows():
            url = row["url"]
            try:
                text, images = await extract_plaintext_and_images(page, url)
                results.append({"url": url, "question_text": text, "image_urls": images})
            except Exception:
                results.append({"url": url, "question_text": "", "image_urls": ""})
            await page.wait_for_timeout(1000)
        await browser.close()
    pd.DataFrame(results).to_csv(output_csv, index=False)

def scrape_and_save_questions(data_dir: str):
    async def _main():
        await scrape_questions(
            os.path.join(data_dir, "algorithms_links.csv"),
            os.path.join(data_dir, "algorithms_questions.csv")
        )
        await scrape_questions(
            os.path.join(data_dir, "data_structures_links.csv"),
            os.path.join(data_dir, "data_structures_questions.csv")
        )
    asyncio.run(_main()) 