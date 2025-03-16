import pandas as pd
import requests
from bs4 import BeautifulSoup
from collections import deque
import re


BASE_URL = "https://en.wikipedia.org"

# Headers to mimic a browser request
HEADERS = {"User-Agent": "Mozilla/5.0"}

# Store visited categories and character pages
visited_categories = set()
character_pages = []

def scrape_wikipedia_category(start_url, max_depth=5):
    """Iteratively scrapes Wikipedia category pages using BFS."""
    queue = deque([(start_url, 0)])  # (URL, depth)

    while queue:
        url, depth = queue.popleft()

        if depth > max_depth or url in visited_categories:
            continue

        visited_categories.add(url)
        response = requests.get(url, headers=HEADERS)
        data = BeautifulSoup(response.text, "html.parser")

        # Extract subcategories
        subcategories = data.select("#mw-subcategories a ")
        for subcat in subcategories:
            if len(queue) == 150:
                break
            if "href" in subcat.attrs:
                subcat_url = BASE_URL + subcat["href"]
                queue.append((subcat_url, depth + 1))  # Add to queue
                print(" " * depth * 2 + f"-> Exploring subcategory: {subcat.text}")


        def clean_name(char_name):
            return re.sub(r"\s*\(.*?\)", "", char_name).strip()

        # Extract character pages
        character_links = data.select("#mw-pages a")
        for link in character_links:
            if len(character_pages) == 30000:
                break
            if "href" in link.attrs:
                char_name = clean_name(link.text)
                if (char_name == "This list may not reflect recent changes" or "List of" in char_name or
                        "Lists of" in char_name or 'Character' in char_name or '0' in char_name):
                    continue
                if not character_pages.__contains__(char_name):
                    character_pages.append(char_name)
                print(" " * depth * 2 + f"--> Found character: {char_name}")

# Scrape the Wikipedia Page
start_url = "https://en.wikipedia.org/wiki/Category:Fictional_characters"
scrape_wikipedia_category(start_url)

# Save results
for name in character_pages:
    print(f"{name}")

print(pd.DataFrame(character_pages, columns=['Character Name']))

df = pd.DataFrame(character_pages, columns=['Name'])
df.to_csv('Fictional_Names.csv', index=False)
