import requests
from bs4 import BeautifulSoup

# Base URL pattern for paginated review requests
base_url = "https://steamcommunity.com/app/271590/homecontent/"

# Query parameters (modify these if needed)
params = {
    "userreviewsoffset": 0,  # Starts at 0, will increment for pagination
    "p": 1,  # Page number
    "browsefilter": "toprated",  # Filter for top-rated reviews
    "appHubSubSection": 10,  # Required for reviews
    "l": "english",  # Language filter
    "filterLanguage": "default",
    "searchText": "",  # No specific search
    "forceanon": 1 # Forces Requests to be anonymous
}

headers = {
    "User-Agent": "Mozilla/5.0"  # Imitates a real browser to avoid blocks
}

# Number of pages to scrape
num_pages = 5  # Adjust as needed

for i in range(num_pages):
    params["userreviewsoffset"] = i + 10  # Adjust the offset for pagination
    params["p"] = i + 1  # Set page number

    response = requests.get(base_url, params=params, headers=headers)

    if response.status_code == 200:
        soup = BeautifulSoup(response.text, "html.parser")

        # Extract reviews
        reviews = soup.find_all("div", class_="apphub_CardTextContent")  # Review text container

        for review in reviews:
            text = review.get_text(strip=True)
            print(f"Review: {text}\n" + "-" * 80)
    else:
        print(f"Failed to fetch page {i + 1} - Status Code:", response.status_code)
