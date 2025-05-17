import aiohttp
import asyncio
import hashlib

# -------------------------
# CONFIGURABLE API KEYS
# -------------------------
API_KEYS = [
     "APIKEY"  # Replace with actual GNews API keys
]

SCRAPED_FILE = "scraped_articles.txt"
OUTPUT_FILE = "fetched_articles.txt"

# In-memory set to simulate stored article hashes (deduplication)
seen_hashes = set()

# -------------------------
# HELPER FUNCTIONS
# -------------------------

def get_scraped_data():
    data = []
    try:
        with open(SCRAPED_FILE, "r", encoding="utf-8") as f:
            for line in f:
                if "||" in line:
                    title, description = line.strip().split("||")
                    data.append((title.strip(), description.strip()))
    except FileNotFoundError:
        print(f"Error: {SCRAPED_FILE} not found.")
    return data

def generate_article_hash(article: dict) -> str:
    content = (article.get("title", "") + article.get("description", "") + article.get("url", "")).strip()
    return hashlib.sha256(content.encode("utf-8")).hexdigest()

def article_hash_exists(article_hash: str) -> bool:
    return article_hash in seen_hashes

def store_article_hash(article_hash: str):
    seen_hashes.add(article_hash)

async def fetch_gnews_with_key(session, key, query):
    url = f"https://gnews.io/api/v4/search?q={query}&token={key}&lang=en&max=10"
    try:
        async with session.get(url) as resp:
            if resp.status == 429:
                print(f"Rate limit hit for key: {key}")
                return None
            elif resp.status == 200:
                return await resp.json()
            else:
                print(f"Key {key} failed with status {resp.status}")
    except Exception as e:
        print(f"Request failed with key {key}: {e}")
    return None

# -------------------------
# MAIN ASYNC FETCH FUNCTION
# -------------------------

async def fetch_relevant_articles_from_scraped():
    scraped_articles = get_scraped_data()  # List of (title, description)
    if not scraped_articles:
        print("No scraped articles found.")
        return []

    fetched_articles = []
    used_keys = set()

    async with aiohttp.ClientSession() as session:
        for title, description in scraped_articles:
            query = title or description
            if not query:
                continue

            articles_found = False
            for key in API_KEYS:
                if key in used_keys:
                    continue  # Skip used keys

                data = await fetch_gnews_with_key(session, key, query)
                used_keys.add(key)

                if data and "articles" in data:
                    for article in data["articles"]:
                        article_hash = generate_article_hash(article)
                        if not article_hash_exists(article_hash):
                            store_article_hash(article_hash)
                            fetched_articles.append(article)
                    articles_found = True
                    break  # Move to next scraped article

            if not articles_found:
                print(f"No new articles found for: {query}")

    return fetched_articles

# -------------------------
# WRITE TO TEXT FILE
# -------------------------

def save_articles_to_file(articles):
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        for idx, art in enumerate(articles, start=1):
            f.write(f"Article #{idx}\n")
            f.write(f"Title: {art.get('title', '')}\n")
            f.write(f"Description: {art.get('description', '')}\n")
            f.write(f"URL: {art.get('url', '')}\n")
            f.write("-" * 80 + "\n")
    print(f"\n {len(articles)} articles saved to '{OUTPUT_FILE}'.")

# -------------------------
# ENTRY POINT
# -------------------------

if __name__ == "__main__":
    articles = asyncio.run(fetch_relevant_articles_from_scraped())
    save_articles_to_file(articles)
