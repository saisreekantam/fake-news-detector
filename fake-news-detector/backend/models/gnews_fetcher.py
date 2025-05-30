import aiohttp
import hashlib
from database.repository import save_articles, get_scraped_data, article_hash_exists, store_article_hash

API_KEYS = ["d368eb729e9bde4eed4f6f9707647b13"]

def generate_article_hash(article: dict) -> str:
    content = (article.get("title", "") + article.get("description", "") + article.get("url", "")).strip()
    return hashlib.sha256(content.encode("utf-8")).hexdigest()

async def fetch_gnews_with_key(session, key, query):
    url = f"https://gnews.io/api/v4/search?q={query}&token={key}&lang=en&max=10"
    async with session.get(url) as resp:
        if resp.status == 429:
            print(f"Rate limit hit for key: {key}")
            return None
        if resp.status == 200:
            return await resp.json()
        else:
            print(f"Key {key} failed with status {resp.status}")
    return None

async def fetch_relevant_articles_from_scraped():
    scraped_articles = get_scraped_data()  # [(title, description)]
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
                    continue  # skip already used keys

                data = await fetch_gnews_with_key(session, key, query)
                used_keys.add(key)

                if data and "articles" in data:
                    for article in data["articles"]:
                        article_hash = generate_article_hash(article)
                        if not article_hash_exists(article_hash):
                            save_articles([article])
                            store_article_hash(article_hash)
                            fetched_articles.append(article)
                    articles_found = True
                    break  # move to next scraped article

            if not articles_found:
                print(f"No new articles found for: {query}")

    return fetched_articles
