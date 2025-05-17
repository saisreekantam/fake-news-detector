from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from .models import Base, GNewsArticle,ArticleHash
import psycopg2

DATABASE_URL = "postgresql://username:password@localhost:5432/yourdb" # Replace with your actual database URL
# Ensure you have the correct database URL format

engine = create_engine(DATABASE_URL)
SessionLocal = sessionmaker(bind=engine)
Base.metadata.create_all(engine)

def save_articles(articles):
    session = SessionLocal()
    for a in articles:
        if not session.query(GNewsArticle).filter_by(url=a['url']).first():
            session.add(GNewsArticle(
                title=a['title'],
                description=a['description'],
                content=a['content'],
                url=a['url'],
                image=a['image'],
                published_at=a.get('publishedAt'),
                source_name=a.get('source', {}).get('name')
            ))
    session.commit()
    session.close()


def get_scraped_data():
    conn = psycopg2.connect("dbname=your_db user=your_user password=your_pass")
    cur = conn.cursor()
    cur.execute("SELECT title, description FROM scraped_articles;")
    rows = cur.fetchall()
    cur.close()
    conn.close()

    return [{"title": row[0], "description": row[1]} for row in rows]
def article_hash_exists(article_hash: str) -> bool:
    session = SessionLocal()
    result = session.query(ArticleHash).filter_by(hash=article_hash).first()
    session.close()
    return result is not None

def store_article_hash(article_hash: str):
    session = SessionLocal()
    new_hash = ArticleHash(hash=article_hash)
    session.add(new_hash)
    session.commit()
    session.close()


