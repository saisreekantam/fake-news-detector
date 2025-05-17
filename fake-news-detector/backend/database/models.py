
from sqlalchemy import Column, Integer, String, Text, DateTime,UniqueConstraint
from sqlalchemy.ext.declarative import declarative_base
from datetime import datetime

Base = declarative_base()

class GNewsArticle(Base):
    __tablename__ = 'gnews_articles'
    
    id = Column(Integer, primary_key=True)
    title = Column(String)
    description = Column(Text)
    content = Column(Text)
    url = Column(String, unique=True)
    image = Column(String)
    published_at = Column(DateTime)
    source_name = Column(String)
    timestamp = Column(DateTime, default=datetime.utcnow)


Base = declarative_base()

class ArticleHash(Base):
    __tablename__ = 'article_hashes'

    id = Column(Integer, primary_key=True, autoincrement=True)
    hash = Column(String(256), nullable=False, unique=True)

    __table_args__ = (
        UniqueConstraint('hash', name='uq_article_hash'),
    )

    def __repr__(self):
        return f"<ArticleHash(id={self.id}, hash='{self.hash}')>"

