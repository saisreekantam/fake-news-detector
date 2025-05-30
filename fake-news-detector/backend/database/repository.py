"""
Database Repository for Fake News Detection System
================================================
This module handles all database operations including article storage,
retrieval, and hash management for duplicate detection.
"""

import os
import logging
from typing import List, Dict, Any, Optional
from datetime import datetime
from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker
from sqlalchemy.exc import SQLAlchemyError
import hashlib

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Import models with error handling
try:
    from .models import Base, GNewsArticle, ArticleHash
    MODELS_AVAILABLE = True
    logger.info("✅ Database models imported successfully")
except ImportError as e:
    logger.warning(f"❌ Could not import database models: {e}")
    try:
        # Try alternative import path
        from models import Base, GNewsArticle, ArticleHash
        MODELS_AVAILABLE = True
        logger.info("✅ Database models imported successfully (alternative path)")
    except ImportError as e2:
        logger.error(f"❌ Failed to import models from both paths: {e2}")
        MODELS_AVAILABLE = False
        Base = None
        GNewsArticle = None
        ArticleHash = None

# Database configuration with fallbacks
def get_database_url():
    """Get database URL from environment with fallbacks"""
    # Try environment variable first
    db_url = os.getenv('DATABASE_URL')
    if db_url:
        return db_url
    
    # Construct from individual environment variables
    host = os.getenv('DB_HOST', 'localhost')
    port = os.getenv('DB_PORT', '5432')
    user = os.getenv('DB_USER', 'fake_news_user')
    password = os.getenv('DB_PASSWORD', 'your_secure_password')
    database = os.getenv('DB_NAME', 'fake_news_db')
    
    return f"postgresql://{user}:{password}@{host}:{port}/{database}"

# Initialize database connection
DATABASE_URL = get_database_url()
engine = None
SessionLocal = None
DATABASE_AVAILABLE = False

try:
    engine = create_engine(DATABASE_URL, echo=False, pool_pre_ping=True)
    SessionLocal = sessionmaker(bind=engine)
    
    # Test connection
    with engine.connect() as conn:
        conn.execute(text("SELECT 1"))
    
    if MODELS_AVAILABLE and Base is not None:
        Base.metadata.create_all(engine)
        logger.info("✅ Database connection established and tables created/verified")
    else:
        logger.warning("⚠ Database connected but models not available")
    
    DATABASE_AVAILABLE = True
    
except Exception as e:
    logger.error(f"❌ Database initialization failed: {e}")
    logger.info("📝 The system will run with limited functionality (no database)")
    DATABASE_AVAILABLE = False

def get_session():
    """Get a database session with error handling"""
    if not DATABASE_AVAILABLE or SessionLocal is None:
        raise Exception("Database not available")
    return SessionLocal()

def is_database_available() -> bool:
    """Check if database is available"""
    return DATABASE_AVAILABLE and MODELS_AVAILABLE

def save_articles(articles: List[Dict[str, Any]]) -> bool:
    """
    Save articles to the database
    
    Args:
        articles: List of article dictionaries
        
    Returns:
        bool: True if successful, False otherwise
    """
    if not articles:
        logger.warning("No articles provided to save")
        return False
    
    if not is_database_available():
        logger.warning("Database not available - cannot save articles")
        return False
    
    try:
        session = get_session()
        saved_count = 0
        
        for article_data in articles:
            try:
                # Extract URL for duplicate checking
                url = article_data.get('url', '')
                if not url:
                    logger.warning("Article missing URL, skipping")
                    continue
                
                # Check if article already exists by URL
                existing = session.query(GNewsArticle).filter_by(url=url).first()
                
                if not existing:
                    # Parse published_at if it's a string
                    published_at = article_data.get('published_at')
                    if isinstance(published_at, str):
                        try:
                            published_at = datetime.fromisoformat(published_at.replace('Z', '+00:00'))
                        except:
                            published_at = None
                    
                    # Extract source name from nested structure or direct field
                    source = article_data.get('source', {})
                    if isinstance(source, dict):
                        source_name = source.get('name', '')
                    else:
                        source_name = article_data.get('source_name', str(source) if source else '')
                    
                    # Create new article
                    article = GNewsArticle(
                        title=article_data.get('title', '')[:500],  # Truncate to fit DB constraint
                        description=article_data.get('description', ''),
                        content=article_data.get('content', ''),
                        url=url[:1000],  # Truncate to fit DB constraint
                        image=article_data.get('image', ''),
                        published_at=published_at,
                        source_name=source_name[:200] if source_name else '',  # Truncate
                        author=article_data.get('author', '')[:200] if article_data.get('author') else '',
                        word_count=len(article_data.get('content', '').split()) if article_data.get('content') else 0
                    )
                    
                    session.add(article)
                    saved_count += 1
                    logger.debug(f"Added article: {article.title[:50]}...")
                else:
                    logger.debug(f"Article already exists: {existing.title[:50]}...")
                    
            except Exception as e:
                logger.error(f"Error processing individual article: {e}")
                continue
        
        session.commit()
        session.close()
        
        if saved_count > 0:
            logger.info(f"✅ Successfully saved {saved_count} new articles")
        else:
            logger.info("ℹ No new articles to save (all duplicates)")
        
        return True
        
    except SQLAlchemyError as e:
        logger.error(f"Database error saving articles: {e}")
        if 'session' in locals():
            session.rollback()
            session.close()
        return False
    except Exception as e:
        logger.error(f"Unexpected error saving articles: {e}")
        if 'session' in locals():
            session.close()
        return False

def get_scraped_data(limit: int = 100) -> List[Dict[str, Any]]:
    """
    Get scraped article data from the database
    
    Args:
        limit: Maximum number of articles to return
        
    Returns:
        List of article dictionaries
    """
    if not is_database_available():
        logger.warning("Database not available - returning sample data")
        return _get_fallback_data()
    
    try:
        session = get_session()
        
        # Get recent articles
        articles = session.query(GNewsArticle).order_by(
            GNewsArticle.timestamp.desc()
        ).limit(limit).all()
        
        result = []
        for article in articles:
            result.append({
                'id': article.id,
                'title': article.title,
                'description': article.description or '',
                'content': article.content or '',
                'url': article.url,
                'image': article.image or '',
                'published_at': article.published_at.isoformat() if article.published_at else None,
                'source_name': article.source_name or '',
                'author': article.author or '',
                'timestamp': article.timestamp.isoformat() if article.timestamp else None,
                'word_count': article.word_count or 0
            })
        
        session.close()
        logger.info(f"Retrieved {len(result)} articles from database")
        return result
        
    except SQLAlchemyError as e:
        logger.error(f"Database error retrieving articles: {e}")
        if 'session' in locals():
            session.close()
        return _get_fallback_data()
    except Exception as e:
        logger.error(f"Unexpected error retrieving articles: {e}")
        if 'session' in locals():
            session.close()
        return _get_fallback_data()

def article_hash_exists(article_hash: str) -> bool:
    """
    Check if an article hash exists in the database
    
    Args:
        article_hash: SHA-256 hash of article content
        
    Returns:
        bool: True if hash exists, False otherwise
    """
    if not article_hash:
        return False
    
    if not is_database_available():
        logger.debug("Database not available - cannot check hash")
        return False
    
    try:
        session = get_session()
        result = session.query(ArticleHash).filter_by(hash=article_hash).first()
        session.close()
        exists = result is not None
        logger.debug(f"Hash check: {article_hash[:16]}... -> {'exists' if exists else 'not found'}")
        return exists
        
    except SQLAlchemyError as e:
        logger.error(f"Database error checking hash: {e}")
        if 'session' in locals():
            session.close()
        return False
    except Exception as e:
        logger.error(f"Unexpected error checking hash: {e}")
        if 'session' in locals():
            session.close()
        return False

def store_article_hash(article_hash: str, article_id: Optional[int] = None) -> bool:
    """
    Store an article hash in the database
    
    Args:
        article_hash: SHA-256 hash of article content
        article_id: Optional article ID to associate with the hash
        
    Returns:
        bool: True if successful, False otherwise
    """
    if not article_hash:
        return False
    
    if not is_database_available():
        logger.debug("Database not available - cannot store hash")
        return False
    
    try:
        session = get_session()
        
        # Check if hash already exists
        existing = session.query(ArticleHash).filter_by(hash=article_hash).first()
        
        if not existing:
            new_hash = ArticleHash(
                hash=article_hash,
                article_id=article_id,
                hash_type='sha256',
                content_type='full'
            )
            session.add(new_hash)
            session.commit()
            logger.debug(f"✅ Stored new hash: {article_hash[:16]}...")
            stored = True
        else:
            logger.debug(f"ℹ Hash already exists: {article_hash[:16]}...")
            stored = True  # Consider it successful even if it already exists
        
        session.close()
        return stored
        
    except SQLAlchemyError as e:
        logger.error(f"Database error storing hash: {e}")
        if 'session' in locals():
            session.rollback()
            session.close()
        return False
    except Exception as e:
        logger.error(f"Unexpected error storing hash: {e}")
        if 'session' in locals():
            session.close()
        return False

def generate_article_hash(title: str, content: str) -> str:
    """
    Generate a SHA-256 hash for an article
    
    Args:
        title: Article title
        content: Article content
        
    Returns:
        SHA-256 hash as hexadecimal string
    """
    # Normalize text for consistent hashing
    normalized_content = f"{title.strip()} {content.strip()}".lower()
    return hashlib.sha256(normalized_content.encode('utf-8')).hexdigest()

def get_database_stats() -> Dict[str, Any]:
    """
    Get database statistics
    
    Returns:
        Dictionary with database statistics
    """
    if not is_database_available():
        return {
            'database_available': False,
            'error': 'Database not available',
            'total_articles': 0,
            'total_hashes': 0
        }
    
    try:
        session = get_session()
        
        stats = {
            'database_available': True,
            'total_articles': session.query(GNewsArticle).count(),
            'total_hashes': session.query(ArticleHash).count(),
            'recent_articles': session.query(GNewsArticle).filter(
                GNewsArticle.timestamp >= datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)
            ).count()
        }
        
        session.close()
        return stats
        
    except Exception as e:
        logger.error(f"Error getting database stats: {e}")
        if 'session' in locals():
            session.close()
        return {
            'database_available': False,
            'error': str(e),
            'total_articles': 0,
            'total_hashes': 0
        }

def test_database_connection() -> bool:
    """
    Test database connection
    
    Returns:
        bool: True if connection successful, False otherwise
    """
    if not DATABASE_AVAILABLE:
        logger.error("Database not initialized")
        return False
    
    try:
        session = get_session()
        # Try a simple query
        result = session.execute(text("SELECT 1")).fetchone()
        session.close()
        logger.info("✅ Database connection test successful")
        return True
        
    except Exception as e:
        logger.error(f"❌ Database connection test failed: {e}")
        if 'session' in locals():
            session.close()
        return False

def _get_fallback_data() -> List[Dict[str, Any]]:
    """Return sample data when database is not available"""
    return [
        {
            'id': 1,
            'title': 'Climate Change Report Shows Rising Global Temperatures',
            'description': 'New scientific report confirms accelerating climate change with rising global temperatures.',
            'content': 'A comprehensive climate report released today by international scientists confirms that global temperatures continue to rise at an accelerating pace. The report shows significant impacts on weather patterns worldwide.',
            'url': 'https://example-news.com/climate-report',
            'image': '',
            'published_at': datetime.now().isoformat(),
            'source_name': 'Science Daily',
            'author': 'Dr. Jane Smith',
            'timestamp': datetime.now().isoformat(),
            'word_count': 150
        },
        {
            'id': 2,
            'title': 'New Technology Breakthrough in Renewable Energy',
            'description': 'Scientists develop more efficient solar panels that could revolutionize renewable energy.',
            'content': 'Researchers have announced a breakthrough in solar panel technology achieving 45% efficiency rates, significantly higher than traditional panels. This advancement could accelerate renewable energy adoption.',
            'url': 'https://example-tech.com/solar-breakthrough',
            'image': '',
            'published_at': datetime.now().isoformat(),
            'source_name': 'Tech Innovation Weekly',
            'author': 'John Doe',
            'timestamp': datetime.now().isoformat(),
            'word_count': 120
        }
    ]

def cleanup_old_articles(days_old: int = 30) -> int:
    """
    Clean up old articles from the database
    
    Args:
        days_old: Articles older than this many days will be deleted
        
    Returns:
        Number of articles deleted
    """
    if not is_database_available():
        logger.warning("Database not available - cannot cleanup articles")
        return 0
    
    try:
        session = get_session()
        
        cutoff_date = datetime.now() - timedelta(days=days_old)
        
        # Delete old articles
        deleted_count = session.query(GNewsArticle).filter(
            GNewsArticle.timestamp < cutoff_date
        ).delete()
        
        session.commit()
        session.close()
        
        logger.info(f"🗑 Cleaned up {deleted_count} old articles")
        return deleted_count
        
    except Exception as e:
        logger.error(f"Error cleaning up old articles: {e}")
        if 'session' in locals():
            session.rollback()
            session.close()
        return 0

# Initialize database and log status
if __name__ == "__main__":
    print("🔧 Testing database repository...")
    
    if test_database_connection():
        print("✅ Database connection successful")
        stats = get_database_stats()
        print(f"📊 Database stats: {stats}")
        
        # Test basic operations
        sample_articles = [
            {
                'title': 'Test Article',
                'description': 'This is a test article',
                'content': 'Test content for the article',
                'url': 'https://test.com/article1',
                'source_name': 'Test Source'
            }
        ]
        
        if save_articles(sample_articles):
            print("✅ Article save test successful")
        
        articles = get_scraped_data(limit=5)
        print(f"✅ Retrieved {len(articles)} articles")
        
    else:
        print("❌ Database connection failed")
        print("🔄 Testing fallback functionality...")
        
        fallback_data = get_scraped_data()
        print(f"✅ Fallback data: {len(fallback_data)} sample articles")