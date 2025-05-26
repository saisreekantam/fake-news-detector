"""
Flask Backend for Fake News Detection System
==========================================
This Flask application serves as the main API backend for the fake news detection system.
It integrates NLP processing, credibility scoring, and provides endpoints for the Chrome extension.
"""

import os
import logging
import traceback
from datetime import datetime
from typing import Dict, Any, List, Optional
import hashlib
from dotenv import load_dotenv

from flask import Flask, request, jsonify
from flask_cors import CORS
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Initialize Flask app
app = Flask(__name__)

# Configure CORS - allow both localhost and Chrome extension
CORS(app, origins=[
    "http://localhost:5173",  # Vite default
    "http://localhost:3000",  # React default
    "chrome-extension://*"    # Chrome extension
])

# Database configuration
DATABASE_URL = os.getenv('DATABASE_URL', 'postgresql://user:password@localhost:5432/fake_news_db')
engine = None
SessionLocal = None

# Import our custom modules with error handling
models_available = {
    'nlp': False,
    'credibility': False,
    'duplicate': False,
    'summarizer': False
}

try:
    from models.nlp_pipeline import process_article, NewsPipeline
    models_available['nlp'] = True
    logger.info("NLP pipeline loaded successfully")
except ImportError as e:
    logger.error(f"Could not import NLP pipeline: {e}")

try:
    from models.credibility import score_credibility, CredibilityScorer
    models_available['credibility'] = True
    logger.info("Credibility scorer loaded successfully")
except ImportError as e:
    logger.error(f"Could not import credibility scorer: {e}")

try:
    from models.duplicate import check_article_duplicate, DuplicateDetector
    models_available['duplicate'] = True
    logger.info("Duplicate detector loaded successfully")
except ImportError as e:
    logger.error(f"Could not import duplicate detector: {e}")

try:
    from models.summarizer import summarize_text
    models_available['summarizer'] = True
    logger.info("Summarizer loaded successfully")
except ImportError as e:
    logger.error(f"Could not import summarizer: {e}")

try:
    from utils.similarity import compute_text_score, image_similarity_score
    logger.info("Similarity utils loaded successfully")
except ImportError as e:
    logger.error(f"Could not import similarity utils: {e}")

try:
    from database.repository import (
        save_articles, get_scraped_data, article_hash_exists, 
        store_article_hash
    )
    from database.models import Base, GNewsArticle, ArticleHash
    logger.info("Database modules loaded successfully")
except ImportError as e:
    logger.error(f"Could not import database modules: {e}")

# Initialize database
def init_database():
    """Initialize database connection and create tables."""
    global engine, SessionLocal
    try:
        engine = create_engine(DATABASE_URL)
        SessionLocal = sessionmaker(bind=engine)
        
        # Create tables if they don't exist
        Base.metadata.create_all(engine)
        logger.info("Database initialized successfully")
        return True
    except Exception as e:
        logger.error(f"Database initialization failed: {e}")
        return False

# Initialize database on startup
db_initialized = init_database()

# Helper functions
def validate_article_data(data: Dict[str, Any]) -> tuple[bool, str]:
    """Validate incoming article data."""
    if not data:
        return False, "No data provided"
    
    if not data.get('content') and not data.get('text'):
        return False, "No article content provided"
    
    if not data.get('title'):
        return False, "No article title provided"
    
    return True, "Valid"

def extract_article_text(data: Dict[str, Any]) -> str:
    """Extract article text from various possible fields."""
    return data.get('content', data.get('text', data.get('description', '')))

# Routes
@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint."""
    status = {
        'status': 'healthy',
        'timestamp': datetime.utcnow().isoformat(),
        'models': models_available,
        'database': db_initialized
    }
    
    # Determine overall health
    if not db_initialized:
        status['status'] = 'degraded'
        status['message'] = 'Database not initialized'
    elif not any(models_available.values()):
        status['status'] = 'unhealthy'
        status['message'] = 'No models available'
    
    return jsonify(status)

@app.route('/analyze', methods=['POST', 'OPTIONS'])
def analyze_article():
    """Analyze an article for fake news detection."""
    if request.method == 'OPTIONS':
        return '', 204
    
    try:
        data = request.json
        
        # Validate input
        is_valid, message = validate_article_data(data)
        if not is_valid:
            return jsonify({'error': message}), 400
        
        # Extract data
        article_text = extract_article_text(data)
        title = data.get('title', '')
        url = data.get('url', '')
        image = data.get('image', '')
        
        # Prepare metadata
        metadata = {
            'url': url,
            'author': data.get('author', ''),
            'published_date': data.get('published_at', ''),
            'source': data.get('source_name', '')
        }
        
        response = {
            'status': 'success',
            'timestamp': datetime.utcnow().isoformat()
        }
        
        # Process with NLP pipeline
        if models_available['nlp']:
            try:
                nlp_analysis = process_article(article_text, title, metadata)
                response['nlp_analysis'] = {
                    'sentiment': nlp_analysis['sentiment'],
                    'named_entities': nlp_analysis['named_entities'],
                    'key_phrases': nlp_analysis['key_phrases'][:10],  # Top 10
                    'bias_markers': nlp_analysis['bias_markers']
                }
            except Exception as e:
                logger.error(f"NLP analysis failed: {e}")
                response['nlp_analysis'] = {'error': str(e)}
        
        # Calculate credibility score
        if models_available['credibility'] and models_available['nlp']:
            try:
                credibility_result = score_credibility(nlp_analysis)
                response['credibility'] = {
                    'score': credibility_result['credibility_score'],
                    'label': credibility_result['credibility_label'],
                    'explanation': credibility_result['explanation'],
                    'fact_check_links': credibility_result['fact_check_links']
                }
            except Exception as e:
                logger.error(f"Credibility scoring failed: {e}")
                response['credibility'] = {'error': str(e)}
        
        # Check for duplicates
        if models_available['duplicate']:
            try:
                duplicate_result = check_article_duplicate({
                    'title': title,
                    'text': article_text
                })
                response['duplicate_check'] = {
                    'is_duplicate': duplicate_result['is_duplicate'],
                    'similarity_score': duplicate_result['similarity_score'],
                    'matches': duplicate_result['matches'][:3]  # Top 3 matches
                }
            except Exception as e:
                logger.error(f"Duplicate check failed: {e}")
                response['duplicate_check'] = {'error': str(e)}
        
        # Generate summary
        if models_available['summarizer']:
            try:
                summary_result = summarize_text(article_text, title)
                response['summary'] = {
                    'text': summary_result['summary'],
                    'method': summary_result['method']
                }
            except Exception as e:
                logger.error(f"Summarization failed: {e}")
                response['summary'] = {'error': str(e)}
        
        # Store article if not duplicate and database is available
        if db_initialized and not response.get('duplicate_check', {}).get('is_duplicate', False):
            try:
                save_articles([{
                    'title': title,
                    'description': article_text[:500],  # First 500 chars
                    'content': article_text,
                    'url': url,
                    'image': image,
                    'publishedAt': data.get('published_at'),
                    'source': {'name': data.get('source_name', '')}
                }])
            except Exception as e:
                logger.error(f"Failed to save article: {e}")
        
        # Simplified response for Chrome extension
        return jsonify({
            'credibility_score': response.get('credibility', {}).get('score', 50),
            'credibility_label': response.get('credibility', {}).get('label', 'unknown'),
            'sentiment': response.get('nlp_analysis', {}).get('sentiment', {}).get('label', 'neutral'),
            'bias_markers': list(response.get('nlp_analysis', {}).get('bias_markers', {}).keys()),
            'explanation': response.get('credibility', {}).get('explanation', 'Analysis incomplete'),
            'fact_check_links': response.get('credibility', {}).get('fact_check_links', []),
            'is_duplicate': response.get('duplicate_check', {}).get('is_duplicate', False),
            'summary': response.get('summary', {}).get('text', '')[:200]  # First 200 chars
        })
        
    except Exception as e:
        logger.error(f"Analysis endpoint error: {traceback.format_exc()}")
        return jsonify({
            'error': 'Analysis failed',
            'message': str(e)
        }), 500

@app.route('/check-duplicate', methods=['POST'])
def check_duplicate():
    """Check if an article is a duplicate."""
    try:
        data = request.json
        
        if not models_available['duplicate']:
            return jsonify({'error': 'Duplicate detection not available'}), 503
        
        # Validate input
        is_valid, message = validate_article_data(data)
        if not is_valid:
            return jsonify({'error': message}), 400
        
        # Check for duplicates
        result = check_article_duplicate({
            'title': data.get('title', ''),
            'text': extract_article_text(data)
        })
        
        return jsonify(result)
        
    except Exception as e:
        logger.error(f"Duplicate check endpoint error: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/summarize', methods=['POST'])
def summarize():
    """Generate a summary of an article."""
    try:
        data = request.json
        
        if not models_available['summarizer']:
            return jsonify({'error': 'Summarization not available'}), 503
        
        # Validate input
        if not data or not data.get('text'):
            return jsonify({'error': 'No text provided'}), 400
        
        # Generate summary
        result = summarize_text(
            data['text'], 
            data.get('title', ''),
            data.get('method', 'auto')
        )
        
        return jsonify(result)
        
    except Exception as e:
        logger.error(f"Summarization endpoint error: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/sources', methods=['GET'])
def get_sources():
    """Get list of known news sources and their credibility."""
    try:
        # This would normally query the database
        # For now, return some example sources
        sources = [
            {'domain': 'reuters.com', 'credibility': 92, 'bias': 'neutral'},
            {'domain': 'apnews.com', 'credibility': 90, 'bias': 'neutral'},
            {'domain': 'bbc.com', 'credibility': 88, 'bias': 'slight left'},
            {'domain': 'nytimes.com', 'credibility': 85, 'bias': 'left'},
            {'domain': 'foxnews.com', 'credibility': 75, 'bias': 'right'}
        ]
        
        return jsonify({'sources': sources})
        
    except Exception as e:
        logger.error(f"Sources endpoint error: {e}")
        return jsonify({'error': str(e)}), 500

@app.errorhandler(404)
def not_found(error):
    """Handle 404 errors."""
    return jsonify({'error': 'Endpoint not found'}), 404

@app.errorhandler(500)
def internal_error(error):
    """Handle 500 errors."""
    logger.error(f"Internal error: {error}")
    return jsonify({'error': 'Internal server error'}), 500

# CLI commands for setup
@app.cli.command()
def init_db():
    """Initialize the database."""
    if init_database():
        print("Database initialized successfully")
    else:
        print("Database initialization failed")

@app.cli.command()
def test_models():
    """Test if all models are loaded correctly."""
    print("Model availability:")
    for model, available in models_available.items():
        status = "✓" if available else "✗"
        print(f"  {status} {model}")

if __name__ == '__main__':
    # Development server
    port = int(os.getenv('PORT', 5000))
    debug = os.getenv('FLASK_ENV') == 'development'
    
    print(f"Starting Fake News Detection API on port {port}")
    print(f"Debug mode: {debug}")
    print(f"Models available: {models_available}")
    
    app.run(
        host='127.0.0.1',
        port=port,
        debug=debug
    )
