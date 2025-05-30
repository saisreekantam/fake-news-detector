"""
Flask Backend for Fake News Detection System
==========================================
This Flask application serves as the main API backend for the fake news detection system.
It integrates NLP processing, credibility scoring, duplicate detection, and provides endpoints for the Chrome extension.
"""

import os
import logging
import traceback
import json
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional
import hashlib
import asyncio
from functools import wraps
import sys

# Add the current directory to Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from flask import Flask, request, jsonify, make_response
from flask_cors import CORS
from dotenv import load_dotenv
import nltk

# Load environment variables
load_dotenv()

# Download required NLTK data
try:
    nltk.data.find('tokenizers/punkt')
    nltk.data.find('corpora/stopwords')
    nltk.data.find('corpora/wordnet')
except LookupError:
    print("Downloading required NLTK data...")
    nltk.download('punkt')
    nltk.download('stopwords')
    nltk.download('wordnet')

# Import our custom modules
try:
    from models.nlp_pipeline import process_article, NewsPipeline
    #loaded_modules['nlp_pipeline'] = True
    print("✅ NLP Pipeline module loaded successfully")
except ImportError as e:
    print(f"❌ Failed to load NLP Pipeline: {e}")

try:
    from models.credibility import score_credibility, CredibilityScorer
    
    #loaded_modules['credibility'] = True
    print("✅ Credibility module loaded successfully")
except ImportError as e:
    print(f"❌ Failed to load Credibility module: {e}")
try:
    from models.duplicate import generate_article_hash,check_article_duplicate
    print("✅ Duplicate Detection module loaded successfully")
except ImportError as e:
    print(f"❌ Failed to load Duplicate Detection: {e}")
    check_article_duplicate = None

try:
    from utils.similarity import compute_text_score, image_similarity_score
    from utils.text_processor import (
            get_text_features, preprocess_complete, clean_text, 
            is_clickbait, extract_keywords, calculate_text_stats
        )
    #loaded_modules['similarity'] = True
    print("✅ Similarity module loaded successfully")
except ImportError as e:
    print(f"❌ Failed to load Similarity module: {e}")

try:
    from models.summarizer import summarize_text, summarize_for_comparison
    #loaded_modules['summarizer'] = True
    print("✅ Summarizer module loaded successfully")
except ImportError as e:
    print(f"❌ Failed to load Summarizer module: {e}")

try:
    from database.repository import (
        save_articles, get_scraped_data, article_hash_exists, 
        store_article_hash
    )
    from database.models import GNewsArticle, ArticleHash
    #loaded_modules['database'] = True
    print("✅ Database modules loaded successfully")
except ImportError as e:
    print(f"❌ Failed to load Database modules: {e}")


# Initialize Flask app
app = Flask(__name__)

# Configure CORS for Chrome extension
CORS(app, origins=['chrome-extension://', 'http://localhost:', 'https://localhost:*'])

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Global variables for caching
pipeline_cache = {}
last_cache_update = None
CACHE_DURATION = timedelta(hours=1)

def require_modules(f):
    """Decorator to check if required modules are loaded"""
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if not MODULES_LOADED:
            return jsonify({
                "error": "Required modules not available",
                "message": "Please ensure all dependencies are installed"
            }), 503
        return f(*args, **kwargs)
    return decorated_function

def validate_json_data(required_fields=None):
    """Decorator to validate JSON request data"""
    def decorator(f):
        @wraps(f)
        def decorated_function(*args, **kwargs):
            if not request.is_json:
                return jsonify({"error": "Request must be JSON"}), 400
            
            data = request.get_json()
            if not data:
                return jsonify({"error": "No JSON data provided"}), 400
            
            if required_fields:
                missing_fields = [field for field in required_fields if field not in data]
                if missing_fields:
                    return jsonify({
                        "error": "Missing required fields",
                        "missing_fields": missing_fields
                    }), 400
            
            return f(data, *args, **kwargs)
        return decorated_function
    return decorator

# Add this at the top of your file after app creation
initialized = False

@app.before_request
def initialize_once():
    global initialized
    if not initialized:
        # Configure logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        
        # Initialize NLP pipeline
        try:
            from models.nlp_pipeline import NewsPipeline
            app.nlp_pipeline = NewsPipeline()
            logging.info("✅ NLP Pipeline initialized successfully")
        except Exception as e:
            logging.error(f"❌ Failed to initialize NLP Pipeline: {e}")
        
        # Initialize credibility scorer
        try:
            from models.credibility import CredibilityScorer
            app.credibility_scorer = CredibilityScorer()
            logging.info("✅ Credibility Scorer initialized successfully")
        except Exception as e:
            logging.error(f"❌ Failed to initialize Credibility Scorer: {e}")
        
        # Test database connection
        try:
            from database.repository import SessionLocal
            session = SessionLocal()
            session.close()
            logging.info("✅ Database connection verified")
        except Exception as e:
            logging.error(f"⚠ Database connection issue: {e}")
        
        logging.info("🚀 Fake News Detection API initialized successfully")
        initialized = True

# ============================================================================
# HEALTH AND STATUS ENDPOINTS
# ============================================================================

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    try:
        # Test database connection
        session = SessionLocal()
        session.execute("SELECT 1")
        session.close()
        db_status = "healthy"
    except Exception as e:
        db_status = f"unhealthy: {str(e)}"
    
    return jsonify({
        "status": "healthy" if MODULES_LOADED else "partial",
        "timestamp": datetime.now().isoformat(),
        "modules_loaded": MODULES_LOADED,
        "database_status": db_status,
        "version": "1.0.0"
    })

@app.route('/status', methods=['GET'])
def get_status():
    """Get detailed system status"""
    try:
        # Get database stats
        session = SessionLocal()
        article_count = session.query(GNewsArticle).count()
        hash_count = session.query(ArticleHash).count()
        session.close()
        
        return jsonify({
            "system_status": "operational" if MODULES_LOADED else "degraded",
            "modules_loaded": MODULES_LOADED,
            "database": {
                "status": "connected",
                "articles_stored": article_count,
                "hashes_stored": hash_count
            },
            "features": {
                "nlp_processing": MODULES_LOADED,
                "credibility_scoring": MODULES_LOADED,
                "duplicate_detection": MODULES_LOADED,
                "text_summarization": MODULES_LOADED
            },
            "timestamp": datetime.now().isoformat()
        })
    except Exception as e:
        print(f"Error getting status: {e}")
        return jsonify({"error": "Failed to get system status"}), 500

# ============================================================================
# MAIN ANALYSIS ENDPOINTS
# ============================================================================

@app.route('/analyze', methods=['POST'])
#@require_modules
#@validate_json_data(['content'])
def analyze_article():
    """Comprehensive article analysis endpoint"""
    try:
        # Extract article information
        print("Inside the function!")  # Add this
        data = request.get_json(force=True)
        print(data)
        # return jsonify({"status": "ok"})
        # print(data)
        title = data.get('title', '').strip()
        content = data.get('content', '').strip()
        url = data.get('url', '').strip()
        image_url = data.get('image', '').strip()
        source_name = data.get('source_name', '').strip()
        published_at = data.get('published_at', '')
        author = data.get('author', '').strip()
        
        if len(content) < 50:
            return jsonify({"error": "Content too short for meaningful analysis"}), 400
        
        # Prepare metadata
        metadata = {
            'url': url,
            'source_name': source_name,
            'published_at': published_at,
            'author': author,
            'image_url': image_url
        }
        
        print(f"Analyzing article: {title[:50]}...")
        
        # Step 1: NLP Processing
        nlp_result = process_article(content, title, metadata)
        
        # Step 2: Credibility Scoring
        credibility_result = score_credibility(nlp_result)
        
        # Step 3: Duplicate Detection
        if check_article_duplicate:
            duplicate_result = check_article_duplicate({
                'title': title,
                'text': content
            })
        else:
            duplicate_result = {'is_duplicate': False, 'similarity_score': 0, 'matches': []}
        
        # Step 4: Text Summarization
        summary_result = summarize_text(content, title, method="auto")
        
        # Step 5: Additional Text Analysis
        text_features = get_text_features(content)
        clickbait_check = is_clickbait(title)
        
        # Compile comprehensive response
        response = {
            "analysis_id": generate_article_hash(title, content),
            "input": {
                "title": title,
                "url": url,
                "source_name": source_name,
                "author": author,
                "word_count": len(content.split())
            },
            "credibility": {
                "score": credibility_result.get('credibility_score', 0),
                "label": credibility_result.get('credibility_label', 'unknown'),
                "explanation": credibility_result.get('explanation', ''),
                "category_scores": credibility_result.get('category_scores', {}),
                "fact_check_links": credibility_result.get('fact_check_links', [])
            },
            "content_analysis": {
                "sentiment": nlp_result.get('sentiment', {}),
                "bias_markers": nlp_result.get('bias_markers', {}),
                "named_entities": nlp_result.get('named_entities', {}),
                "key_phrases": nlp_result.get('key_phrases', []),
                "language_features": text_features.get('language_features', {}),
                "statistics": text_features.get('statistics', {})
            },
            "duplicate_check": {
                "is_duplicate": duplicate_result.get('is_duplicate', False),
                "similarity_score": duplicate_result.get('similarity_score', 0),
                "matches": duplicate_result.get('matches', [])[:3],  # Top 3 matches
                "content_hash": duplicate_result.get('content_hash', '')
            },
            "summary": {
                "text": summary_result.get('summary', ''),
                "method": summary_result.get('method', 'unknown'),
                "compression_ratio": summary_result.get('compression_ratio', 0)
            },
            "clickbait_analysis": {
                "is_clickbait": clickbait_check[0],
                "matched_patterns": clickbait_check[1]
            },
            "metadata": {
                "analysis_timestamp": datetime.now().isoformat(),
                "processing_version": "1.0.0"
            }
        }
        
        print(f"Analysis completed. Credibility: {response['credibility']['score']}/100")
        print("Sending Response : ", response)
        return jsonify(response)
        
    except Exception as e:
        print(f"Error analyzing article: {e}")
        print(traceback.format_exc())
        return jsonify({
            "error": "Analysis failed",
            "message": str(e),
            "timestamp": datetime.now().isoformat()
        }), 500

@app.route('/analyze/batch', methods=['POST'])
@require_modules
@validate_json_data(['articles'])
def analyze_batch(data):
    """Batch analysis endpoint for multiple articles"""
    try:
        articles = data.get('articles', [])
        
        if not articles:
            return jsonify({"error": "No articles provided"}), 400
        
        if len(articles) > 10:  # Limit batch size
            return jsonify({"error": "Maximum 10 articles per batch"}), 400
        
        results = []
        
        for i, article in enumerate(articles):
            try:
                title = article.get('title', '').strip()
                content = article.get('content', '').strip()
                
                if len(content) < 50:
                    results.append({
                        "index": i,
                        "error": "Content too short",
                        "title": title
                    })
                    continue
                
                # Process single article
                nlp_result = process_article(content, title, article)
                credibility_result = score_credibility(nlp_result)
                duplicate_result = check_article_duplicate({
                    'title': title,
                    'text': content
                })
                
                results.append({
                    "index": i,
                    "title": title,
                    "credibility_score": credibility_result.get('credibility_score', 0),
                    "credibility_label": credibility_result.get('credibility_label', 'unknown'),
                    "is_duplicate": duplicate_result.get('is_duplicate', False),
                    "similarity_score": duplicate_result.get('similarity_score', 0),
                    "sentiment": nlp_result.get('sentiment', {}).get('label', 'neutral')
                })
                
            except Exception as e:
                results.append({
                    "index": i,
                    "error": str(e),
                    "title": article.get('title', 'Unknown')
                })
        
        return jsonify({
            "results": results,
            "processed_count": len([r for r in results if 'error' not in r]),
            "error_count": len([r for r in results if 'error' in r]),
            "timestamp": datetime.now().isoformat()
        })
        
    except Exception as e:
        print(f"Error in batch analysis: {e}")
        return jsonify({"error": "Batch analysis failed"}), 500

# ============================================================================
# SPECIALIZED ANALYSIS ENDPOINTS
# ============================================================================

@app.route('/credibility', methods=['POST'])
@require_modules
@validate_json_data(['content'])
def analyze_credibility(data):
    """Focused credibility analysis"""
    try:
        title = data.get('title', '')
        content = data.get('content', '')
        
        nlp_result = process_article(content, title, data)
        credibility_result = score_credibility(nlp_result)
        
        return jsonify(credibility_result)
        
    except Exception as e:
        print(f"Error in credibility analysis: {e}")
        return jsonify({"error": "Credibility analysis failed"}), 500

@app.route('/duplicate', methods=['POST'])
@require_modules
@validate_json_data(['content'])
def check_duplicate(data):
    """Duplicate detection endpoint"""
    try:
        title = data.get('title', '')
        content = data.get('content', '')
        
        result = check_article_duplicate({
            'title': title,
            'text': content
        })
        
        return jsonify(result)
        
    except Exception as e:
        print(f"Error in duplicate check: {e}")
        return jsonify({"error": "Duplicate check failed"}), 500

@app.route('/summarize', methods=['POST'])
@require_modules
@validate_json_data(['content'])
def summarize_article(data):
    """Text summarization endpoint"""
    try:
        title = data.get('title', '')
        content = data.get('content', '')
        method = data.get('method', 'auto')
        max_length = data.get('max_length', 150)
        
        result = summarize_text(content, title, method=method)
        
        return jsonify(result)
        
    except Exception as e:
        print(f"Error in summarization: {e}")
        return jsonify({"error": "Summarization failed"}), 500

@app.route('/similarity', methods=['POST'])
@require_modules
@validate_json_data(['text1', 'text2'])
def compare_similarity(data):
    """Text similarity comparison endpoint"""
    try:
        text1 = data.get('text1', '')
        text2 = data.get('text2', '')
        
        if not text1 or not text2:
            return jsonify({"error": "Both texts are required"}), 400
        
        similarity_score = compute_text_score(text1, text2)
        
        return jsonify({
            "similarity_score": 100 - similarity_score,  # Convert distance to similarity
            "distance_score": similarity_score,
            "interpretation": "high" if similarity_score < 20 else "medium" if similarity_score < 50 else "low"
        })
        
    except Exception as e:
        print(f"Error in similarity comparison: {e}")
        return jsonify({"error": "Similarity comparison failed"}), 500

# ============================================================================
# DATABASE AND STORAGE ENDPOINTS
# ============================================================================

@app.route('/articles', methods=['GET'])
def get_articles():
    """Get stored articles with pagination"""
    try:
        page = request.args.get('page', 1, type=int)
        per_page = min(request.args.get('per_page', 10, type=int), 50)  # Max 50 per page
        
        session = SessionLocal()
        
        # Get total count
        total = session.query(GNewsArticle).count()
        
        # Get paginated articles
        articles = session.query(GNewsArticle)\
            .order_by(GNewsArticle.timestamp.desc())\
            .offset((page - 1) * per_page)\
            .limit(per_page)\
            .all()
        
        session.close()
        
        # Convert to dict
        article_list = []
        for article in articles:
            article_list.append({
                "id": article.id,
                "title": article.title,
                "description": article.description,
                "url": article.url,
                "source_name": article.source_name,
                "published_at": article.published_at.isoformat() if article.published_at else None,
                "timestamp": article.timestamp.isoformat() if article.timestamp else None
            })
        
        return jsonify({
            "articles": article_list,
            "pagination": {
                "page": page,
                "per_page": per_page,
                "total": total,
                "pages": (total + per_page - 1) // per_page
            }
        })
        
    except Exception as e:
        print(f"Error fetching articles: {e}")
        return jsonify({"error": "Failed to fetch articles"}), 500

@app.route('/articles', methods=['POST'])
@validate_json_data(['articles'])
def store_articles(data):
    """Store new articles"""
    try:
        articles = data.get('articles', [])
        
        if not articles:
            return jsonify({"error": "No articles provided"}), 400
        
        saved_count = 0
        for article in articles:
            try:
                # Generate hash to check for duplicates
                title = article.get('title', '')
                content = article.get('content', '')
                article_hash = generate_article_hash(title, content)
                
                if not article_hash_exists(article_hash):
                    # Save article
                    save_articles([article])
                    store_article_hash(article_hash)
                    saved_count += 1
                    
            except Exception as e:
                print(f"Error saving individual article: {e}")
                continue
        
        return jsonify({
            "message": f"Saved {saved_count} new articles",
            "saved_count": saved_count,
            "total_provided": len(articles)
        })
        
    except Exception as e:
        print(f"Error storing articles: {e}")
        return jsonify({"error": "Failed to store articles"}), 500

# ============================================================================
# UTILITY AND HELPER ENDPOINTS
# ============================================================================

@app.route('/source/reputation', methods=['GET'])
@require_modules
def check_source_reputation():
    """Check reputation of a news source"""
    try:
        domain = request.args.get('domain', '').strip()
        
        if not domain:
            return jsonify({"error": "Domain parameter is required"}), 400
        
        reputation = get_source_reputation(domain)
        
        return jsonify({
            "domain": domain,
            "reputation": reputation,
            "timestamp": datetime.now().isoformat()
        })
        
    except Exception as e:
        print(f"Error checking source reputation: {e}")
        return jsonify({"error": "Failed to check source reputation"}), 500

@app.route('/keywords', methods=['POST'])
@validate_json_data(['text'])
def extract_article_keywords(data):
    """Extract keywords from text"""
    try:
        text = data.get('text', '')
        top_n = data.get('top_n', 10)
        
        if not text:
            return jsonify({"error": "Text is required"}), 400
        
        keywords = extract_keywords(text, top_n)
        
        return jsonify({
            "keywords": [{"word": word, "frequency": freq} for word, freq in keywords],
            "total_found": len(keywords)
        })
        
    except Exception as e:
        print(f"Error extracting keywords: {e}")
        return jsonify({"error": "Keyword extraction failed"}), 500

@app.route('/clickbait', methods=['POST'])
@validate_json_data(['title'])
def check_clickbait(data):
    """Check if a title is clickbait"""
    try:
        title = data.get('title', '')
        
        if not title:
            return jsonify({"error": "Title is required"}), 400
        
        is_clickbait_result, patterns = is_clickbait(title)
        
        return jsonify({
            "is_clickbait": is_clickbait_result,
            "matched_patterns": patterns,
            "confidence": len(patterns) / 10.0  # Simple confidence score
        })
        
    except Exception as e:
        print(f"Error checking clickbait: {e}")
        return jsonify({"error": "Clickbait check failed"}), 500

# ============================================================================
# ERROR HANDLERS
# ============================================================================

@app.errorhandler(404)
def not_found(error):
    return jsonify({
        "error": "Endpoint not found",
        "message": "The requested endpoint does not exist",
        "available_endpoints": [
            "/health", "/status", "/analyze", "/analyze/batch",
            "/credibility", "/duplicate", "/summarize", "/similarity",
            "/articles", "/source/reputation", "/keywords", "/clickbait"
        ]
    }), 404

@app.errorhandler(405)
def method_not_allowed(error):
    return jsonify({
        "error": "Method not allowed",
        "message": "The HTTP method is not allowed for this endpoint"
    }), 405

@app.errorhandler(500)
def internal_error(error):
    return jsonify({
        "error": "Internal server error",
        "message": "An unexpected error occurred",
        "timestamp": datetime.now().isoformat()
    }), 500

# ============================================================================
# MAIN APPLICATION RUNNER
# ============================================================================

if __name__ == '__main__':  # Fix the name check (was '_main_')
    # Print startup information
    print("\n" + "="*60)
    print("🚀 FAKE NEWS DETECTION API SERVER")
    print("="*60)
    print(f"📅 Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"🔧 Modules loaded: ✅ Yes ")
    print(f"🌐 CORS enabled for Chrome extensions")
    print(f"📊 Database integration: ✅ Active")
    print("="*60)
    print("Available endpoints:")
    print("  🏥 GET  /health - Health check")
    print("  📊 GET  /status - System status")
    print("  🔍 POST /analyze - Full article analysis")
    print("  📦 POST /analyze/batch - Batch analysis")
    print("  ⭐ POST /credibility - Credibility scoring")
    print("  👥 POST /duplicate - Duplicate detection")
    print("  📝 POST /summarize - Text summarization")
    print("  🔗 POST /similarity - Text similarity")
    print("  📚 GET  /articles - Get stored articles")
    print("  💾 POST /articles - Store articles")
    print("="*60)
    print("\n🔥 Server starting on http://localhost:5000")
    print("💡 Use Ctrl+C to stop the server\n")
    
    # Run the application
    app.run(
        debug=os.getenv('FLASK_DEBUG', 'True').lower() == 'true',
        host='0.0.0.0',
        port=int(os.getenv('FLASK_PORT', 5000)),
        threaded=True
    )