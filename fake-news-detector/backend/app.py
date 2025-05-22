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

from flask import Flask, request, jsonify
from flask_cors import CORS

# Import our custom modules
try:
    from models.nlp_pipeline import process_article, NewsPipeline
    from models.credibility import score_credibility, CredibilityScorer
    from utils.similarity import compute_text_score, image_similarity_score
    from database.repository import (
        save_articles, get_scraped_data, article_hash_exists, 
        store_article_hash
    )
    from database.models import GNewsArticle, ArticleHash
except ImportError as e:
    print(f"Warning: Could not import some modules: {e}")
    print("Some features may not be available.")

# Initialize Flask app
app = Flask(__name__)

# Enable CORS for frontend communication
CORS(app, origins=["chrome-extension://*", "http://localhost:3000", "http://localhost:5173"])

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Initialize NLP pipeline and credibility scorer
try:
    nlp_pipeline = NewsPipeline()
    credibility_scorer = CredibilityScorer()
    logger.info("NLP pipeline and credibility scorer initialized successfully")
except Exception as e:
    logger.error(f"Error initializing NLP components: {e}")
    nlp_pipeline = None
    credibility_scorer = None

# Configuration
app.config['SECRET_KEY'] = os.environ.get('SECRET_KEY', 'dev-secret-key-change-in-production')
app.config['DEBUG'] = os.environ.get('FLASK_DEBUG', 'False').lower() == 'true'


@app.route('/', methods=['GET'])
def health_check():
    """Health check endpoint."""
    return jsonify({
        "status": "healthy",
        "service": "Fake News Detection API",
        "version": "1.0.0",
        "timestamp": datetime.now().isoformat(),
        "components": {
            "nlp_pipeline": nlp_pipeline is not None,
            "credibility_scorer": credibility_scorer is not None
        }
    })


@app.route('/analyze', methods=['POST'])
def analyze_article():
    """
    Main endpoint for analyzing news articles.
    Expected input format from Chrome extension:
    {
        "text": "article content",
        "metadata": {
            "title": "article title",
            "description": "meta description", 
            "author": "author name",
            "url": "article url"
        },
        "images": ["image_url1", "image_url2"]
    }
    """
    try:
        # Get request data
        data = request.get_json()
        if not data:
            return jsonify({"error": "No data provided"}), 400
        
        # Extract article information
        article_text = data.get('text', '')
        metadata = data.get('metadata', {})
        images = data.get('images', [])
        
        # Basic validation
        if not article_text and not metadata.get('title'):
            return jsonify({"error": "No article text or title provided"}), 400
        
        # Extract title and other metadata
        title = metadata.get('title', '')
        description = metadata.get('description', '')
        author = metadata.get('author', '')
        url = metadata.get('url', '')
        
        logger.info(f"Analyzing article: {title[:100]}...")
        
        # Combine text sources for analysis
        full_text = f"{title} {description} {article_text}".strip()
        
        # Check if we've seen this article before (duplicate detection)
        article_hash = generate_article_hash(full_text, title, url)
        is_duplicate = False
        
        try:
            if article_hash_exists(article_hash):
                is_duplicate = True
                logger.info(f"Duplicate article detected: {article_hash}")
        except Exception as e:
            logger.warning(f"Could not check for duplicates: {e}")
        
        # Process through NLP pipeline
        if nlp_pipeline:
            try:
                nlp_analysis = nlp_pipeline.process_article(
                    article_text=full_text,
                    title=title,
                    metadata={
                        'url': url,
                        'author': author,
                        'description': description,
                        'images': images
                    }
                )
            except Exception as e:
                logger.error(f"NLP pipeline error: {e}")
                # Fallback to basic analysis
                nlp_analysis = create_fallback_analysis(full_text, title, metadata)
        else:
            nlp_analysis = create_fallback_analysis(full_text, title, metadata)
        
        # Score credibility
        if credibility_scorer:
            try:
                credibility_analysis = credibility_scorer.score_article(nlp_analysis)
            except Exception as e:
                logger.error(f"Credibility scoring error: {e}")
                credibility_analysis = create_fallback_credibility(nlp_analysis)
        else:
            credibility_analysis = create_fallback_credibility(nlp_analysis)
        
        # Check for similar articles in database
        similarity_results = []
        try:
            similar_articles = find_similar_articles(full_text, title)
            similarity_results = similar_articles[:3]  # Top 3 similar articles
        except Exception as e:
            logger.warning(f"Could not check for similar articles: {e}")
        
        # Store article hash for future duplicate detection
        try:
            if not is_duplicate:
                store_article_hash(article_hash)
        except Exception as e:
            logger.warning(f"Could not store article hash: {e}")
        
        # Prepare response
        response = {
            "credibility_score": credibility_analysis.get('credibility_score', 50),
            "credibility_label": credibility_analysis.get('credibility_label', 'unknown'),
            "sentiment": nlp_analysis.get('sentiment', {}).get('label', 'neutral'),
            "sentiment_score": nlp_analysis.get('sentiment', {}).get('polarity', 0),
            "bias_markers": list(nlp_analysis.get('bias_markers', {}).keys()),
            "explanation": credibility_analysis.get('explanation', 'Analysis completed with limited information.'),
            "fact_check_links": credibility_analysis.get('fact_check_links', []),
            "named_entities": nlp_analysis.get('named_entities', {}),
            "key_phrases": nlp_analysis.get('key_phrases', []),
            "is_duplicate": is_duplicate,
            "similar_articles": similarity_results,
            "analysis_metadata": {
                "processed_at": datetime.now().isoformat(),
                "word_count": nlp_analysis.get('word_count', 0),
                "processing_time": "< 1 second"
            }
        }
        
        logger.info(f"Analysis completed. Credibility score: {response['credibility_score']}")
        return jsonify(response)
        
    except Exception as e:
        logger.error(f"Error in analyze_article: {e}")
        logger.error(traceback.format_exc())
        return jsonify({
            "error": "Internal server error during analysis",
            "message": str(e) if app.config['DEBUG'] else "Please try again later"
        }), 500


@app.route('/similarity', methods=['POST'])
def check_similarity():
    """
    Endpoint for checking similarity between two articles.
    Expected input:
    {
        "text1": "first article text",
        "text2": "second article text",
        "title1": "first article title",
        "title2": "second article title"
    }
    """
    try:
        data = request.get_json()
        if not data:
            return jsonify({"error": "No data provided"}), 400
        
        text1 = data.get('text1', '')
        text2 = data.get('text2', '')
        title1 = data.get('title1', '')
        title2 = data.get('title2', '')
        
        if not text1 or not text2:
            return jsonify({"error": "Both texts are required"}), 400
        
        # Calculate text similarity
        text_similarity = compute_text_score(text1, text2)
        title_similarity = compute_text_score(title1, title2) if title1 and title2 else 0
        
        # Overall similarity (weighted average)
        overall_similarity = (text_similarity * 0.8) + (title_similarity * 0.2)
        
        return jsonify({
            "text_similarity": 100 - text_similarity,  # Convert distance to similarity
            "title_similarity": 100 - title_similarity,
            "overall_similarity": 100 - overall_similarity,
            "is_similar": overall_similarity < 20,  # Less than 20% difference = similar
            "similarity_threshold": 80
        })
        
    except Exception as e:
        logger.error(f"Error in check_similarity: {e}")
        return jsonify({"error": "Error calculating similarity"}), 500


@app.route('/sources', methods=['GET'])
def get_source_reputation():
    """Get reputation information for news sources."""
    try:
        domain = request.args.get('domain')
        if not domain:
            return jsonify({"error": "Domain parameter required"}), 400
        
        if credibility_scorer:
            reputation = credibility_scorer._get_source_reputation(domain)
        else:
            reputation = {"credibility": 50, "bias": "unknown", "factual_reporting": "mixed"}
        
        return jsonify({
            "domain": domain,
            "reputation": reputation
        })
        
    except Exception as e:
        logger.error(f"Error getting source reputation: {e}")
        return jsonify({"error": "Error retrieving source information"}), 500


@app.route('/batch-analyze', methods=['POST'])
def batch_analyze():
    """
    Endpoint for analyzing multiple articles at once.
    Expected input:
    {
        "articles": [
            {"text": "...", "title": "...", "metadata": {...}},
            ...
        ]
    }
    """
    try:
        data = request.get_json()
        if not data or 'articles' not in data:
            return jsonify({"error": "Articles array required"}), 400
        
        articles = data['articles']
        if len(articles) > 10:  # Limit batch size
            return jsonify({"error": "Maximum 10 articles per batch"}), 400
        
        results = []
        for i, article in enumerate(articles):
            try:
                # Process each article similar to single analysis
                article_text = article.get('text', '')
                title = article.get('title', '')
                metadata = article.get('metadata', {})
                
                # Basic NLP processing
                if nlp_pipeline:
                    nlp_analysis = nlp_pipeline.process_article(article_text, title, metadata)
                else:
                    nlp_analysis = create_fallback_analysis(article_text, title, metadata)
                
                # Credibility scoring
                if credibility_scorer:
                    credibility_analysis = credibility_scorer.score_article(nlp_analysis)
                else:
                    credibility_analysis = create_fallback_credibility(nlp_analysis)
                
                results.append({
                    "index": i,
                    "title": title,
                    "credibility_score": credibility_analysis.get('credibility_score', 50),
                    "sentiment": nlp_analysis.get('sentiment', {}).get('label', 'neutral'),
                    "status": "success"
                })
                
            except Exception as e:
                logger.error(f"Error processing article {i}: {e}")
                results.append({
                    "index": i,
                    "title": article.get('title', 'Unknown'),
                    "status": "error",
                    "error": str(e)
                })
        
        return jsonify({
            "results": results,
            "processed_count": len([r for r in results if r["status"] == "success"]),
            "error_count": len([r for r in results if r["status"] == "error"])
        })
        
    except Exception as e:
        logger.error(f"Error in batch_analyze: {e}")
        return jsonify({"error": "Error in batch processing"}), 500


def generate_article_hash(text: str, title: str, url: str) -> str:
    """Generate a hash for duplicate detection."""
    content = f"{title}{text}{url}".strip()
    return hashlib.sha256(content.encode('utf-8')).hexdigest()


def create_fallback_analysis(text: str, title: str, metadata: Dict[str, Any]) -> Dict[str, Any]:
    """Create basic analysis when NLP pipeline is unavailable."""
    word_count = len(text.split()) if text else 0
    
    return {
        'original_text': text,
        'title': title,
        'metadata': metadata,
        'word_count': word_count,
        'named_entities': {'PERSON': [], 'ORG': [], 'GPE': [], 'LOC': [], 'DATE': [], 'MISC': []},
        'key_phrases': [],
        'sentiment': {'polarity': 0, 'label': 'neutral', 'confidence': 0.5},
        'bias_markers': {
            'political_left': 0, 'political_right': 0, 'emotional': 0,
            'clickbait': 0, 'exaggeration': 0, 'uncertainty': 0
        },
        'features': {
            'text_stats': {
                'sentence_count': len(text.split('.')) if text else 0,
                'word_count': word_count,
                'avg_word_length': 5,
                'avg_sentence_length': 15
            }
        }
    }


def create_fallback_credibility(nlp_analysis: Dict[str, Any]) -> Dict[str, Any]:
    """Create basic credibility analysis when scorer is unavailable."""
    return {
        'credibility_score': 50,
        'credibility_label': 'uncertain',
        'explanation': 'Basic analysis completed. For detailed credibility assessment, ensure all NLP components are properly installed.',
        'fact_check_links': [
            'https://www.snopes.com',
            'https://www.politifact.com',
            'https://factcheck.org'
        ]
    }


def find_similar_articles(text: str, title: str, limit: int = 5) -> List[Dict[str, Any]]:
    """Find similar articles in the database."""
    try:
        # Get articles from database
        scraped_articles = get_scraped_data()
        similar_articles = []
        
        for article in scraped_articles[:50]:  # Limit comparison to recent articles
            article_title = article.get('title', '')
            article_text = article.get('description', '') or article.get('content', '')
            
            if article_title and article_text:
                # Calculate similarity
                title_similarity = compute_text_score(title, article_title)
                text_similarity = compute_text_score(text[:1000], article_text[:1000])
                
                # Weighted similarity
                overall_similarity = (text_similarity * 0.7) + (title_similarity * 0.3)
                
                if overall_similarity < 30:  # Less than 30% difference = similar
                    similar_articles.append({
                        'title': article_title,
                        'similarity_score': 100 - overall_similarity,
                        'url': article.get('url', ''),
                        'source': article.get('source', 'Unknown')
                    })
        
        # Sort by similarity and return top results
        similar_articles.sort(key=lambda x: x['similarity_score'], reverse=True)
        return similar_articles[:limit]
        
    except Exception as e:
        logger.warning(f"Error finding similar articles: {e}")
        return []


@app.errorhandler(404)
def not_found(error):
    """Handle 404 errors."""
    return jsonify({"error": "Endpoint not found"}), 404


@app.errorhandler(500)
def internal_error(error):
    """Handle 500 errors."""
    logger.error(f"Internal server error: {error}")
    return jsonify({"error": "Internal server error"}), 500


if __name__ == '__main__':
    # Development server configuration
    port = int(os.environ.get('PORT', 5000))
    debug_mode = os.environ.get('FLASK_DEBUG', 'True').lower() == 'true'
    
    logger.info(f"Starting Fake News Detection API on port {port}")
    logger.info(f"Debug mode: {debug_mode}")
    
    app.run(
        host='0.0.0.0',
        port=port,
        debug=debug_mode,
        threaded=True
    )
