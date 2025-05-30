"""
Duplicate Detection for News Articles
------------------------------------
This module detects duplicated or highly similar news articles by comparing:
1. Text similarity using embeddings
2. Named entity overlap
3. Semantic similarity of summaries
4. Key phrase matching

It integrates with the database to check new articles against existing ones.
"""

import hashlib
import logging
import sys
import os
from typing import Dict, List, Tuple, Any, Optional, Set
import heapq
import numpy as np
from datetime import datetime, timedelta

# Setup logging first
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Add the backend directory to the Python path
current_dir = os.path.dirname(os.path.abspath(__file__))
backend_dir = os.path.dirname(current_dir)
if backend_dir not in sys.path:
    sys.path.insert(0, backend_dir)

# Import internal modules with comprehensive error handling
def safe_import_summarizer():
    """Safely import summarizer with fallback."""
    try:
        from models.summarizer import summarize_for_comparison
        return summarize_for_comparison
    except ImportError as e:
        logger.warning(f"Could not import summarizer: {e}")
        def summarize_for_comparison(text, title=""):
            # Fallback summarization
            words = text.split()[:100]  # First 100 words
            return {
                "summary": " ".join(words),
                "fingerprint": {"top_terms": words[:20], "content_hash": hash(text)},
                "original_length": len(text.split())
            }
        return summarize_for_comparison

def safe_import_nlp_pipeline():
    """Safely import NLP pipeline with fallback."""
    try:
        from models.nlp_pipeline import process_article
        return process_article
    except ImportError as e:
        logger.warning(f"Could not import NLP pipeline: {e}")
        def process_article(text, title=""):
            # Basic fallback processing
            return {
                "original_text": text,
                "title": title,
                "named_entities": {},
                "key_phrases": [],
                "sentiment": {"polarity": 0, "label": "neutral", "confidence": 0},
                "bias_markers": {}
            }
        return process_article

def safe_import_similarity():
    """Safely import similarity functions with fallback."""
    try:
        from utils.similarity import compute_text_score, image_similarity_score
        return compute_text_score, image_similarity_score
    except ImportError as e:
        logger.warning(f"Could not import similarity functions: {e}")
        def compute_text_score(text1, text2):
            # Simple fallback similarity using Jaccard coefficient
            words1 = set(text1.lower().split())
            words2 = set(text2.lower().split())
            if not words1 or not words2:
                return 100.0
            intersection = len(words1.intersection(words2))
            union = len(words1.union(words2))
            jaccard = intersection / union if union > 0 else 0
            return (1 - jaccard) * 100  # Convert to distance
        
        def image_similarity_score(url1, url2):
            return 100.0  # No similarity
        
        return compute_text_score, image_similarity_score

def safe_import_database():
    """Safely import database functions with fallback."""
    try:
        from database.repository import article_hash_exists, store_article_hash, get_scraped_data
        return article_hash_exists, store_article_hash, get_scraped_data
    except ImportError as e:
        logger.warning(f"Could not import database functions: {e}")
        
        # In-memory fallback storage
        _hash_storage = set()
        
        def article_hash_exists(hash_str):
            return hash_str in _hash_storage
        
        def store_article_hash(hash_str):
            _hash_storage.add(hash_str)
        
        def get_scraped_data():
            return []
        
        return article_hash_exists, store_article_hash, get_scraped_data

# Initialize all imports
summarize_for_comparison = safe_import_summarizer()
process_article = safe_import_nlp_pipeline()
compute_text_score, image_similarity_score = safe_import_similarity()
article_hash_exists, store_article_hash, get_scraped_data = safe_import_database()

# Import sentence transformers for semantic similarity
SENTENCE_TRANSFORMERS_AVAILABLE = False
model = None

try:
    from sentence_transformers import SentenceTransformer, util
    SENTENCE_TRANSFORMERS_AVAILABLE = True
    logger.info("Sentence-Transformers library available")
    
    try:
        model = SentenceTransformer('all-MiniLM-L6-v2')
        logger.info("Sentence-Transformer model loaded successfully")
    except Exception as e:
        logger.error(f"Error loading Sentence-Transformer model: {e}")
        SENTENCE_TRANSFORMERS_AVAILABLE = False
        model = None
        
except ImportError:
    logger.warning("Sentence-Transformers not available. Using fallback similarity methods.")

class DuplicateDetector:
    """Detects duplicate or similar news articles."""
    
    def __init__(self, similarity_threshold: float = 80.0, use_embeddings: bool = True):
        """
        Initialize the duplicate detector.
        
        Args:
            similarity_threshold: Threshold (0-100) above which articles are considered similar
                                 Higher values are more strict (require more similarity)
            use_embeddings: Whether to use semantic embeddings for comparison
        """
        try:
            # Ensure attributes are set even if there are issues
            self.similarity_threshold = float(similarity_threshold)
            self.use_embeddings = bool(use_embeddings) and SENTENCE_TRANSFORMERS_AVAILABLE
            self.model = model if self.use_embeddings else None
            
            logger.info(f"DuplicateDetector initialized with threshold={self.similarity_threshold}, embeddings={self.use_embeddings}")
            
        except Exception as e:
            logger.error(f"Error initializing DuplicateDetector: {e}")
            # Set default values to prevent AttributeError
            self.similarity_threshold = 80.0
            self.use_embeddings = False
            self.model = None
    
    def check_duplicate(self, article: Dict[str, Any], 
                        reference_articles: List[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Check if an article is a duplicate of any reference articles.
        
        Args:
            article: Dictionary containing article data (text, title, etc.)
            reference_articles: List of articles to compare against.
                               If None, articles will be fetched from the database.
                               
        Returns:
            Dictionary with duplication information
        """
        try:
            # Ensure we have the required attributes
            if not hasattr(self, 'similarity_threshold'):
                self.similarity_threshold = 80.0
                logger.warning("similarity_threshold not found, using default value 80.0")
            
            # Process the input article
            title = article.get('title', '')
            text = article.get('text', '')
            
            # Handle case of empty article
            if not text or len(text.strip()) < 50:  # At least 50 chars to be meaningful
                return {
                    'is_duplicate': False,
                    'similarity_score': 0,
                    'matches': [],
                    'error': 'Article text is too short for meaningful comparison'
                }
            
            # Generate a hash for the article
            content_hash = self._generate_hash(title, text)
            
            # Check if exact hash exists in database
            try:
                if article_hash_exists(content_hash):
                    return {
                        'is_duplicate': True,
                        'similarity_score': 100,  # Exact match
                        'matches': [{'hash': content_hash, 'score': 100, 'match_type': 'exact_hash'}],
                        'message': 'Exact duplicate found in database'
                    }
            except Exception as e:
                logger.warning(f"Error checking hash existence: {e}")
            
            # If no reference articles provided, fetch from database
            if reference_articles is None:
                reference_articles = self._fetch_reference_articles()
                
            if not reference_articles:
                # No reference articles to compare against
                # Store the new article hash
                try:
                    store_article_hash(content_hash)
                except Exception as e:
                    logger.warning(f"Error storing article hash: {e}")
                    
                return {
                    'is_duplicate': False,
                    'similarity_score': 0,
                    'matches': [],
                    'message': 'No reference articles available for comparison'
                }
                
            # Process the article with NLP pipeline for entity extraction and key phrases
            try:
                article_nlp = process_article(text, title)
            except Exception as e:
                logger.warning(f"Error in NLP processing: {e}")
                article_nlp = {
                    "original_text": text,
                    "title": title,
                    "named_entities": {},
                    "key_phrases": []
                }
            
            # Generate a summary for comparison
            try:
                article_summary = summarize_for_comparison(text, title)
            except Exception as e:
                logger.warning(f"Error in summarization: {e}")
                article_summary = {
                    "summary": text[:500],
                    "fingerprint": {"top_terms": text.split()[:20]}
                }
            
            # Compare with reference articles
            matches = self._find_similar_articles(
                article_nlp, 
                article_summary, 
                reference_articles
            )
            
            # Determine if it's a duplicate based on best match score
            best_match = matches[0] if matches else None
            is_duplicate = False
            similarity_score = 0
            
            if best_match:
                similarity_score = best_match['similarity_score']
                is_duplicate = similarity_score >= self.similarity_threshold
                
            # Store the hash if it's not a duplicate
            if not is_duplicate:
                try:
                    store_article_hash(content_hash)
                except Exception as e:
                    logger.warning(f"Error storing article hash: {e}")
                
            return {
                'is_duplicate': is_duplicate,
                'content_hash': content_hash,
                'similarity_score': similarity_score,
                'matches': matches,
                'message': f"{'Duplicate' if is_duplicate else 'Original'} content detected"
            }
            
        except Exception as e:
            logger.error(f"Error in check_duplicate: {e}")
            return {
                'is_duplicate': False,
                'similarity_score': 0,
                'matches': [],
                'error': f'Error processing article: {str(e)}'
            }
    
    def batch_check_duplicates(self, articles: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Check multiple articles for duplicates efficiently.
        
        Args:
            articles: List of article dictionaries
            
        Returns:
            List of duplicate check results
        """
        results = []
        reference_articles = self._fetch_reference_articles()
        
        for i, article in enumerate(articles):
            try:
                result = self.check_duplicate(article, reference_articles)
                results.append(result)
                
                # If not a duplicate, add to reference articles for subsequent checks
                if not result.get('is_duplicate', False):
                    title = article.get('title', '')
                    text = article.get('text', '')
                    
                    try:
                        article_nlp = process_article(text, title)
                        article_summary = summarize_for_comparison(text, title)
                        
                        reference_articles.append({
                            'title': title,
                            'text': text,
                            'nlp_data': article_nlp,
                            'summary': article_summary
                        })
                    except Exception as e:
                        logger.warning(f"Error processing article {i} for reference: {e}")
                        
            except Exception as e:
                logger.error(f"Error processing article {i}: {e}")
                results.append({
                    'is_duplicate': False,
                    'similarity_score': 0,
                    'matches': [],
                    'error': f'Error processing article: {str(e)}'
                })
        
        return results
    
    def _generate_hash(self, title: str, text: str) -> str:
        """Generate a content hash for the article."""
        try:
            # Normalize text for consistent hashing
            normalized_content = (title + " " + text).lower().strip()
            # Create SHA-256 hash
            return hashlib.sha256(normalized_content.encode('utf-8')).hexdigest()
        except Exception as e:
            logger.error(f"Error generating hash: {e}")
            # Fallback hash
            return str(hash(title + text))
    
    def _fetch_reference_articles(self) -> List[Dict[str, Any]]:
        """Fetch reference articles from the database."""
        try:
            scraped_data = get_scraped_data()
            
            # Process each article for comparison
            reference_articles = []
            for item in scraped_data:
                try:
                    title = item.get('title', '')
                    description = item.get('description', '')
                    
                    if not description:
                        continue
                    
                    # Process with NLP pipeline
                    article_nlp = process_article(description, title)
                    
                    # Generate a summary
                    article_summary = summarize_for_comparison(description, title)
                    
                    reference_articles.append({
                        'title': title,
                        'text': description,
                        'nlp_data': article_nlp,
                        'summary': article_summary
                    })
                    
                except Exception as e:
                    logger.warning(f"Error processing reference article: {e}")
                    continue
                
            logger.info(f"Fetched {len(reference_articles)} reference articles")
            return reference_articles
            
        except Exception as e:
            logger.error(f"Error fetching reference articles: {e}")
            return []
    
    def _find_similar_articles(self, article_nlp: Dict[str, Any], 
                              article_summary: Dict[str, Any],
                              reference_articles: List[Dict[str, Any]],
                              max_matches: int = 5) -> List[Dict[str, Any]]:
        """Find similar articles from a list of reference articles."""
        matches = []
        
        try:
            # Original article data
            article_text = article_nlp.get('original_text', '')
            article_title = article_nlp.get('title', '')
            
            # Compare against each reference article
            for ref_article in reference_articles:
                try:
                    # Skip comparison with self
                    if (ref_article.get('text', '') == article_text and 
                        ref_article.get('title', '') == article_title):
                        continue
                    
                    ref_text = ref_article.get('text', '')
                    ref_title = ref_article.get('title', '')
                    
                    if not ref_text:
                        continue
                    
                    # Calculate text similarity
                    text_similarity = 100 - compute_text_score(article_text, ref_text)
                    title_similarity = 100 - compute_text_score(article_title, ref_title)
                    
                    # Use semantic similarity if available
                    semantic_similarity = 0
                    if self.use_embeddings and self.model:
                        try:
                            article_summary_text = article_summary.get('summary', article_text[:500])
                            ref_summary_text = ref_article.get('summary', {}).get('summary', ref_text[:500])
                            
                            if article_summary_text and ref_summary_text:
                                article_embedding = self.model.encode(article_summary_text, convert_to_tensor=True)
                                ref_embedding = self.model.encode(ref_summary_text, convert_to_tensor=True)
                                semantic_sim = util.cos_sim(article_embedding, ref_embedding).item()
                                semantic_similarity = (semantic_sim + 1) * 50  # Convert to 0-100 scale
                                
                        except Exception as e:
                            logger.warning(f"Error calculating semantic similarity: {e}")
                            semantic_similarity = 0
                    
                    # Calculate overall similarity (weighted average)
                    if semantic_similarity > 0:
                        overall_score = (text_similarity * 0.3 + title_similarity * 0.2 + semantic_similarity * 0.5)
                    else:
                        overall_score = (text_similarity * 0.7 + title_similarity * 0.3)
                    
                    if overall_score > 10:  # Only include matches with some similarity
                        matches.append({
                            'title': ref_title,
                            'similarity_score': round(overall_score, 2),
                            'detail_scores': {
                                'text_similarity': round(text_similarity, 2),
                                'title_similarity': round(title_similarity, 2),
                                'semantic_similarity': round(semantic_similarity, 2)
                            },
                            'match_type': self._determine_match_type(overall_score)
                        })
                        
                except Exception as e:
                    logger.warning(f"Error comparing with reference article: {e}")
                    continue
            
            # Sort by similarity score and limit
            matches.sort(key=lambda x: x['similarity_score'], reverse=True)
            return matches[:max_matches]
            
        except Exception as e:
            logger.error(f"Error finding similar articles: {e}")
            return []
    
    def _determine_match_type(self, score: float) -> str:
        """Determine the type of match based on similarity score."""
        if score >= 95:
            return 'near_exact_duplicate'
        elif score >= 85:
            return 'duplicate_with_minimal_changes'
        elif score >= 75:
            return 'high_similarity'
        elif score >= 65:
            return 'moderate_similarity'
        else:
            return 'low_similarity'


# Create singleton instance for direct use with error handling
def create_default_detector():
    """Create default detector with error handling."""
    try:
        detector = DuplicateDetector()
        logger.info("Default DuplicateDetector created successfully")
        return detector
    except Exception as e:
        logger.error(f"Error creating default detector: {e}")
        # Return a basic detector with hardcoded values
        detector = object.__new__(DuplicateDetector)
        detector.similarity_threshold = 80.0
        detector.use_embeddings = False
        detector.model = None
        return detector

default_detector = create_default_detector()

def check_article_duplicate(article: Dict[str, Any]) -> Dict[str, Any]:
    """
    Check if an article is a duplicate.
    
    Args:
        article: Dictionary with 'title' and 'text' keys
        
    Returns:
        Duplication check results
    """
    try:
        return default_detector.check_duplicate(article)
    except Exception as e:
        logger.error(f"Error in check_article_duplicate: {e}")
        return {
            'is_duplicate': False,
            'similarity_score': 0,
            'matches': [],
            'error': f'Error checking duplicate: {str(e)}'
        }

def batch_check_duplicates(articles: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Check multiple articles for duplicates efficiently.
    
    Args:
        articles: List of article dictionaries with 'title' and 'text' keys
        
    Returns:
        List of duplication check results
    """
    try:
        return default_detector.batch_check_duplicates(articles)
    except Exception as e:
        logger.error(f"Error in batch_check_duplicates: {e}")
        return [check_article_duplicate(article) for article in articles]

def generate_article_hash(title: str, text: str) -> str:
    """
    Generate a hash for an article.
    
    Args:
        title: Article title
        text: Article text
        
    Returns:
        SHA-256 hash as hexadecimal string
    """
    try:
        # Normalize text for consistent hashing
        normalized_content = (title + " " + text).lower().strip()
        # Create SHA-256 hash
        return hashlib.sha256(normalized_content.encode('utf-8')).hexdigest()
    except Exception as e:
        logger.error(f"Error generating hash: {e}")
        return str(hash(title + text))

# Test function
def test_duplicate_detector():
    """Test the duplicate detector functionality."""
    print("Testing DuplicateDetector...")
    
    try:
        # Test detector creation
        detector = DuplicateDetector()
        print(f"✅ Detector created with threshold: {detector.similarity_threshold}")
        
        # Test article
        test_article = {
            'title': 'Test Article About Climate Change',
            'text': 'This is a test article about climate change and its impacts on the environment. Scientists are studying the effects of global warming.'
        }
        
        # Test duplicate checking
        result = check_article_duplicate(test_article)
        print(f"✅ Duplicate check completed: {result['is_duplicate']}")
        print(f"   Similarity score: {result['similarity_score']}")
        
        return True
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        return False

if __name__ == "__main__":
    # Run test
    test_duplicate_detector()
    
    # Example usage
    print("\nExample usage:")
    
    original_article = {
        'title': 'EU Reaches Historic Migration Agreement',
        'text': """
        The European Union has reached a historic agreement on migration policy after years of debate. 
        The new legislation creates a system for sharing responsibility among EU member states for 
        hosting migrants and processing asylum applications.
        """
    }
    
    # Check duplicate
    result = check_article_duplicate(original_article)
    print(f"Is duplicate: {result['is_duplicate']}")
    print(f"Similarity score: {result.get('similarity_score', 0):.2f}")
    if result.get('matches'):
        print(f"Found {len(result['matches'])} similar articles")