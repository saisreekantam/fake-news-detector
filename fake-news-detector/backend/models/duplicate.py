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
from typing import Dict, List, Tuple, Any, Optional, Set
import heapq
import numpy as np
from datetime import datetime, timedelta

# Import internal modules
from .summarizer import summarize_for_comparison
from .nlp_pipeline import process_article
from ..utils.similarity import compute_text_score, image_similarity_score
from ..database.repository import article_hash_exists, store_article_hash, get_scraped_data

# Import sentence transformers for semantic similarity
try:
    from sentence_transformers import SentenceTransformer, util
    SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
    SENTENCE_TRANSFORMERS_AVAILABLE = False
    logging.warning("Sentence-Transformers not available. Using fallback similarity methods.")

# Initialize logging
logger = logging.getLogger(__name__)
handler = logging.StreamHandler()
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
handler.setFormatter(formatter)
logger.addHandler(handler)
logger.setLevel(logging.INFO)

# Initialize sentence transformer model if available
model = None
if SENTENCE_TRANSFORMERS_AVAILABLE:
    try:
        model = SentenceTransformer('all-MiniLM-L6-v2')
        logger.info("Sentence-Transformer model loaded successfully")
    except Exception as e:
        logger.error(f"Error loading Sentence-Transformer model: {e}")
        SENTENCE_TRANSFORMERS_AVAILABLE = False

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
        self.similarity_threshold = similarity_threshold
        self.use_embeddings = use_embeddings and SENTENCE_TRANSFORMERS_AVAILABLE
        self.model = model if self.use_embeddings else None
    
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
        if article_hash_exists(content_hash):
            return {
                'is_duplicate': True,
                'similarity_score': 100,  # Exact match
                'matches': [{'hash': content_hash, 'score': 100, 'match_type': 'exact_hash'}],
                'message': 'Exact duplicate found in database'
            }
        
        # If no reference articles provided, fetch from database
        if reference_articles is None:
            reference_articles = self._fetch_reference_articles()
            
        if not reference_articles:
            # No reference articles to compare against
            # Store the new article hash
            store_article_hash(content_hash)
            return {
                'is_duplicate': False,
                'similarity_score': 0,
                'matches': [],
                'message': 'No reference articles available for comparison'
            }
            
        # Process the article with NLP pipeline for entity extraction and key phrases
        article_nlp = process_article(text, title)
        
        # Generate a summary for comparison
        article_summary = summarize_for_comparison(text, title)
        
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
            store_article_hash(content_hash)
            
        return {
            'is_duplicate': is_duplicate,
            'content_hash': content_hash,
            'similarity_score': similarity_score,
            'matches': matches,
            'message': f"{'Duplicate' if is_duplicate else 'Original'} content detected"
        }
    
    def batch_check_duplicates(self, articles: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Check multiple articles for duplicates efficiently.
        
        Args:
            articles: List of article dictionaries
            
        Returns:
            List of duplicate check results
        """
        # Fetch reference articles once for all checks
        reference_articles = self._fetch_reference_articles()
        
        results = []
        for article in articles:
            result = self.check_duplicate(article, reference_articles)
            results.append(result)
            
            # If not a duplicate, add to reference articles for subsequent checks
            if not result['is_duplicate']:
                # Process article for comparison with subsequent articles
                title = article.get('title', '')
                text = article.get('text', '')
                article_nlp = process_article(text, title)
                article_summary = summarize_for_comparison(text, title)
                
                reference_articles.append({
                    'title': title,
                    'text': text,
                    'nlp_data': article_nlp,
                    'summary': article_summary
                })
        
        return results
    
    def _generate_hash(self, title: str, text: str) -> str:
        """Generate a content hash for the article."""
        # Normalize text for consistent hashing
        normalized_content = (title + " " + text).lower().strip()
        # Create SHA-256 hash
        return hashlib.sha256(normalized_content.encode('utf-8')).hexdigest()
    
    def _fetch_reference_articles(self) -> List[Dict[str, Any]]:
        """Fetch reference articles from the database."""
        # This would typically query the database for recent articles
        # For now, we'll use the get_scraped_data function
        try:
            scraped_data = get_scraped_data()
            
            # Process each article for comparison
            reference_articles = []
            for item in scraped_data:
                title = item.get('title', '')
                description = item.get('description', '')
                
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
                
            return reference_articles
        except Exception as e:
            logger.error(f"Error fetching reference articles: {e}")
            return []
    
    def _find_similar_articles(self, article_nlp: Dict[str, Any], 
                              article_summary: Dict[str, Any],
                              reference_articles: List[Dict[str, Any]],
                              max_matches: int = 5) -> List[Dict[str, Any]]:
        """
        Find similar articles from a list of reference articles.
        
        Args:
            article_nlp: NLP processing results for the article
            article_summary: Summary of the article
            reference_articles: List of reference articles to compare against
            max_matches: Maximum number of matches to return
            
        Returns:
            List of match dictionaries, ordered by similarity score
        """
        matches = []
        
        # Original article data
        article_text = article_nlp.get('original_text', '')
        article_title = article_nlp.get('title', '')
        article_entities = article_nlp.get('named_entities', {})
        article_key_phrases = article_nlp.get('key_phrases', [])
        article_fingerprint = article_summary.get('fingerprint', {})
        article_summary_text = article_summary.get('summary', '')
        
        # Prepare embeddings for semantic similarity if available
        article_embedding = None
        if self.use_embeddings and article_summary_text:
            try:
                article_embedding = self.model.encode(article_summary_text, convert_to_tensor=True)
            except Exception as e:
                logger.error(f"Error encoding article for semantic similarity: {e}")
        
        # Compare against each reference article
        for ref_article in reference_articles:
            # Skip comparison with self (if in the reference set)
            if ref_article.get('text', '') == article_text and ref_article.get('title', '') == article_title:
                continue
                
            ref_nlp = ref_article.get('nlp_data', {})
            ref_summary = ref_article.get('summary', {})
            
            # Calculate multiple similarity metrics
            similarity_scores = self._calculate_similarity_metrics(
                article_text, article_title, article_entities, article_key_phrases, 
                article_fingerprint, article_summary_text, article_embedding,
                ref_article.get('text', ''), ref_article.get('title', ''),
                ref_nlp.get('named_entities', {}), ref_nlp.get('key_phrases', []),
                ref_summary.get('fingerprint', {}), ref_summary.get('summary', '')
            )
            
            # Calculate overall similarity score (weighted average)
            overall_score = self._calculate_overall_similarity(similarity_scores)
            
            if overall_score > 0:
                matches.append({
                    'title': ref_article.get('title', ''),
                    'similarity_score': overall_score,
                    'detail_scores': similarity_scores,
                    'match_type': self._determine_match_type(overall_score, similarity_scores)
                })
        
        # Sort by similarity score (descending) and limit to max_matches
        matches.sort(key=lambda x: x['similarity_score'], reverse=True)
        return matches[:max_matches]
    
    def _calculate_similarity_metrics(self, 
                                     article_text: str, article_title: str, 
                                     article_entities: Dict[str, List[str]],
                                     article_key_phrases: List[str],
                                     article_fingerprint: Dict[str, Any],
                                     article_summary: str, article_embedding: Any,
                                     ref_text: str, ref_title: str,
                                     ref_entities: Dict[str, List[str]],
                                     ref_key_phrases: List[str],
                                     ref_fingerprint: Dict[str, Any],
                                     ref_summary: str) -> Dict[str, float]:
        """
        Calculate various similarity metrics between two articles.
        
        Returns:
            Dictionary of similarity scores (0-100 scale, higher means more similar)
        """
        scores = {}
        
        # 1. Text similarity using compute_text_score from similarity.py
        # This returns a distance (0-100), we convert to similarity
        text_distance = compute_text_score(article_text, ref_text)
        scores['text_similarity'] = 100 - text_distance
        
        # 2. Title similarity
        title_distance = compute_text_score(article_title, ref_title)
        scores['title_similarity'] = 100 - title_distance
        
        # 3. Entity overlap - especially important for news articles
        entity_similarity = self._calculate_entity_similarity(article_entities, ref_entities)
        scores['entity_similarity'] = entity_similarity
        
        # 4. Key phrase overlap
        phrase_similarity = self._calculate_phrase_similarity(
            article_key_phrases, ref_key_phrases
        )
        scores['phrase_similarity'] = phrase_similarity
        
        # 5. Fingerprint term overlap
        if article_fingerprint and ref_fingerprint:
            term_similarity = self._calculate_term_similarity(
                article_fingerprint.get('top_terms', []),
                ref_fingerprint.get('top_terms', [])
            )
            scores['term_similarity'] = term_similarity
        
        # 6. Summary similarity using semantic embeddings (if available)
        if self.use_embeddings and article_embedding is not None and ref_summary:
            try:
                ref_embedding = self.model.encode(ref_summary, convert_to_tensor=True)
                semantic_similarity = util.cos_sim(article_embedding, ref_embedding).item()
                # Convert from [-1, 1] to [0, 100]
                scores['semantic_similarity'] = (semantic_similarity + 1) * 50
            except Exception as e:
                logger.error(f"Error in semantic similarity calculation: {e}")
                scores['semantic_similarity'] = 0
        else:
            # Fallback to text similarity on summaries
            summary_distance = compute_text_score(article_summary, ref_summary)
            scores['summary_similarity'] = 100 - summary_distance
        
        return scores
    
    def _calculate_overall_similarity(self, similarity_scores: Dict[str, float]) -> float:
        """
        Calculate weighted average of similarity scores.
        
        Args:
            similarity_scores: Dictionary of similarity scores
            
        Returns:
            Overall similarity score (0-100)
        """
        # Define weights for different metrics
        weights = {
            'text_similarity': 0.15,
            'title_similarity': 0.15,
            'entity_similarity': 0.20,
            'phrase_similarity': 0.10,
            'term_similarity': 0.10,
            'semantic_similarity': 0.30,  # Higher weight for semantic similarity
            'summary_similarity': 0.30    # Used when semantic_similarity isn't available
        }
        
        # Calculate weighted sum
        total_weight = 0
        weighted_sum = 0
        
        for metric, score in similarity_scores.items():
            if metric in weights:
                weighted_sum += score * weights[metric]
                total_weight += weights[metric]
        
        # Return weighted average or 0 if no valid scores
        if total_weight > 0:
            return weighted_sum / total_weight
        else:
            return 0
    
    def _determine_match_type(self, overall_score: float, 
                             similarity_scores: Dict[str, float]) -> str:
        """Determine the type of match based on similarity patterns."""
        if overall_score >= 95:
            return 'near_exact_duplicate'
        elif overall_score >= 85:
            return 'duplicate_with_minimal_changes'
        elif overall_score >= 75:
            # Check if entity overlap is high but text similarity lower
            if (similarity_scores.get('entity_similarity', 0) > 80 and 
                similarity_scores.get('text_similarity', 0) < 70):
                return 'rewrite_same_story'
            else:
                return 'high_similarity'
        elif overall_score >= 65:
            return 'moderate_similarity'
        else:
            return 'low_similarity'
    
    def _calculate_entity_similarity(self, 
                                    article_entities: Dict[str, List[str]],
                                    ref_entities: Dict[str, List[str]]) -> float:
        """
        Calculate similarity based on named entity overlap.
        
        Returns:
            Similarity score (0-100)
        """
        if not article_entities or not ref_entities:
            return 0
        
        # Weight entities by type importance (PERSON, ORG, GPE are most important for news)
        type_weights = {
            'PERSON': 0.4,
            'ORG': 0.3,
            'GPE': 0.2,  # Geo-political entities (countries, cities)
            'LOC': 0.1,  # Other locations
            'DATE': 0.05,
            'MISC': 0.05
        }
        
        total_similarity = 0
        total_weight = 0
        
        for entity_type, weight in type_weights.items():
            if entity_type in article_entities and entity_type in ref_entities:
                article_ents = set(e.lower() for e in article_entities[entity_type])
                ref_ents = set(e.lower() for e in ref_entities[entity_type])
                
                if article_ents and ref_ents:
                    # Jaccard similarity: intersection size / union size
                    intersection = len(article_ents.intersection(ref_ents))
                    union = len(article_ents.union(ref_ents))
                    
                    if union > 0:
                        type_similarity = (intersection / union) * 100
                        total_similarity += type_similarity * weight
                        total_weight += weight
        
        # Return weighted average or 0 if no valid comparisons
        if total_weight > 0:
            return total_similarity / total_weight
        else:
            return 0
    
    def _calculate_phrase_similarity(self, 
                                   article_phrases: List[str], 
                                   ref_phrases: List[str]) -> float:
        """
        Calculate similarity based on key phrase overlap.
        
        Returns:
            Similarity score (0-100)
        """
        if not article_phrases or not ref_phrases:
            return 0
        
        # Normalize phrases
        article_phrases_norm = set(phrase.lower() for phrase in article_phrases)
        ref_phrases_norm = set(phrase.lower() for phrase in ref_phrases)
        
        # Calculate Jaccard similarity
        intersection = len(article_phrases_norm.intersection(ref_phrases_norm))
        union = len(article_phrases_norm.union(ref_phrases_norm))
        
        if union > 0:
            return (intersection / union) * 100
        else:
            return 0
    
    def _calculate_term_similarity(self, 
                                 article_terms: List[str], 
                                 ref_terms: List[str]) -> float:
        """
        Calculate similarity based on top term overlap.
        
        Returns:
            Similarity score (0-100)
        """
        if not article_terms or not ref_terms:
            return 0
        
        # Normalize terms
        article_terms_norm = set(term.lower() for term in article_terms)
        ref_terms_norm = set(term.lower() for term in ref_terms)
        
        # Calculate Jaccard similarity
        intersection = len(article_terms_norm.intersection(ref_terms_norm))
        union = len(article_terms_norm.union(ref_terms_norm))
        
        if union > 0:
            return (intersection / union) * 100
        else:
            return 0

# Create singleton instance for direct use
default_detector = DuplicateDetector()

def check_article_duplicate(article: Dict[str, Any]) -> Dict[str, Any]:
    """
    Check if an article is a duplicate.
    
    Args:
        article: Dictionary with 'title' and 'text' keys
        
    Returns:
        Duplication check results
    """
    return default_detector.check_duplicate(article)

def batch_check_duplicates(articles: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Check multiple articles for duplicates efficiently.
    
    Args:
        articles: List of article dictionaries with 'title' and 'text' keys
        
    Returns:
        List of duplication check results
    """
    return default_detector.batch_check_duplicates(articles)

def generate_article_hash(title: str, text: str) -> str:
    """
    Generate a hash for an article.
    
    Args:
        title: Article title
        text: Article text
        
    Returns:
        SHA-256 hash as hexadecimal string
    """
    # Normalize text for consistent hashing
    normalized_content = (title + " " + text).lower().strip()
    # Create SHA-256 hash
    return hashlib.sha256(normalized_content.encode('utf-8')).hexdigest()

if __name__ == "__main__":
    # Example usage
    original_article = {
        'title': 'EU Reaches Historic Migration Agreement',
        'text': """
        The European Union has reached a historic agreement on migration policy after years of debate. 
        The new legislation creates a system for sharing responsibility among EU member states for 
        hosting migrants and processing asylum applications. Under the new rules, countries can either 
        accept migrants, pay a financial contribution, or provide operational support. European Commission 
        President Ursula von der Leyen called it a "landmark agreement that delivers for all member states."
        """
    }
    
    duplicate_article = {
        'title': 'European Union Agrees on New Migration Policy',
        'text': """
        After years of negotiations, the EU has finally reached a historic agreement on migration.
        The new system creates a mechanism for sharing responsibility among all European Union states
        for hosting migrants and processing asylum claims. Member states can choose to accept migrants,
        provide financial contributions, or offer operational support. Ursula von der Leyen, President of
        the European Commission, described it as a landmark agreement that works for all member countries.
        """
    }
    
    # Check duplicate
    result = check_article_duplicate(duplicate_article)
    print(f"Is duplicate: {result['is_duplicate']}")
    print(f"Similarity score: {result['similarity_score']:.2f}")
    print(f"Match type: {result['matches'][0]['match_type'] if result['matches'] else 'No matches'}")
