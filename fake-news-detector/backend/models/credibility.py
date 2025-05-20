"""
Credibility Scoring Module for Fake News Detection
-------------------------------------------------
This module evaluates the credibility of news articles based on multiple factors:
1. Source reputation
2. Author reputation
3. Content analysis (sentiment, bias, complexity)
4. Citation and reference analysis
5. Publication metadata

The scoring algorithm produces a 0-100 credibility score with detailed explanation.
"""

import re
import logging
from typing import Dict, List, Tuple, Any, Optional
import numpy as np
from datetime import datetime
import json
import os
from pathlib import Path

# Setup logging
logger = logging.getLogger(__name__)
handler = logging.StreamHandler()
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
handler.setFormatter(formatter)
logger.addHandler(handler)
logger.setLevel(logging.INFO)

# Load source reputation data if available
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
SOURCE_REPUTATION_PATH = os.path.join(SCRIPT_DIR, '..', 'data', 'source_reputation.json')

source_reputation_data = {}
try:
    if os.path.exists(SOURCE_REPUTATION_PATH):
        with open(SOURCE_REPUTATION_PATH, 'r') as f:
            source_reputation_data = json.load(f)
    else:
        logger.warning(f"Source reputation data not found at {SOURCE_REPUTATION_PATH}")
        # Create minimal reputation data
        source_reputation_data = {
            "default": {
                "credibility": 50,
                "bias": "unknown",
                "factual_reporting": "mixed"
            },
            # Add some well-known sources
            "apnews.com": {
                "credibility": 90,
                "bias": "neutral",
                "factual_reporting": "very high"
            },
            "reuters.com": {
                "credibility": 92,
                "bias": "neutral",
                "factual_reporting": "very high"
            },
            "bbc.com": {
                "credibility": 88,
                "bias": "slight left",
                "factual_reporting": "high"
            },
            "theonion.com": {
                "credibility": 5,
                "bias": "satire",
                "factual_reporting": "very low"
            }
        }
except Exception as e:
    logger.error(f"Error loading source reputation data: {e}")
    # Create fallback reputation data
    source_reputation_data = {"default": {"credibility": 50, "bias": "unknown", "factual_reporting": "mixed"}}

class CredibilityScorer:
    """Evaluates and scores the credibility of news articles."""
    
    def __init__(self, source_data: Dict[str, Any] = None):
        """
        Initialize the credibility scorer.
        
        Args:
            source_data: Dictionary of source reputation data
        """
        self.source_data = source_data or source_reputation_data
        
        # Define category weights for scoring
        self.weights = {
            "source_reputation": 0.25,
            "content_quality": 0.30,
            "verifiability": 0.25,
            "presentation": 0.20
        }
    
    def score_article(self, article_analysis: Dict[str, Any]) -> Dict[str, Any]:
        """
        Calculate a credibility score for an article.
        
        Args:
            article_analysis: Dictionary containing NLP analysis results
            
        Returns:
            Dictionary with credibility score and explanations
        """
        # Extract needed information from analysis
        source_url = self._extract_source_url(article_analysis)
        original_text = article_analysis.get('original_text', '')
        title = article_analysis.get('title', '')
        metadata = article_analysis.get('metadata', {})
        named_entities = article_analysis.get('named_entities', {})
        sentiment = article_analysis.get('sentiment', {})
        bias_markers = article_analysis.get('bias_markers', {})
        
        # Calculate individual category scores
        source_score, source_details = self._evaluate_source(source_url, metadata)
        content_score, content_details = self._evaluate_content(
            original_text, title, named_entities, sentiment, bias_markers
        )
        verifiability_score, verifiability_details = self._evaluate_verifiability(
            original_text, named_entities
        )
        presentation_score, presentation_details = self._evaluate_presentation(
            original_text, title, bias_markers
        )
        
        # Calculate overall score (weighted average)
        category_scores = {
            "source_reputation": source_score,
            "content_quality": content_score,
            "verifiability": verifiability_score,
            "presentation": presentation_score
        }
        
        overall_score = sum(
            score * self.weights[category] for category, score in category_scores.items()
        )
        
        # Round to nearest integer
        overall_score = round(overall_score)
        
        # Generate explanation
        explanation = self._generate_explanation(
            overall_score, 
            category_scores,
            source_details,
            content_details,
            verifiability_details,
            presentation_details
        )
        
        # Get credibility label
        credibility_label = self._get_credibility_label(overall_score)
        
        # Generate fact-check links
        fact_check_links = self._generate_fact_check_links(article_analysis)
        
        return {
            "credibility_score": overall_score,
            "credibility_label": credibility_label,
            "category_scores": category_scores,
            "explanation": explanation,
            "details": {
                "source": source_details,
                "content": content_details,
                "verifiability": verifiability_details,
                "presentation": presentation_details
            },
            "fact_check_links": fact_check_links
        }
    
    def _extract_source_url(self, article_analysis: Dict[str, Any]) -> str:
        """Extract the source URL from article metadata."""
        metadata = article_analysis.get('metadata', {})
        
        # Try various common metadata fields for URL
        url = metadata.get('url', '')
        if not url:
            url = metadata.get('source_url', '')
        if not url:
            url = metadata.get('link', '')
            
        return url
    
    def _evaluate_source(self, url: str, metadata: Dict[str, Any]) -> Tuple[float, Dict[str, Any]]:
        """
        Evaluate the credibility of the source.
        
        Returns:
            Tuple of (score 0-100, details dictionary)
        """
        # Extract domain from URL
        domain = self._extract_domain(url)
        author = metadata.get('author', '')
        
        # Get source reputation
        source_reputation = self._get_source_reputation(domain)
        
        # Base score on reputation or default
        base_score = source_reputation.get('credibility', 50)
        
        # Adjust for author credibility
        author_adjustment = 0
        if author:
            # In a real system, we would check an author database
            # For now, just give a small boost for having an author
            author_adjustment = 5
        
        # Adjust for HTTPS (minor security indicator)
        https_adjustment = 2 if url.startswith('https://') else 0
        
        # Final score calculation
        final_score = min(100, max(0, base_score + author_adjustment + https_adjustment))
        
        return final_score, {
            "domain": domain,
            "author": author,
            "source_reputation": source_reputation,
            "has_author": bool(author),
            "uses_https": url.startswith('https://')
        }
    
    def _evaluate_content(self, text: str, title: str, 
                        named_entities: Dict[str, List[str]],
                        sentiment: Dict[str, Any],
                        bias_markers: Dict[str, float]) -> Tuple[float, Dict[str, Any]]:
        """
        Evaluate the quality of the content.
        
        Returns:
            Tuple of (score 0-100, details dictionary)
        """
        # Calculate text complexity
        complexity_score, complexity_details = self._analyze_text_complexity(text)
        
        # Evaluate clickbait title
        clickbait_score = 0
        clickbait_patterns = [
            r"you won't believe",
            r"shocking",
            r"mind-blowing",
            r"what happens next",
            r"secret",
            r"\d+ ways",
            r"\d+ things",
            r"this is why"
        ]
        
        title_lower = title.lower()
        clickbait_matches = [p for p in clickbait_patterns if re.search(p, title_lower)]
        
        if clickbait_matches:
            # More matches = more clickbaity
            clickbait_penalty = min(30, len(clickbait_matches) * 10)
            clickbait_score = 100 - clickbait_penalty
        else:
            clickbait_score = 100
        
        # Analyze excessive bias markers
        bias_score = 100
        high_bias_threshold = 5.0  # Per 1000 words
        
        for category, score in bias_markers.items():
            if score > high_bias_threshold:
                bias_penalty = min(30, int(score))
                bias_score -= bias_penalty
        
        bias_score = max(0, bias_score)
        
        # Check for emotional content
        sentiment_penalty = 0
        sentiment_polarity = sentiment.get('polarity', 0)
        sentiment_confidence = sentiment.get('confidence', 0)
        
        # Extremely polarized sentiment with high confidence suggests emotional content
        if abs(sentiment_polarity) > 0.7 and sentiment_confidence > 0.8:
            sentiment_penalty = 15
        elif abs(sentiment_polarity) > 0.5 and sentiment_confidence > 0.7:
            sentiment_penalty = 10
            
        sentiment_score = 100 - sentiment_penalty
        
        # Entity usage score - good articles reference specific entities
        entity_score = 0
        total_entities = sum(len(entities) for entities in named_entities.values())
        
        if total_entities > 10:  # Good number of specific references
            entity_score = 100
        elif total_entities > 5:
            entity_score = 80
        elif total_entities > 2:
            entity_score = 60
        else:
            entity_score = 40
        
        # Calculate weighted content score
        subcategory_weights = {
            "complexity": 0.25,
            "clickbait": 0.20,
            "bias": 0.25,
            "sentiment": 0.15,
            "entities": 0.15
        }
        
        subcategory_scores = {
            "complexity": complexity_score,
            "clickbait": clickbait_score,
            "bias": bias_score,
            "sentiment": sentiment_score,
            "entities": entity_score
        }
        
        final_score = sum(
            score * subcategory_weights[category] 
            for category, score in subcategory_scores.items()
        )
        
        return final_score, {
            "complexity": complexity_details,
            "clickbait_score": clickbait_score,
            "clickbait_matches": clickbait_matches,
            "bias_score": bias_score,
            "high_bias_categories": [
                category for category, score in bias_markers.items() 
                if score > high_bias_threshold
            ],
            "sentiment_score": sentiment_score,
            "sentiment_polarity": sentiment_polarity,
            "entity_score": entity_score,
            "total_entities": total_entities,
            "subcategory_scores": subcategory_scores
        }
    
    def _evaluate_verifiability(self, text: str, 
                              named_entities: Dict[str, List[str]]) -> Tuple[float, Dict[str, Any]]:
        """
        Evaluate how verifiable the article claims are.
        
        Returns:
            Tuple of (score 0-100, details dictionary)
        """
        # Check for quotes
        quotes_pattern = r'"([^"]*)"'
        quotes = re.findall(quotes_pattern, text)
        
        quotes_score = 0
        if len(quotes) >= 3:
            quotes_score = 100
        elif len(quotes) >= 2:
            quotes_score = 80
        elif len(quotes) >= 1:
            quotes_score = 60
        else:
            quotes_score = 40
            
        # Check for links/references
        links_pattern = r'https?://[^\s)"]+'
        links = re.findall(links_pattern, text)
        
        links_score = 0
        if len(links) >= 3:
            links_score = 100
        elif len(links) >= 2:
            links_score = 80
        elif len(links) >= 1:
            links_score = 60
        else:
            links_score = 40
        
        # Check for specific claims with attributions
        attribution_patterns = [
            r'according to',
            r'said by',
            r'stated by',
            r'reported by',
            r'confirms',
            r'verified by'
        ]
        
        attribution_matches = []
        for pattern in attribution_patterns:
            matches = re.findall(pattern, text.lower())
            attribution_matches.extend(matches)
        
        attribution_score = 0
        if len(attribution_matches) >= 3:
            attribution_score = 100
        elif len(attribution_matches) >= 2:
            attribution_score = 80
        elif len(attribution_matches) >= 1:
            attribution_score = 60
        else:
            attribution_score = 40
        
        # Calculate overall verifiability score
        subcategory_weights = {
            "quotes": 0.3,
            "links": 0.4,
            "attributions": 0.3
        }
        
        subcategory_scores = {
            "quotes": quotes_score,
            "links": links_score,
            "attributions": attribution_score
        }
        
        final_score = sum(
            score * subcategory_weights[category] 
            for category, score in subcategory_scores.items()
        )
        
        return final_score, {
            "quotes_found": len(quotes),
            "links_found": len(links),
            "attributions_found": len(attribution_matches),
            "subcategory_scores": subcategory_scores
        }
    
    def _evaluate_presentation(self, text: str, title: str, 
                             bias_markers: Dict[str, float]) -> Tuple[float, Dict[str, Any]]:
        """
        Evaluate the presentation quality of the article.
        
        Returns:
            Tuple of (score 0-100, details dictionary)
        """
        # Check for ALL CAPS in title (sensationalism)
        caps_ratio = sum(1 for c in title if c.isupper()) / max(1, len(title))
        
        caps_score = 0
        if caps_ratio > 0.5:  # More than half capitalized
            caps_score = 30
        elif caps_ratio > 0.3:
            caps_score = 50
        elif caps_ratio > 0.2:
            caps_score = 70
        else:
            caps_score = 100
            
        # Check for excessive exclamation points
        exclamation_count = text.count('!')
        text_length = len(text.split())
        
        normalized_exclamation = (exclamation_count / max(1, text_length)) * 1000  # Per 1000 words
        
        exclamation_score = 0
        if normalized_exclamation > 10:
            exclamation_score = 30
        elif normalized_exclamation > 5:
            exclamation_score = 50
        elif normalized_exclamation > 2:
            exclamation_score = 70
        else:
            exclamation_score = 100
            
        # Check for extreme emotional language
        emotional_score = 100 - min(100, int(bias_markers.get('emotional', 0) * 10))
        
        # Check for proper paragraphing (text structure)
        paragraphs = text.split('\n\n')
        avg_paragraph_length = np.mean([len(p.split()) for p in paragraphs]) if paragraphs else 0
        
        structure_score = 0
        if 20 <= avg_paragraph_length <= 100:  # Good paragraph size
            structure_score = 100
        elif 10 <= avg_paragraph_length <= 150:
            structure_score = 80
        elif 5 <= avg_paragraph_length <= 200:
            structure_score = 60
        else:
            structure_score = 40
            
        # Calculate overall presentation score
        subcategory_weights = {
            "capitalization": 0.20,
            "exclamations": 0.20,
            "emotional_language": 0.35,
            "structure": 0.25
        }
        
        subcategory_scores = {
            "capitalization": caps_score,
            "exclamations": exclamation_score,
            "emotional_language": emotional_score,
            "structure": structure_score
        }
        
        final_score = sum(
            score * subcategory_weights[category] 
            for category, score in subcategory_scores.items()
        )
        
        return final_score, {
            "caps_ratio": caps_ratio,
            "exclamation_density": normalized_exclamation,
            "avg_paragraph_length": avg_paragraph_length,
            "subcategory_scores": subcategory_scores
        }
    
    def _analyze_text_complexity(self, text: str) -> Tuple[float, Dict[str, Any]]:
        """
        Analyze the complexity of the text.
        
        Returns:
            Tuple of (score 0-100, details dictionary)
        """
        # Simple implementation of Flesch-Kincaid readability
        words = text.split()
        word_count = len(words)
        
        if word_count == 0:
            return 50, {"readability": "unknown", "avg_word_length": 0, "word_count": 0}
        
        # Count sentences
        sentences = re.split(r'[.!?]+', text)
        sentence_count = len([s for s in sentences if s.strip()])
        
        if sentence_count == 0:
            sentence_count = 1  # Avoid division by zero
        
        # Calculate average word length
        avg_word_length = np.mean([len(word) for word in words])
        
        # Calculate avg words per sentence
        avg_words_per_sentence = word_count / sentence_count
        
        # Simple readability score (higher is more complex)
        # Based on averages observed in news articles
        readability_index = (avg_word_length * 0.39) + (avg_words_per_sentence * 0.05) - 1.0
        
        # Map to a credibility score (news articles should be moderately complex)
        # Too simple = possibly oversimplified/sensationalist
        # Too complex = possibly academic/specialized
        # Best credibility = moderate complexity
        if 0.5 <= readability_index <= 1.2:  # Good range for news
            complexity_score = 100
        elif 0.3 <= readability_index <= 1.5:
            complexity_score = 80
        elif 0.1 <= readability_index <= 2.0:
            complexity_score = 60
        else:
            complexity_score = 40
            
        return complexity_score, {
            "readability_index": readability_index,
            "avg_word_length": avg_word_length,
            "avg_words_per_sentence": avg_words_per_sentence,
            "word_count": word_count,
            "sentence_count": sentence_count
        }
    
    def _extract_domain(self, url: str) -> str:
        """Extract domain name from URL."""
        if not url:
            return ""
            
        try:
            # Simple regex to extract domain
            domain_match = re.search(r'https?://(?:www\.)?([^/]+)', url)
            if domain_match:
                return domain_match.group(1)
            return ""
        except Exception as e:
            logger.error(f"Error extracting domain from URL {url}: {e}")
            return ""
    
    def _get_source_reputation(self, domain: str) -> Dict[str, Any]:
        """Get reputation data for a source domain."""
        if not domain:
            return self.source_data.get("default", {"credibility": 50})
            
        # Try exact match
        if domain in self.source_data:
            return self.source_data[domain]
            
        # Try without subdomain
        base_domain = '.'.join(domain.split('.')[-2:])
        if base_domain in self.source_data:
            return self.source_data[base_domain]
            
        # Return default
        return self.source_data.get("default", {"credibility": 50})
    
    def _generate_explanation(self, overall_score: float, 
                            category_scores: Dict[str, float],
                            source_details: Dict[str, Any],
                            content_details: Dict[str, Any],
                            verifiability_details: Dict[str, Any],
                            presentation_details: Dict[str, Any]) -> str:
        """Generate a human-readable explanation of the credibility score."""
        # Determine most impactful positive and negative factors
        category_impacts = {}
        for category, score in category_scores.items():
            # Calculate impact as distance from neutral (50)
            impact = (score - 50) * self.weights[category]
            category_impacts[category] = impact
        
        # Get highest positive and negative impacts
        positive_categories = sorted(
            [(cat, impact) for cat, impact in category_impacts.items() if impact > 0],
            key=lambda x: x[1],
            reverse=True
        )
        
        negative_categories = sorted(
            [(cat, abs(impact)) for cat, impact in category_impacts.items() if impact < 0],
            key=lambda x: x[1],
            reverse=True
        )
        
        # Start with overall assessment
        explanation = f"This article has a credibility score of {overall_score}/100. "
        
        # Add positive factors
        if positive_categories:
            explanation += "Positive factors include "
            positive_points = []
            
            for category, _ in positive_categories[:2]:  # Top 2 positive factors
                if category == "source_reputation":
                    source_name = source_details.get("domain", "the source")
                    reputation = source_details.get("source_reputation", {})
                    credibility = reputation.get("credibility", "unknown")
                    
                    if credibility > 75:
                        positive_points.append(f"coming from {source_name}, a highly credible source")
                    else:
                        positive_points.append(f"reasonable source credibility")
                        
                elif category == "content_quality":
                    subcategory_scores = content_details.get("subcategory_scores", {})
                    best_subcategory = max(subcategory_scores.items(), key=lambda x: x[1])
                    
                    if best_subcategory[0] == "complexity" and best_subcategory[1] > 70:
                        positive_points.append("appropriate writing complexity for news content")
                    elif best_subcategory[0] == "clickbait" and best_subcategory[1] > 70:
                        positive_points.append("using a factual, non-clickbait headline")
                    elif best_subcategory[0] == "bias" and best_subcategory[1] > 70:
                        positive_points.append("showing limited bias markers in the text")
                    elif best_subcategory[0] == "entities" and best_subcategory[1] > 70:
                        positive_points.append(f"referencing specific entities ({content_details.get('total_entities', 0)} found)")
                        
                elif category == "verifiability":
                    quotes = verifiability_details.get("quotes_found", 0)
                    links = verifiability_details.get("links_found", 0)
                    attributions = verifiability_details.get("attributions_found", 0)
                    
                    if quotes > 2:
                        positive_points.append(f"including {quotes} direct quotes")
                    elif links > 2:
                        positive_points.append(f"containing {links} links to sources")
                    elif attributions > 2:
                        positive_points.append("properly attributing claims to sources")
                        
                elif category == "presentation":
                    subcategory_scores = presentation_details.get("subcategory_scores", {})
                    best_subcategory = max(subcategory_scores.items(), key=lambda x: x[1])
                    
                    if best_subcategory[0] == "structure" and best_subcategory[1] > 70:
                        positive_points.append("well-structured paragraphs and organization")
                    elif best_subcategory[0] == "capitalization" and best_subcategory[1] > 70:
                        positive_points.append("appropriate use of capitalization")
                    elif best_subcategory[0] == "exclamations" and best_subcategory[1] > 70:
                        positive_points.append("limited use of exclamation points")
            
            if positive_points:
                explanation += ", ".join(positive_points) + ". "
        
        # Add negative factors
        if negative_categories:
            explanation += "Areas of concern include "
            negative_points = []
            
            for category, _ in negative_categories[:2]:  # Top 2 negative factors
                if category == "source_reputation":
                    source_name = source_details.get("domain", "the source")
                    reputation = source_details.get("source_reputation", {})
                    credibility = reputation.get("credibility", "unknown")
                    
                    if credibility < 30:
                        negative_points.append(f"coming from {source_name}, a source with low credibility")
                    else:
                        negative_points.append("uncertain source reputation")
                        
                elif category == "content_quality":
                    subcategory_scores = content_details.get("subcategory_scores", {})
                    worst_subcategory = min(subcategory_scores.items(), key=lambda x: x[1])
                    
                    if worst_subcategory[0] == "clickbait" and worst_subcategory[1] < 60:
                        matches = content_details.get("clickbait_matches", [])
                        negative_points.append(f"using clickbait phrases in the headline")
                    elif worst_subcategory[0] == "bias" and worst_subcategory[1] < 60:
                        high_bias = content_details.get("high_bias_categories", [])
                        bias_type = high_bias[0] if high_bias else "general"
                        negative_points.append(f"showing {bias_type} bias markers")
                    elif worst_subcategory[0] == "entities" and worst_subcategory[1] < 60:
                        negative_points.append("lacking specific named entities")
                        
                elif category == "verifiability":
                    quotes = verifiability_details.get("quotes_found", 0)
                    links = verifiability_details.get("links_found", 0)
                    attributions = verifiability_details.get("attributions_found", 0)
                    
                    if quotes == 0 and links == 0:
                        negative_points.append("missing sources and attributions")
                    elif quotes == 0:
                        negative_points.append("lacking direct quotes")
                    elif links == 0:
                        negative_points.append("lacking links to sources")
                        
                elif category == "presentation":
                    subcategory_scores = presentation_details.get("subcategory_scores", {})
                    worst_subcategory = min(subcategory_scores.items(), key=lambda x: x[1])
                    
                    if worst_subcategory[0] == "emotional_language" and worst_subcategory[1] < 60:
                        negative_points.append("using highly emotional language")
                    elif worst_subcategory[0] == "exclamations" and worst_subcategory[1] < 60:
                        negative_points.append("overusing exclamation points")
                    elif worst_subcategory[0] == "capitalization" and worst_subcategory[1] < 60:
                        negative_points.append("excessive use of capital letters")
            
            if negative_points:
                explanation += ", ".join(negative_points) + ". "
        
        # Add general advice
        explanation += f"The article ranks as {self._get_credibility_label(overall_score)}."
        
        return explanation
    
    def _get_credibility_label(self, score: float) -> str:
        """Get a human-readable label for the credibility score."""
        if score >= 90:
            return "highly credible"
        elif score >= 75:
            return "credible"
        elif score >= 60:
            return "somewhat credible"
        elif score >= 40:
            return "questionable"
        elif score >= 25:
            return "low credibility"
        else:
            return "very low credibility"
    
    def _generate_fact_check_links(self, article_analysis: Dict[str, Any]) -> List[str]:
        """Generate relevant fact-check links based on article content."""
        # This would normally query fact-checking APIs
        # For this implementation, we'll generate some example links
        
        named_entities = article_analysis.get('named_entities', {})
        title = article_analysis.get('title', '')
        
        # Extract key persons, organizations, and places
        people = named_entities.get('PERSON', [])[:2]
        orgs = named_entities.get('ORG', [])[:2]
        places = named_entities.get('GPE', []) + named_entities.get('LOC', [])
        places = places[:2]
        
        # Create search terms
        search_terms = []
        if title:
            search_terms.append(title.replace(' ', '+'))
        
        for person in people:
            search_terms.append(person.replace(' ', '+'))
            
        for org in orgs:
            search_terms.append(org.replace(' ', '+'))
            
        # Generate example links to fact-checking sites
        fact_check_links = []
        
        for term in search_terms[:2]:  # Use top 2 terms
            fact_check_links.append(f"https://www.snopes.com/?s={term}")
            fact_check_links.append(f"https://www.politifact.com/search/?q={term}")
            fact_check_links.append(f"https://factcheck.org/search/{term}")
            
        # Remove duplicates and limit to 5 links
        unique_links = list(set(fact_check_links))
        return unique_links[:5]

# Create singleton instance for direct use
default_scorer = CredibilityScorer()

def score_credibility(article_analysis: Dict[str, Any]) -> Dict[str, Any]:
    """
    Calculate a credibility score for an article.
    
    Args:
        article_analysis: Dictionary containing NLP analysis results
        
    Returns:
        Dictionary with credibility score and explanations
    """
    return default_scorer.score_article(article_analysis)

def get_source_reputation(domain: str) -> Dict[str, Any]:
    """
    Get reputation data for a source domain.
    
    Args:
        domain: Website domain
        
    Returns:
        Reputation data dictionary
    """
    return default_scorer._get_source_reputation(domain)

if __name__ == "__main__":
    # Example usage
    sample_analysis = {
        'original_text': """
            The United States announced new sanctions against Russia today, 
            targeting key industries including oil and banking. President Biden 
            stated that "these measures are designed to hold Russia accountable
            for its actions in Ukraine." Russian officials immediately condemned
            the move as "hostile and counterproductive." According to reports from
            the State Department, these sanctions could reduce Russian GDP by 3%
            over the next year. The European Union is expected to follow with 
            similar measures next week, according to diplomatic sources.
        """,
        'title': 'US Imposes New Sanctions on Russia Over Ukraine Conflict',
        'metadata': {
            'url': 'https://example-news.com/russia-sanctions',
            'author': 'Jane Smith',
            'published_date': '2023-08-15'
        },
        'named_entities': {
            'PERSON': ['Biden'],
            'ORG': ['United States', 'Russia', 'State Department', 'European Union'],
            'GPE': ['Ukraine'],
            'LOC': [],
            'DATE': ['today', 'next week'],
            'MISC': []
        },
        'sentiment': {
            'polarity': -0.2,
            'label': 'negative',
            'confidence': 0.65
        },
        'bias_markers': {
            'political_left': 1.2,
            'political_right': 0.8,
            'emotional': 2.1,
            'clickbait': 0.5,
            'exaggeration': 1.3,
            'uncertainty': 3.2
        }
    }
    
    result = score_credibility(sample_analysis)
    print(f"Credibility Score: {result['credibility_score']}/100")
    print(f"Label: {result['credibility_label']}")
    print(f"Explanation: {result['explanation']}")
    print(f"Fact-check links: {result['fact_check_links']}")
