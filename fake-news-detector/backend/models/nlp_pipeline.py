"""
NLP Pipeline for Fake News Detection
------------------------------------
This module processes news articles through various NLP techniques:
1. Text cleaning and preprocessing
2. Named Entity Recognition
3. Sentiment Analysis
4. Bias Detection
5. Feature Extraction for further analysis

The pipeline serves as the core NLP processor for the fake news detection system.
"""

import re
import spacy
import numpy as np
from typing import Dict, List, Tuple, Any, Optional
from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification
from sklearn.feature_extraction.text import TfidfVectorizer

# Initialize models and pipelines
try:
    # Load spaCy model for NER and basic NLP tasks
    nlp = spacy.load("en_core_web_md")
except OSError:
    # Fallback to smaller model if the medium one is not available
    try:
        nlp = spacy.load("en_core_web_sm")
        print("Warning: Using smaller spaCy model. For better performance, install 'en_core_web_md'")
    except OSError:
        print("Error: spaCy model not found. Please install it with:")
        print("python -m spacy download en_core_web_sm")
        raise

# Sentiment analysis with transformers
sentiment_analyzer = pipeline(
    "sentiment-analysis", 
    model="distilbert-base-uncased-finetuned-sst-2-english",
    max_length=512,
    truncation=True
)

# Bias detection model - using a simple approach with loaded keywords
# In a production system, this would be replaced with a fine-tuned classifier
BIAS_INDICATORS = {
    "political_left": ["progressive", "liberal", "democrat", "left-wing", "socialism"],
    "political_right": ["conservative", "republican", "right-wing", "traditional"],
    "emotional": ["shocking", "outrageous", "unbelievable", "devastating", "terrifying", "alarming"],
    "clickbait": ["you won't believe", "shocking truth", "what happens next", "this is why", "secret"],
    "exaggeration": ["absolutely", "completely", "totally", "never", "always", "every", "all"],
    "uncertainty": ["reportedly", "allegedly", "sources say", "could be", "might be", "possibly"]
}

class NewsPipeline:
    """Main NLP pipeline for news article processing."""
    
    def __init__(self):
        self.vectorizer = TfidfVectorizer(max_features=5000, stop_words='english')
        
    def process_article(self, article_text: str, title: str = "", 
                        metadata: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Process a news article through the complete NLP pipeline.
        
        Args:
            article_text: The main content of the article
            title: The article title
            metadata: Additional metadata like author, publication date, etc.
            
        Returns:
            Dict containing all extracted features and analysis results
        """
        # Initialize result dictionary
        result = {
            "original_text": article_text,
            "title": title,
            "metadata": metadata or {},
            "word_count": len(article_text.split()),
            "preprocessed_text": ""
        }
        
        # Clean and preprocess text
        cleaned_text = self._preprocess_text(article_text)
        result["preprocessed_text"] = cleaned_text
        
        # Process with spaCy for NER and basic linguistic features
        doc = nlp(cleaned_text)
        
        # Extract named entities
        result["named_entities"] = self._extract_named_entities(doc)
        
        # Extract key noun phrases for topics
        result["key_phrases"] = self._extract_key_phrases(doc)
        
        # Analyze sentiment
        result["sentiment"] = self._analyze_sentiment(cleaned_text, title)
        
        # Detect bias markers
        result["bias_markers"] = self._detect_bias(cleaned_text, title)
        
        # Extract features for machine learning
        result["features"] = self._extract_features(doc, result)
        
        return result
    
    def batch_process(self, articles: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Process multiple articles in batch.
        
        Args:
            articles: List of dictionaries containing article text and metadata
            
        Returns:
            List of processed article results
        """
        results = []
        for article in articles:
            text = article.get("text", "")
            title = article.get("title", "")
            metadata = {k: v for k, v in article.items() if k not in ["text", "title"]}
            results.append(self.process_article(text, title, metadata))
        return results
    
    def _preprocess_text(self, text: str) -> str:
        """Clean and normalize text for analysis."""
        # Remove URLs
        text = re.sub(r'https?://\S+|www\.\S+', '', text)
        
        # Remove HTML tags
        text = re.sub(r'<.*?>', '', text)
        
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        
        return text
    
    def _extract_named_entities(self, doc) -> Dict[str, List[str]]:
        """Extract named entities from processed text."""
        entities = {
            "PERSON": [],
            "ORG": [],
            "GPE": [],  # Countries, cities, states
            "LOC": [],  # Non-GPE locations
            "DATE": [],
            "MISC": []  # Other entities
        }
        
        # Extract and deduplicate entities
        for ent in doc.ents:
            if ent.label_ in entities:
                if ent.text not in entities[ent.label_]:
                    entities[ent.label_].append(ent.text)
            else:
                if ent.text not in entities["MISC"]:
                    entities["MISC"].append(ent.text)
        
        return entities
    
    def _extract_key_phrases(self, doc) -> List[str]:
        """Extract key noun phrases that may represent topics."""
        noun_phrases = []
        
        # Extract noun chunks
        for chunk in doc.noun_chunks:
            # Filter to more meaningful noun phrases (longer than one token)
            if chunk.root.pos_ == "NOUN" and len(chunk) > 1:
                # Exclude chunks with determiners only
                if not (len(chunk) == 2 and chunk[0].pos_ == "DET"):
                    noun_phrases.append(chunk.text)
        
        # Deduplicate and limit to top phrases
        unique_phrases = list(set(noun_phrases))
        return unique_phrases[:20]  # Limit to top 20 phrases
    
    def _analyze_sentiment(self, text: str, title: str) -> Dict[str, Any]:
        """Analyze the sentiment of the article."""
        # Combine title and beginning of text for sentiment analysis
        # This is often more indicative of the overall sentiment than the full text
        sample_text = f"{title}. {text[:1000]}"
        
        try:
            sentiment_result = sentiment_analyzer(sample_text)[0]
            
            # Convert to a standardized format
            if sentiment_result["label"] == "POSITIVE":
                polarity = sentiment_result["score"]
            else:
                polarity = -sentiment_result["score"]
                
            return {
                "polarity": polarity,
                "label": "positive" if polarity > 0 else "negative" if polarity < 0 else "neutral",
                "confidence": abs(sentiment_result["score"])
            }
        except Exception as e:
            print(f"Error in sentiment analysis: {e}")
            return {"polarity": 0, "label": "neutral", "confidence": 0}
    
    def _detect_bias(self, text: str, title: str) -> Dict[str, float]:
        """
        Detect bias markers in the text.
        
        Returns:
            Dictionary with bias categories and their scores
        """
        text_lower = (title + " " + text).lower()
        bias_scores = {}
        
        # Check for bias markers in different categories
        for category, terms in BIAS_INDICATORS.items():
            count = 0
            for term in terms:
                count += text_lower.count(term)
            
            # Normalize by text length
            text_length = len(text_lower.split())
            if text_length > 0:
                normalized_score = (count / text_length) * 1000  # Per 1000 words
            else:
                normalized_score = 0
                
            bias_scores[category] = normalized_score
        
        return bias_scores
    
    def _extract_features(self, doc, analysis_results: Dict[str, Any]) -> Dict[str, Any]:
        """
        Extract features that can be used for machine learning models.
        
        Args:
            doc: spaCy processed document
            analysis_results: Results from previous analysis steps
            
        Returns:
            Dictionary of features
        """
        # Calculate basic text statistics
        sentence_count = len(list(doc.sents))
        word_count = len([token for token in doc if not token.is_punct and not token.is_space])
        avg_word_length = np.mean([len(token.text) for token in doc 
                                  if not token.is_punct and not token.is_space]) if word_count > 0 else 0
        
        # Calculate readability metrics (simplified Flesch-Kincaid)
        if sentence_count > 0:
            avg_sentence_length = word_count / sentence_count
        else:
            avg_sentence_length = 0
            
        # Extract POS tag distribution
        pos_counts = {}
        for token in doc:
            pos = token.pos_
            if pos not in pos_counts:
                pos_counts[pos] = 0
            pos_counts[pos] += 1
        
        # Normalize POS counts by total tokens
        total_tokens = len(doc)
        pos_distribution = {pos: count/total_tokens for pos, count in pos_counts.items()} if total_tokens > 0 else {}
        
        # Combine features
        features = {
            "text_stats": {
                "sentence_count": sentence_count,
                "word_count": word_count,
                "avg_word_length": avg_word_length,
                "avg_sentence_length": avg_sentence_length
            },
            "pos_distribution": pos_distribution,
            "entity_counts": {entity_type: len(entities) 
                             for entity_type, entities in analysis_results["named_entities"].items()},
            "sentiment_features": {
                "polarity": analysis_results["sentiment"]["polarity"],
                "sentiment_confidence": analysis_results["sentiment"]["confidence"]
            },
            "bias_features": analysis_results["bias_markers"]
        }
        
        return features


# Functions for direct use without instantiating the class
def process_article(text: str, title: str = "", metadata: Dict[str, Any] = None) -> Dict[str, Any]:
    """Process a single article with the NLP pipeline."""
    pipeline = NewsPipeline()
    return pipeline.process_article(text, title, metadata)

def batch_process_articles(articles: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Process multiple articles with the NLP pipeline."""
    pipeline = NewsPipeline()
    return pipeline.batch_process(articles)

def get_text_summary(analysis_result: Dict[str, Any]) -> str:
    """
    Generate a simple text summary from analysis results.
    This is a placeholder - for actual summarization, use the summarizer module.
    """
    entities = analysis_result["named_entities"]
    sentiment = analysis_result["sentiment"]["label"]
    
    # Build a simple summary
    key_people = ", ".join(entities["PERSON"][:3])
    key_orgs = ", ".join(entities["ORG"][:3])
    key_places = ", ".join(entities["GPE"][:3] + entities["LOC"][:3])
    
    summary = f"Article mentions {key_people or 'no specific people'}"
    if key_orgs:
        summary += f", organizations including {key_orgs}"
    if key_places:
        summary += f", and locations like {key_places}"
    summary += f". The sentiment appears {sentiment}."
    
    return summary

if __name__ == "__main__":
    # Example usage
    sample_text = """
    The recent climate summit in Paris brought together leaders from around the world. 
    President Biden and Chancellor Merkel discussed new environmental policies that would 
    significantly reduce carbon emissions by 2030. Critics from the Republican party argue 
    that these measures could harm economic growth. Meanwhile, Greenpeace activists protested 
    outside the venue, demanding more immediate action on climate change.
    """
    
    result = process_article(sample_text, "Climate Summit Brings Mixed Reactions")
    print(f"Entities: {result['named_entities']}")
    print(f"Sentiment: {result['sentiment']}")
    print(f"Bias Markers: {result['bias_markers']}")
    print(f"Summary: {get_text_summary(result)}")
