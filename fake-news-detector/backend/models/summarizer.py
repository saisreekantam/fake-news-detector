"""
Text Summarization Module for Fake News Detection
-------------------------------------------------
This module provides text summarization capabilities for news articles,
using both extractive and abstractive techniques.

The summarization is a critical component for duplicate detection,
as comparing full article text would be inefficient and prone to error.
"""

import re
import nltk
from nltk.tokenize import sent_tokenize
from nltk.corpus import stopwords
import numpy as np
from typing import List, Dict, Tuple, Any, Optional
import heapq
import logging

# For advanced transformers-based summarization
try:
    from transformers import pipeline, AutoModelForSeq2SeqLM, AutoTokenizer
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False
    logging.warning("Transformers library not available. Using only extractive summarization.")

# Initialize NLTK resources if not already downloaded
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')

# Initialize logging
logger = logging.getLogger(__name__)
handler = logging.StreamHandler()
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
handler.setFormatter(formatter)
logger.addHandler(handler)
logger.setLevel(logging.INFO)

class Summarizer:
    """Text summarization for news articles."""
    
    def __init__(self, use_transformers: bool = True, model_name: str = "facebook/bart-large-cnn"):
        """
        Initialize the summarizer.
        
        Args:
            use_transformers: Whether to use transformers-based abstractive summarization
            model_name: Name of the pre-trained model to use for abstractive summarization
        """
        self.use_transformers = use_transformers and TRANSFORMERS_AVAILABLE
        self.model_name = model_name
        self.stop_words = set(stopwords.words('english'))
        
        # Initialize abstractive summarizer if available
        if self.use_transformers:
            try:
                # Load abstractive summarization model (BART)
                logger.info(f"Loading abstractive summarization model: {model_name}")
                self.abstractive_summarizer = pipeline(
                    "summarization", 
                    model=model_name,
                    truncation=True
                )
                logger.info("Abstractive summarization model loaded successfully")
            except Exception as e:
                logger.error(f"Error loading abstractive summarization model: {e}")
                self.use_transformers = False
    
    def summarize(self, text: str, title: str = "", max_length: int = 150, 
                  min_length: int = 50, ratio: float = 0.2, 
                  method: str = "auto") -> Dict[str, Any]:
        """
        Generate a summary of the given text.
        
        Args:
            text: The text to summarize
            title: The title of the article (used to improve summarization)
            max_length: Maximum length of the summary
            min_length: Minimum length of the summary
            ratio: For extractive summarization, portion of sentences to keep
            method: 'abstractive', 'extractive', or 'auto' (chooses based on text length)
            
        Returns:
            Dictionary containing summary and metadata
        """
        # Clean the text
        clean_text = self._preprocess_text(text)
        
        # Choose method based on text length if 'auto' is selected
        if method == "auto":
            # For longer texts or when transformers unavailable, use extractive
            if len(clean_text.split()) > 500 or not self.use_transformers:
                method = "extractive"
            else:
                method = "abstractive"
        
        # Generate summary based on method
        if method == "abstractive" and self.use_transformers:
            try:
                summary_text = self._abstractive_summarize(
                    clean_text, title, max_length, min_length
                )
            except Exception as e:
                logger.error(f"Error in abstractive summarization: {e}")
                logger.info("Falling back to extractive summarization")
                summary_text = self._extractive_summarize(clean_text, title, ratio)
        else:
            summary_text = self._extractive_summarize(clean_text, title, ratio)
        
        # Create fingerprint for the summary (useful for duplicate detection)
        summary_fingerprint = self._create_fingerprint(summary_text)
        
        return {
            "summary": summary_text,
            "method": method,
            "original_length": len(text.split()),
            "summary_length": len(summary_text.split()),
            "compression_ratio": len(summary_text.split()) / len(text.split()) if len(text.split()) > 0 else 0,
            "fingerprint": summary_fingerprint
        }
    
    def _preprocess_text(self, text: str) -> str:
        """Clean and normalize text for summarization."""
        # Remove URLs
        text = re.sub(r'https?://\S+|www\.\S+', '', text)
        
        # Remove HTML tags
        text = re.sub(r'<.*?>', '', text)
        
        # Replace newlines with spaces
        text = re.sub(r'\n+', ' ', text)
        
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        
        return text
    
    def _extractive_summarize(self, text: str, title: str = "", ratio: float = 0.2) -> str:
        """
        Perform extractive summarization by ranking sentences.
        
        Args:
            text: Text to summarize
            title: Article title
            ratio: Portion of sentences to include in summary
            
        Returns:
            Extractive summary text
        """
        # Tokenize into sentences
        sentences = sent_tokenize(text)
        
        # Handle very short texts
        if len(sentences) <= 3:
            return text
        
        # Create frequency distribution of words
        word_frequencies = {}
        for sentence in sentences:
            for word in nltk.word_tokenize(sentence.lower()):
                if word not in self.stop_words and word.isalnum():
                    if word not in word_frequencies:
                        word_frequencies[word] = 1
                    else:
                        word_frequencies[word] += 1
        
        # Normalize word frequencies
        max_frequency = max(word_frequencies.values()) if word_frequencies else 1
        for word in word_frequencies:
            word_frequencies[word] = word_frequencies[word] / max_frequency
        
        # Score sentences based on word frequencies
        sentence_scores = {}
        for i, sentence in enumerate(sentences):
            # Give higher weight to beginning sentences
            position_weight = 1.0 if i < len(sentences) * 0.3 else 0.8
            
            # Title bias - sentences containing title words get higher scores
            title_words = set(w.lower() for w in nltk.word_tokenize(title) 
                           if w.lower() not in self.stop_words and w.isalnum())
            sentence_words = set(w.lower() for w in nltk.word_tokenize(sentence) 
                              if w.lower() not in self.stop_words and w.isalnum())
            title_overlap = len(title_words.intersection(sentence_words)) / len(title_words) if title_words else 0
            title_weight = 1.0 + (0.5 * title_overlap)
            
            # Calculate base score from word frequencies
            score = 0
            for word in nltk.word_tokenize(sentence.lower()):
                if word in word_frequencies:
                    score += word_frequencies[word]
            
            # Normalize by sentence length and apply weights
            if len(nltk.word_tokenize(sentence)) > 0:
                score = score / len(nltk.word_tokenize(sentence)) * position_weight * title_weight
                sentence_scores[i] = score
        
        # Determine how many sentences to include
        num_sentences = max(int(len(sentences) * ratio), 3)  # At least 3 sentences
        num_sentences = min(num_sentences, 10)  # But no more than 10
        
        # Get top scoring sentences while preserving original order
        top_sentence_indices = heapq.nlargest(num_sentences, sentence_scores, key=sentence_scores.get)
        top_sentence_indices = sorted(top_sentence_indices)
        
        # Combine sentences into summary
        summary = " ".join([sentences[i] for i in top_sentence_indices])
        
        return summary
    
    def _abstractive_summarize(self, text: str, title: str = "", 
                               max_length: int = 150, min_length: int = 50) -> str:
        """
        Perform abstractive summarization using transformers.
        
        Args:
            text: Text to summarize
            title: Article title
            max_length: Maximum length in tokens
            min_length: Minimum length in tokens
            
        Returns:
            Abstractive summary text
        """
        if not self.use_transformers:
            raise ValueError("Transformers library not available for abstractive summarization")
        
        # Add title to beginning if available (helps with focus)
        full_text = f"{title}. {text}" if title else text
        
        # Truncate if needed (model has a token limit)
        words = full_text.split()
        if len(words) > 1024:  # Most models have a context limit of around 1024 tokens
            full_text = " ".join(words[:1024])
        
        # Generate summary
        try:
            summary = self.abstractive_summarizer(
                full_text, 
                max_length=max_length,
                min_length=min_length,
                do_sample=False
            )
            return summary[0]['summary_text']
        except Exception as e:
            logger.error(f"Error in abstractive summarization: {e}")
            # Fall back to extractive in case of error
            return self._extractive_summarize(text, title)
    
    def _create_fingerprint(self, text: str) -> Dict[str, Any]:
        """
        Create a fingerprint of the text for duplicate detection.
        
        Returns:
            Dictionary with fingerprint data
        """
        # Tokenize and filter out stopwords
        words = [word.lower() for word in nltk.word_tokenize(text) 
                if word.lower() not in self.stop_words and word.isalnum()]
        
        # Get most frequent terms (useful for duplicate detection)
        word_freq = {}
        for word in words:
            if word not in word_freq:
                word_freq[word] = 1
            else:
                word_freq[word] += 1
        
        top_terms = heapq.nlargest(20, word_freq, key=word_freq.get)
        
        # Extract named entities using basic patterns
        # (A more sophisticated approach would use NER from spaCy or similar)
        capital_word_pattern = r'\b[A-Z][a-zA-Z]*\b'
        potential_entities = re.findall(capital_word_pattern, text)
        entity_candidates = [e for e in potential_entities if e.lower() not in self.stop_words]
        
        # Create simple content hash based on top terms
        content_hash = hash(" ".join(sorted(top_terms)))
        
        return {
            "top_terms": top_terms,
            "entity_candidates": entity_candidates[:10],  # Top 10 potential entities
            "content_hash": content_hash,
            "word_count": len(words)
        }

# Create a singleton instance for direct use
default_summarizer = Summarizer(use_transformers=TRANSFORMERS_AVAILABLE)

def summarize_text(text: str, title: str = "", method: str = "auto") -> Dict[str, Any]:
    """
    Generate a summary of the given text.
    
    Args:
        text: The text to summarize
        title: The title of the article
        method: 'abstractive', 'extractive', or 'auto'
        
    Returns:
        Dictionary containing summary and metadata
    """
    return default_summarizer.summarize(text, title, method=method)

def summarize_for_comparison(text: str, title: str = "") -> Dict[str, Any]:
    """
    Generate a summary optimized for duplicate detection.
    This uses a consistent method to ensure summaries are comparable.
    
    Args:
        text: The text to summarize
        title: The title of the article
        
    Returns:
        Dictionary containing summary and fingerprint
    """
    # For comparison, always use a consistent method and settings
    result = default_summarizer.summarize(
        text, 
        title, 
        method="extractive",  # Extractive is more deterministic for comparison
        ratio=0.25  # Standardize the size for comparison
    )
    
    return {
        "summary": result["summary"],
        "fingerprint": result["fingerprint"],
        "original_length": result["original_length"]
    }

if __name__ == "__main__":
    # Example usage
    sample_text = """
    The European Union has reached a historic agreement on migration policy after years of debate. 
    The new legislation creates a system for sharing responsibility among EU member states for 
    hosting migrants and processing asylum applications. Under the new rules, countries can either 
    accept migrants, pay a financial contribution, or provide operational support. European Commission 
    President Ursula von der Leyen called it a "landmark agreement that delivers for all member states." 
    Critics argue that the agreement does not do enough to protect the rights of asylum seekers and 
    might lead to more detentions. However, supporters say it strikes a balance between border security 
    and humanitarian obligations. The legislation now needs to be formally approved by the European 
    Parliament and the Council before it can be implemented, which is expected to happen by 2026.
    """
    
    summary_result = summarize_text(sample_text, "EU Reaches Historic Migration Agreement")
    print(f"Summary: {summary_result['summary']}")
    print(f"Method: {summary_result['method']}")
    print(f"Compression: {summary_result['compression_ratio']:.2f}")
    
    # Generate fingerprint for comparison
    comparison_result = summarize_for_comparison(sample_text, "EU Reaches Historic Migration Agreement")
    print(f"Fingerprint Top Terms: {comparison_result['fingerprint']['top_terms']}")
