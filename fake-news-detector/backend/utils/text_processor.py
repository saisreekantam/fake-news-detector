"""
Text Processing Utilities for Fake News Detection
-------------------------------------------------
This module provides functions for cleaning, normalizing, and
extracting features from news article text.

Functions include:
- Text cleaning (HTML removal, URL removal, whitespace normalization)
- Text normalization (stemming, lemmatization, stopword removal)
- Feature extraction (keyword identification, statistics)
- Structure analysis (paragraph detection, quote extraction)
"""

import re
import string
import unicodedata
import logging
from typing import List, Dict, Set, Tuple, Any, Optional
from collections import Counter
import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer, WordNetLemmatizer

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Ensure NLTK resources are available
try:
    nltk.data.find('tokenizers/punkt')
    nltk.data.find('corpora/stopwords')
    nltk.data.find('corpora/wordnet')
except LookupError:
    logger.warning("NLTK resources not found, downloading...")
    nltk.download('punkt')
    nltk.download('stopwords')
    nltk.download('wordnet')

# Initialize NLTK tools
stemmer = PorterStemmer()
lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))

def clean_html(text: str) -> str:
    """
    Remove HTML tags from text.
    
    Args:
        text: String containing HTML tags
        
    Returns:
        Cleaned string with HTML tags removed
    """
    # Remove HTML tags
    html_pattern = re.compile('<.*?>')
    clean_text = re.sub(html_pattern, '', text)
    
    # Remove HTML entities
    html_entity_pattern = re.compile('&[a-z]+;')
    clean_text = re.sub(html_entity_pattern, ' ', clean_text)
    
    return clean_text

def clean_urls(text: str) -> str:
    """
    Remove URLs from text.
    
    Args:
        text: String potentially containing URLs
        
    Returns:
        String with URLs removed
    """
    # Match common URL patterns
    url_pattern = re.compile(r'https?://\S+|www\.\S+')
    return re.sub(url_pattern, '', text)

def normalize_whitespace(text: str) -> str:
    """
    Normalize whitespace by removing extra spaces, tabs, newlines.
    
    Args:
        text: Input string
        
    Returns:
        String with normalized whitespace
    """
    # Replace multiple whitespace characters with a single space
    return re.sub(r'\s+', ' ', text).strip()

def normalize_unicode(text: str) -> str:
    """
    Normalize Unicode characters and remove control characters.
    
    Args:
        text: Input string
        
    Returns:
        Normalized string
    """
    # Normalize Unicode to the canonical form
    normalized = unicodedata.normalize('NFKD', text)
    
    # Remove control characters
    return ''.join(ch for ch in normalized if not unicodedata.category(ch).startswith('C'))

def remove_punctuation(text: str) -> str:
    """
    Remove punctuation from text.
    
    Args:
        text: Input string
        
    Returns:
        String without punctuation
    """
    # Create a translation table to remove punctuation
    translator = str.maketrans('', '', string.punctuation)
    return text.translate(translator)

def preprocess_text(text: str, 
                   remove_html: bool = True,
                   remove_urls: bool = True,
                   normalize_space: bool = True,
                   normalize_chars: bool = True) -> str:
    """
    Perform complete text preprocessing.
    
    Args:
        text: Input text
        remove_html: Whether to remove HTML tags
        remove_urls: Whether to remove URLs
        normalize_space: Whether to normalize whitespace
        normalize_chars: Whether to normalize Unicode characters
        
    Returns:
        Preprocessed text
    """
    if not text:
        return ""
    
    try:
        # Apply preprocessing steps in sequence
        if remove_html:
            text = clean_html(text)
        
        if remove_urls:
            text = clean_urls(text)
        
        if normalize_chars:
            text = normalize_unicode(text)
        
        if normalize_space:
            text = normalize_whitespace(text)
        
        return text
    except Exception as e:
        logger.error(f"Error preprocessing text: {e}")
        return text

def tokenize_text(text: str) -> List[str]:
    """
    Tokenize text into words.
    
    Args:
        text: Input text
        
    Returns:
        List of word tokens
    """
    return word_tokenize(text.lower())

def tokenize_sentences(text: str) -> List[str]:
    """
    Split text into sentences.
    
    Args:
        text: Input text
        
    Returns:
        List of sentences
    """
    return sent_tokenize(text)

def remove_stopwords(tokens: List[str]) -> List[str]:
    """
    Remove common stopwords from a token list.
    
    Args:
        tokens: List of word tokens
        
    Returns:
        List with stopwords removed
    """
    return [word for word in tokens if word.lower() not in stop_words]

def stem_words(tokens: List[str]) -> List[str]:
    """
    Apply stemming to word tokens.
    
    Args:
        tokens: List of word tokens
        
    Returns:
        List of stemmed tokens
    """
    return [stemmer.stem(word) for word in tokens]

def lemmatize_words(tokens: List[str]) -> List[str]:
    """
    Apply lemmatization to word tokens.
    
    Args:
        tokens: List of word tokens
        
    Returns:
        List of lemmatized tokens
    """
    return [lemmatizer.lemmatize(word) for word in tokens]

def extract_keywords(text: str, top_n: int = 10) -> List[Tuple[str, int]]:
    """
    Extract keywords from text based on frequency.
    
    Args:
        text: Input text
        top_n: Number of top keywords to return
        
    Returns:
        List of (keyword, frequency) tuples
    """
    # Tokenize and prepare text
    tokens = tokenize_text(text)
    tokens = remove_stopwords(tokens)
    
    # Remove very short words (likely not meaningful)
    tokens = [token for token in tokens if len(token) > 2]
    
    # Count occurrences
    word_counts = Counter(tokens)
    
    # Return top N keywords
    return word_counts.most_common(top_n)

def extract_quotes(text: str) -> List[str]:
    """
    Extract quoted text from article.
    
    Args:
        text: Article text
        
    Returns:
        List of extracted quotes
    """
    # Match text within double quotes
    quote_pattern = re.compile(r'"([^"]*)"')
    quotes = quote_pattern.findall(text)
    
    # Filter out short quotes (likely not actual quotations)
    return [quote for quote in quotes if len(quote) > 15]

def extract_paragraphs(text: str) -> List[str]:
    """
    Split text into paragraphs.
    
    Args:
        text: Article text
        
    Returns:
        List of paragraphs
    """
    # Split by double newlines (common paragraph separator)
    paragraphs = re.split(r'\n\s*\n', text)
    
    # Clean up each paragraph
    return [normalize_whitespace(p) for p in paragraphs if p.strip()]

def calculate_text_stats(text: str) -> Dict[str, Any]:
    """
    Calculate various statistics about the text.
    
    Args:
        text: Input text
        
    Returns:
        Dictionary of text statistics
    """
    # Prepare data
    words = tokenize_text(text)
    sentences = tokenize_sentences(text)
    paragraphs = extract_paragraphs(text)
    
    # Calculate statistics
    stats = {
        "word_count": len(words),
        "sentence_count": len(sentences),
        "paragraph_count": len(paragraphs),
        "avg_words_per_sentence": len(words) / max(1, len(sentences)),
        "avg_sentences_per_paragraph": len(sentences) / max(1, len(paragraphs)),
        "avg_word_length": sum(len(word) for word in words) / max(1, len(words))
    }
    
    return stats

def detect_language_features(text: str) -> Dict[str, Any]:
    """
    Detect various language features that might indicate credibility.
    
    Args:
        text: Input text
        
    Returns:
        Dictionary of language features
    """
    # Calculate basic metrics
    word_count = len(tokenize_text(text))
    
    # Check for clickbait patterns
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
    
    clickbait_matches = []
    for pattern in clickbait_patterns:
        matches = re.findall(pattern, text.lower())
        clickbait_matches.extend(matches)
    
    # Check for excessive punctuation
    exclamation_count = text.count('!')
    question_count = text.count('?')
    
    # Check for ALL CAPS words
    words = text.split()
    all_caps_words = [word for word in words if word.isupper() and len(word) > 2]
    
    # Check for attributions/citations
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
    
    return {
        "clickbait_matches": clickbait_matches,
        "clickbait_score": len(clickbait_matches) / max(1, word_count / 100),  # Normalized per 100 words
        "exclamation_density": exclamation_count / max(1, word_count / 100),  # Per 100 words
        "question_density": question_count / max(1, word_count / 100),  # Per 100 words
        "all_caps_ratio": len(all_caps_words) / max(1, len(words)),
        "attribution_count": len(attribution_matches),
        "has_attributions": len(attribution_matches) > 0
    }

def extract_dates(text: str) -> List[str]:
    """
    Extract dates mentioned in the text.
    
    Args:
        text: Input text
        
    Returns:
        List of date strings
    """
    # Basic date pattern matching (can be improved with more complex patterns)
    date_patterns = [
        r'\b\d{1,2}/\d{1,2}/\d{2,4}\b',  # MM/DD/YYYY
        r'\b\d{1,2}-\d{1,2}-\d{2,4}\b',  # MM-DD-YYYY
        r'\b(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]* \d{1,2},? \d{4}\b',  # Month DD, YYYY
        r'\b\d{1,2} (?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]* \d{4}\b',  # DD Month YYYY
        r'\b(?:yesterday|today|tomorrow)\b',
        r'\blast (?:week|month|year)\b',
        r'\bnext (?:week|month|year)\b'
    ]
    
    dates = []
    for pattern in date_patterns:
        matches = re.findall(pattern, text, re.IGNORECASE)
        dates.extend(matches)
    
    return dates

def extract_named_entities_basic(text: str) -> Dict[str, List[str]]:
    """
    Extract named entities using basic pattern matching.
    This is a simplified version - for better results, use spaCy or another NER tool.
    
    Args:
        text: Input text
        
    Returns:
        Dictionary of entity types and entities
    """
    entities = {
        "PERSON": [],
        "ORG": [],
        "LOCATION": []
    }
    
    # Match capitalized words that might be proper nouns
    proper_noun_pattern = r'\b[A-Z][a-zA-Z]+ (?:[A-Z][a-zA-Z]+)?\b'
    candidates = re.findall(proper_noun_pattern, text)
    
    # Organization patterns
    org_indicators = [
        'Inc', 'Corp', 'Corporation', 'Company', 'Ltd', 'LLC', 'LLP',
        'Association', 'University', 'College', 'School', 'Institute',
        'Department', 'Agency', 'Bureau', 'Commission', 'Committee',
        'Party', 'Group', 'Foundation', 'Society'
    ]
    
    # Location patterns
    location_indicators = [
        'Street', 'Road', 'Avenue', 'Boulevard', 'Lane', 'Drive',
        'City', 'Town', 'Village', 'County', 'District', 'State',
        'Country', 'Republic', 'Kingdom', 'Empire', 'Union',
        'River', 'Lake', 'Ocean', 'Sea', 'Mountain', 'Valley'
    ]
    
    # Categorize candidates
    for candidate in candidates:
        if any(indicator in candidate for indicator in org_indicators):
            if candidate not in entities["ORG"]:
                entities["ORG"].append(candidate)
        elif any(indicator in candidate for indicator in location_indicators):
            if candidate not in entities["LOCATION"]:
                entities["LOCATION"].append(candidate)
        else:
            # Assume person if not matched to other categories
            if candidate not in entities["PERSON"]:
                entities["PERSON"].append(candidate)
    
    return entities

def get_important_sentences(text: str, top_n: int = 5) -> List[str]:
    """
    Extract most important sentences from text.
    This is a simple extraction method, not a full summarization algorithm.
    
    Args:
        text: Input text
        top_n: Number of top sentences to return
        
    Returns:
        List of important sentences
    """
    # Split into sentences
    sentences = tokenize_sentences(text)
    
    if not sentences:
        return []
    
    # Calculate sentence scores based on word frequency
    word_freq = Counter()
    
    # Tokenize and count all words (excluding stopwords)
    for sentence in sentences:
        tokens = tokenize_text(sentence)
        tokens = remove_stopwords(tokens)
        word_freq.update(tokens)
    
    # Score sentences
    sentence_scores = []
    for sentence in sentences:
        tokens = tokenize_text(sentence)
        tokens = remove_stopwords(tokens)
        
        # Sum the frequencies of words in this sentence
        score = sum(word_freq[token] for token in tokens)
        
        # Normalize by sentence length to avoid bias toward longer sentences
        score = score / max(1, len(tokens))
        
        sentence_scores.append((sentence, score))
    
    # Sort by score and return top sentences
    top_sentences = sorted(sentence_scores, key=lambda x: x[1], reverse=True)
    return [sentence for sentence, score in top_sentences[:top_n]]

def preprocess_complete(text: str) -> Dict[str, Any]:
    """
    Perform complete preprocessing and feature extraction for a text.
    
    Args:
        text: Input text
        
    Returns:
        Dictionary containing all processed data and features
    """
    # Basic cleaning
    cleaned_text = preprocess_text(text)
    
    # Tokenization
    tokens = tokenize_text(cleaned_text)
    sentences = tokenize_sentences(cleaned_text)
    
    # Extract features
    stats = calculate_text_stats(text)
    language_features = detect_language_features(text)
    
    # Structured content
    paragraphs = extract_paragraphs(text)
    quotes = extract_quotes(text)
    dates = extract_dates(text)
    entities = extract_named_entities_basic(text)
    
    # Keywords and important content
    keywords = extract_keywords(cleaned_text)
    important_sentences = get_important_sentences(cleaned_text)
    
    # Assemble result
    result = {
        "original_text": text,
        "cleaned_text": cleaned_text,
        "tokens": tokens[:100],  # Limit token list to avoid too much data
        "sentences": sentences,
        "paragraphs": paragraphs,
        "quotes": quotes,
        "dates": dates,
        "statistics": stats,
        "language_features": language_features,
        "entities": entities,
        "keywords": keywords,
        "important_sentences": important_sentences
    }
    
    return result

# Simple standalone functions for direct usage
def clean_text(text: str) -> str:
    """Simple wrapper for text cleaning."""
    return preprocess_text(text)

def get_text_features(text: str) -> Dict[str, Any]:
    """Extract features from text for analysis."""
    cleaned_text = preprocess_text(text)
    return {
        "statistics": calculate_text_stats(text),
        "language_features": detect_language_features(text),
        "entities": extract_named_entities_basic(cleaned_text),
        "keywords": extract_keywords(cleaned_text)
    }

def is_clickbait(title: str) -> Tuple[bool, List[str]]:
    """
    Check if a title appears to be clickbait.
    
    Args:
        title: Article title
        
    Returns:
        Tuple of (is_clickbait, matched_patterns)
    """
    # Clean and lowercase the title
    clean_title = preprocess_text(title).lower()
    
    # Common clickbait patterns
    clickbait_patterns = [
        r"you won't believe",
        r"shocking",
        r"mind-blowing",
        r"what happens next",
        r"secret",
        r"\d+ ways",
        r"\d+ things",
        r"this is why",
        r"will make you",
        r"changed my life",
        r"jaw-dropping",
        r"they don't want you to know",
        r"you've been doing wrong",
        r"simple trick"
    ]
    
    # Check for matches
    matches = []
    for pattern in clickbait_patterns:
        if re.search(pattern, clean_title):
            matches.append(pattern)
    
    # Consider it clickbait if matching at least one pattern
    return (len(matches) > 0, matches)

if __name__ == "__main__":
    # Example usage
    sample_text = """
    The recent climate summit in Paris brought together leaders from around the world. 
    President Biden and Chancellor Merkel discussed new environmental policies that would 
    significantly reduce carbon emissions by 2030. Critics from the Republican party argue 
    that these measures could harm economic growth. Meanwhile, Greenpeace activists protested 
    outside the venue, demanding more immediate action on climate change.
    """
    
    print("Processing sample text...")
    result = preprocess_complete(sample_text)
    
    print(f"Keywords: {result['keywords']}")
    print(f"Entities: {result['entities']}")
    print(f"Statistics: {result['statistics']}")
    print(f"Important sentences: {result['important_sentences']}")
