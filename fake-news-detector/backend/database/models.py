"""
Database Models for Fake News Detection System
==============================================
This module defines all database models using SQLAlchemy ORM for the fake news detection system.

Models include:
- GNewsArticle: Main article storage
- ArticleHash: Content deduplication
- AnalysisResult: NLP analysis results storage  
- CredibilityScore: Credibility assessment storage
- SourceReputation: Source credibility ratings
- UserFeedback: User ratings and feedback
- ProcessingLog: System processing logs
"""

from sqlalchemy import (
    Column, Integer, String, Text, DateTime, Float, Boolean, 
    JSON, ForeignKey, UniqueConstraint, Index, Enum
)
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship
from datetime import datetime
import enum

# Create base class for all models
Base = declarative_base()

# Enums for various fields
class CredibilityLevel(enum.Enum):
    VERY_LOW = "very_low"
    LOW = "low"
    QUESTIONABLE = "questionable"
    SOMEWHAT_CREDIBLE = "somewhat_credible"
    CREDIBLE = "credible"
    HIGHLY_CREDIBLE = "highly_credible"

class SentimentLabel(enum.Enum):
    POSITIVE = "positive"
    NEGATIVE = "negative"
    NEUTRAL = "neutral"

class BiasDirection(enum.Enum):
    LEFT = "left"
    RIGHT = "right"
    NEUTRAL = "neutral"
    SATIRE = "satire"
    UNKNOWN = "unknown"

class ProcessingStatus(enum.Enum):
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"


class GNewsArticle(Base):
    """
    Main article storage model for news articles from GNews API and other sources.
    """
    __tablename__ = 'gnews_articles'
    
    # Primary key
    id = Column(Integer, primary_key=True, autoincrement=True)
    
    # Article content
    title = Column(String(500), nullable=False, index=True)
    description = Column(Text)
    content = Column(Text)
    url = Column(String(1000), unique=True, nullable=False, index=True)
    image = Column(String(1000))
    
    # Publication meta_data
    published_at = Column(DateTime, index=True)
    source_name = Column(String(200), index=True)
    author = Column(String(200))
    
    # System meta_data
    timestamp = Column(DateTime, default=datetime.utcnow, index=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    # Processing flags
    is_processed = Column(Boolean, default=False, index=True)
    is_duplicate = Column(Boolean, default=False, index=True)
    processing_status = Column(Enum(ProcessingStatus), default=ProcessingStatus.PENDING, index=True)
    
    # Content metrics
    word_count = Column(Integer)
    sentence_count = Column(Integer)
    
    # Relationships
    analysis_results = relationship("AnalysisResult", back_populates="article", cascade="all, delete-orphan")
    credibility_scores = relationship("CredibilityScore", back_populates="article", cascade="all, delete-orphan")
    user_feedback = relationship("UserFeedback", back_populates="article", cascade="all, delete-orphan")
    
    # Indexes for performance
    _table_args_ = (
        Index('idx_article_source_date', 'source_name', 'published_at'),
        Index('idx_article_status_timestamp', 'processing_status', 'timestamp'),
        Index('idx_article_duplicate_processed', 'is_duplicate', 'is_processed'),
    )
    
    def _repr_(self):
        return f"<GNewsArticle(id={self.id}, title='{self.title[:50]}...', source='{self.source_name}')>"
    
    def to_dict(self):
        """Convert article to dictionary representation."""
        return {
            'id': self.id,
            'title': self.title,
            'description': self.description,
            'content': self.content,
            'url': self.url,
            'image': self.image,
            'published_at': self.published_at.isoformat() if self.published_at else None,
            'source_name': self.source_name,
            'author': self.author,
            'timestamp': self.timestamp.isoformat() if self.timestamp else None,
            'is_processed': self.is_processed,
            'is_duplicate': self.is_duplicate,
            'processing_status': self.processing_status.value if self.processing_status else None,
            'word_count': self.word_count,
            'sentence_count': self.sentence_count
        }


class ArticleHash(Base):
    """
    Content hashing for duplicate detection.
    Stores SHA-256 hashes of article content for quick duplicate checking.
    """
    __tablename__ = 'article_hashes'
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    hash = Column(String(64), nullable=False, unique=True, index=True)  # SHA-256 hash
    article_id = Column(Integer, ForeignKey('gnews_articles.id'), nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow, index=True)
    
    # Optional meta_data for the hash
    hash_type = Column(String(50), default='sha256')  # For future algorithm flexibility
    content_type = Column(String(50), default='full')  # 'full', 'title', 'summary', etc.
    
    # Relationship
    article = relationship("GNewsArticle")
    
    _table_args_ = (
        UniqueConstraint('hash', name='uq_article_hash'),
        Index('idx_hash_created', 'created_at'),
    )
    
    def _repr_(self):
        return f"<ArticleHash(id={self.id}, hash='{self.hash[:16]}...', type='{self.content_type}')>"


class AnalysisResult(Base):
    """
    Storage for NLP analysis results including sentiment, entities, bias detection, etc.
    """
    __tablename__ = 'analysis_results'
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    article_id = Column(Integer, ForeignKey('gnews_articles.id'), nullable=False, index=True)
    
    # Analysis meta_data
    analysis_version = Column(String(20), default='1.0')
    processed_at = Column(DateTime, default=datetime.utcnow, index=True)
    processing_time_ms = Column(Integer)  # Time taken for analysis in milliseconds
    
    # Text preprocessing results
    cleaned_text = Column(Text)
    word_count = Column(Integer)
    sentence_count = Column(Integer)
    paragraph_count = Column(Integer)
    
    # Named entities (stored as JSON)
    named_entities = Column(JSON)  # {"PERSON": ["John Doe"], "ORG": ["NASA"], ...}
    key_phrases = Column(JSON)     # ["climate change", "global warming", ...]
    
    # Sentiment analysis
    sentiment_label = Column(Enum(SentimentLabel))
    sentiment_polarity = Column(Float)      # -1.0 to 1.0
    sentiment_confidence = Column(Float)    # 0.0 to 1.0
    
    # Bias detection
    bias_markers = Column(JSON)  # {"political_left": 2.1, "emotional": 3.5, ...}
    overall_bias_score = Column(Float)      # 0.0 to 100.0
    bias_direction = Column(Enum(BiasDirection))
    
    # Language features
    readability_score = Column(Float)
    avg_word_length = Column(Float)
    avg_sentence_length = Column(Float)
    
    # Content quality indicators
    clickbait_score = Column(Float)         # 0.0 to 100.0
    emotional_language_score = Column(Float) # 0.0 to 100.0
    factual_language_score = Column(Float)   # 0.0 to 100.0
    
    # Verification indicators
    quotes_count = Column(Integer)
    links_count = Column(Integer)
    attributions_count = Column(Integer)
    
    # Full feature vector (for ML models)
    feature_vector = Column(JSON)
    
    # Relationship
    article = relationship("GNewsArticle", back_populates="analysis_results")
    
    _table_args_ = (
        Index('idx_analysis_article_version', 'article_id', 'analysis_version'),
        Index('idx_analysis_processed_at', 'processed_at'),
    )
    
    def _repr_(self):
        return f"<AnalysisResult(id={self.id}, article_id={self.article_id}, sentiment='{self.sentiment_label}')>"
    
    def to_dict(self):
        """Convert analysis result to dictionary."""
        return {
            'id': self.id,
            'article_id': self.article_id,
            'analysis_version': self.analysis_version,
            'processed_at': self.processed_at.isoformat() if self.processed_at else None,
            'processing_time_ms': self.processing_time_ms,
            'word_count': self.word_count,
            'sentence_count': self.sentence_count,
            'named_entities': self.named_entities,
            'key_phrases': self.key_phrases,
            'sentiment': {
                'label': self.sentiment_label.value if self.sentiment_label else None,
                'polarity': self.sentiment_polarity,
                'confidence': self.sentiment_confidence
            },
            'bias_markers': self.bias_markers,
            'overall_bias_score': self.overall_bias_score,
            'bias_direction': self.bias_direction.value if self.bias_direction else None,
            'readability_score': self.readability_score,
            'clickbait_score': self.clickbait_score,
            'emotional_language_score': self.emotional_language_score,
            'quotes_count': self.quotes_count,
            'links_count': self.links_count,
            'attributions_count': self.attributions_count
        }


class CredibilityScore(Base):
    """
    Storage for credibility assessment results.
    """
    __tablename__ = 'credibility_scores'
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    article_id = Column(Integer, ForeignKey('gnews_articles.id'), nullable=False, index=True)
    
    # Scoring meta_data
    scoring_version = Column(String(20), default='1.0')
    scored_at = Column(DateTime, default=datetime.utcnow, index=True)
    
    # Overall credibility
    overall_score = Column(Float, nullable=False, index=True)  # 0.0 to 100.0
    credibility_level = Column(Enum(CredibilityLevel), index=True)
    
    # Category scores
    source_reputation_score = Column(Float)     # 0.0 to 100.0
    content_quality_score = Column(Float)       # 0.0 to 100.0
    verifiability_score = Column(Float)         # 0.0 to 100.0
    presentation_score = Column(Float)          # 0.0 to 100.0
    
    # Detailed breakdown (stored as JSON)
    score_breakdown = Column(JSON)  # Detailed explanation of scoring
    
    # Source information
    source_domain = Column(String(200))
    source_credibility = Column(Float)
    
    # Human-readable explanation
    explanation = Column(Text)
    
    # Fact-check links
    fact_check_links = Column(JSON)  # Array of fact-checking URLs
    
    # Relationship
    article = relationship("GNewsArticle", back_populates="credibility_scores")
    
    _table_args_ = (
        Index('idx_credibility_score_level', 'overall_score', 'credibility_level'),
        Index('idx_credibility_article_version', 'article_id', 'scoring_version'),
    )
    
    def _repr_(self):
        return f"<CredibilityScore(id={self.id}, article_id={self.article_id}, score={self.overall_score}, level='{self.credibility_level}')>"
    
    def to_dict(self):
        """Convert credibility score to dictionary."""
        return {
            'id': self.id,
            'article_id': self.article_id,
            'scoring_version': self.scoring_version,
            'scored_at': self.scored_at.isoformat() if self.scored_at else None,
            'overall_score': self.overall_score,
            'credibility_level': self.credibility_level.value if self.credibility_level else None,
            'category_scores': {
                'source_reputation': self.source_reputation_score,
                'content_quality': self.content_quality_score,
                'verifiability': self.verifiability_score,
                'presentation': self.presentation_score
            },
            'score_breakdown': self.score_breakdown,
            'source_domain': self.source_domain,
            'source_credibility': self.source_credibility,
            'explanation': self.explanation,
            'fact_check_links': self.fact_check_links
        }


class SourceReputation(Base):
    """
    Storage for source reputation data and credibility ratings.
    """
    __tablename__ = 'source_reputation'
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    
    # Source identification
    domain = Column(String(200), unique=True, nullable=False, index=True)
    source_name = Column(String(300))
    
    # Reputation scores
    credibility_score = Column(Float, default=50.0)  # 0.0 to 100.0
    factual_reporting = Column(String(50))           # 'very high', 'high', 'mixed', 'low', 'very low'
    bias_rating = Column(String(50))                 # 'left', 'slight left', 'neutral', 'slight right', 'right'
    
    # Meta_data
    last_updated = Column(DateTime, default=datetime.utcnow, index=True)
    verification_source = Column(String(200))        # Where the reputation data came from
    
    # Statistics
    articles_analyzed = Column(Integer, default=0)
    avg_article_credibility = Column(Float)
    
    # Additional info (stored as JSON)
    additional_meta_data = Column(JSON)
    
    _table_args_ = (
        Index('idx_source_credibility', 'credibility_score'),
        Index('idx_source_updated', 'last_updated'),
    )
    
    def _repr_(self):
        return f"<SourceReputation(id={self.id}, domain='{self.domain}', credibility={self.credibility_score})>"


class UserFeedback(Base):
    """
    Storage for user feedback and ratings on articles and credibility assessments.
    """
    __tablename__ = 'user_feedback'
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    article_id = Column(Integer, ForeignKey('gnews_articles.id'), nullable=False, index=True)
    
    # User identification (anonymous for privacy)
    user_session_id = Column(String(64), index=True)  # Hashed session ID
    
    # Feedback data
    credibility_rating = Column(Integer)  # 1-5 scale
    is_helpful = Column(Boolean)          # Was our analysis helpful?
    is_accurate = Column(Boolean)         # Was our analysis accurate?
    
    # User's assessment
    user_thinks_fake = Column(Boolean)    # User's opinion on article authenticity
    user_bias_rating = Column(String(50)) # User's perception of bias
    
    # Comments
    comment = Column(Text)                # Optional user comment
    
    # Meta_data
    submitted_at = Column(DateTime, default=datetime.utcnow, index=True)
    ip_hash = Column(String(64))          # Hashed IP for spam prevention
    
    # Relationship
    article = relationship("GNewsArticle", back_populates="user_feedback")
    
    _table_args_ = (
        Index('idx_feedback_article_date', 'article_id', 'submitted_at'),
        Index('idx_feedback_rating', 'credibility_rating'),
    )
    
    def _repr_(self):
        return f"<UserFeedback(id={self.id}, article_id={self.article_id}, rating={self.credibility_rating})>"


class ProcessingLog(Base):
    """
    Logging table for system processing events and errors.
    """
    __tablename__ = 'processing_logs'
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    
    # Processing information
    process_type = Column(String(50), nullable=False, index=True)  # 'article_fetch', 'analysis', 'credibility', etc.
    process_status = Column(Enum(ProcessingStatus), nullable=False, index=True)
    
    # Related entities
    article_id = Column(Integer, ForeignKey('gnews_articles.id'), nullable=True, index=True)
    batch_id = Column(String(64), index=True)  # For batch processing tracking
    
    # Timing
    started_at = Column(DateTime, nullable=False, index=True)
    completed_at = Column(DateTime)
    processing_time_ms = Column(Integer)
    
    # Results
    items_processed = Column(Integer, default=0)
    items_successful = Column(Integer, default=0)
    items_failed = Column(Integer, default=0)
    
    # Error information
    error_message = Column(Text)
    error_type = Column(String(100))
    stack_trace = Column(Text)
    
    # Additional meta_data
    meta_data = Column(JSON)
    
    # Relationship
    article = relationship("GNewsArticle")
    
    _table_args_ = (
        Index('idx_log_type_status', 'process_type', 'process_status'),
        Index('idx_log_started_at', 'started_at'),
        Index('idx_log_batch', 'batch_id'),
    )
    
    def _repr_(self):
        return f"<ProcessingLog(id={self.id}, type='{self.process_type}', status='{self.process_status}')>"


class DuplicateMatch(Base):
    """
    Storage for duplicate detection results and similar article matches.
    """
    __tablename__ = 'duplicate_matches'
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    
    # Article relationships
    original_article_id = Column(Integer, ForeignKey('gnews_articles.id'), nullable=False, index=True)
    duplicate_article_id = Column(Integer, ForeignKey('gnews_articles.id'), nullable=False, index=True)
    
    # Similarity metrics
    similarity_score = Column(Float, nullable=False, index=True)  # 0.0 to 100.0
    match_type = Column(String(50))  # 'exact', 'near_duplicate', 'similar', etc.
    
    # Detailed similarity breakdown
    text_similarity = Column(Float)
    title_similarity = Column(Float)
    entity_similarity = Column(Float)
    semantic_similarity = Column(Float)
    
    # Detection meta_data
    detected_at = Column(DateTime, default=datetime.utcnow, index=True)
    detection_algorithm = Column(String(50))
    algorithm_version = Column(String(20))
    
    # Relationships
    original_article = relationship("GNewsArticle", foreign_keys=[original_article_id])
    duplicate_article = relationship("GNewsArticle", foreign_keys=[duplicate_article_id])
    
    _table_args_ = (
        UniqueConstraint('original_article_id', 'duplicate_article_id', name='uq_duplicate_pair'),
        Index('idx_duplicate_similarity', 'similarity_score'),
        Index('idx_duplicate_type', 'match_type'),
    )
    
    def _repr_(self):
        return f"<DuplicateMatch(id={self.id}, original={self.original_article_id}, duplicate={self.duplicate_article_id}, score={self.similarity_score})>"


# Create all tables function
def create_tables(engine):
    """Create all database tables."""
    Base.meta_data.create_all(engine)


# Drop all tables function  
def drop_tables(engine):
    """Drop all database tables."""
    Base.meta_data.drop_all(engine)


# Helper function to get model by name
def get_model_by_name(model_name: str):
    """Get a model class by its name."""
    models = {
        'GNewsArticle': GNewsArticle,
        'ArticleHash': ArticleHash,
        'AnalysisResult': AnalysisResult,
        'CredibilityScore': CredibilityScore,
        'SourceReputation': SourceReputation,
        'UserFeedback': UserFeedback,
        'ProcessingLog': ProcessingLog,
        'DuplicateMatch': DuplicateMatch
    }
    return models.get(model_name)


# Model registry for easy access
MODEL_REGISTRY = {
    'article': GNewsArticle,
    'hash': ArticleHash,
    'analysis': AnalysisResult,
    'credibility': CredibilityScore,
    'source': SourceReputation,
    'feedback': UserFeedback,
    'log': ProcessingLog,
    'duplicate': DuplicateMatch
}