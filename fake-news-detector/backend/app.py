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

