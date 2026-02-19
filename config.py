"""
Centralized configuration for the Auslan Sign Retrieval System.
All paths, model names, thresholds, and settings in one place.
"""

import os

# Base directory (project root)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Data paths
GLOSS_DICT_PATH = os.path.join(BASE_DIR, "data", "gloss", "auslan_dictionary.json")
TARGET_WORDS_PATH = os.path.join(BASE_DIR, "data", "target_words.json")
SYNONYM_MAPPING_PATH = os.path.join(
    BASE_DIR, "data", "synonyms", "synonym_mapping.json"
)
WORDNET_SYNONYMS_PATH = os.path.join(
    BASE_DIR, "data", "synonyms", "wordnet_synonyms.json"
)
# Video directories — scraped videos go to SCRAPED_VIDEO_DIR (D: drive) to save
# C: drive space.  The original 46 hand-curated videos stay in the repo.
# The app checks both when serving.
VIDEO_DIR = os.path.join(BASE_DIR, "media", "videos")
SCRAPED_VIDEO_DIR = os.environ.get("SCRAPED_VIDEO_DIR", r"D:\nlp\auslan-videos")
VIDEO_DIRS = [VIDEO_DIR, SCRAPED_VIDEO_DIR]

# Embedding cache
EMBEDDING_CACHE_DIR = os.path.join(BASE_DIR, ".cache")

# Model names — semantic model is configurable; swap to any sentence-transformers model
# Options: "intfloat/e5-base-v2" (recommended), "intfloat/e5-small-v2" (speed),
#          "Snowflake/snowflake-arctic-embed-m-v2.0", "jinaai/jina-embeddings-v3" (max accuracy),
#          "all-MiniLM-L6-v2" (legacy)
SEMANTIC_MODEL_NAME = "intfloat/e5-base-v2"
SENTIMENT_MODEL_NAME = "distilbert-base-uncased-finetuned-sst-2-english"
EMOTION_MODEL_NAME = "j-hartmann/emotion-english-distilroberta-base"
SPACY_MODEL_NAME = "en_core_web_sm"

# E5 models require prefixes for queries and passages.
# Set to empty strings for models that don't need them (e.g., MiniLM, arctic).
SEMANTIC_QUERY_PREFIX = "query: "
SEMANTIC_PASSAGE_PREFIX = "passage: "

# Matching defaults
DEFAULT_SEMANTIC_THRESHOLD = 0.6
FUZZY_MATCH_THRESHOLD = 85  # RapidFuzz score threshold (0-100)
SYNONYM_CONFIDENCE = 0.9
FUZZY_CONFIDENCE = 0.85
PHRASE_SYNONYM_CONFIDENCE = 0.95
EXACT_CONFIDENCE = 1.0

# Server settings
SERVER_HOST = "0.0.0.0"
SERVER_PORT = 5000
SECRET_KEY = os.environ.get("SECRET_KEY", "auslan-sign-system-dev")
DEBUG = os.environ.get("FLASK_DEBUG", "true").lower() == "true"

# LLM settings (Ollama)
LLM_MODEL = "qwen3:8b"
LLM_HOST = os.environ.get("OLLAMA_HOST", "http://localhost:11434")
LLM_TIMEOUT = 10  # seconds
LLM_ENABLED = True  # set False to disable LLM fallback entirely
