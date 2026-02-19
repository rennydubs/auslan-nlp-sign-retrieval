"""
Shared pytest fixtures for the Auslan Sign Retrieval test suite.
Uses lightweight mock data so tests run without loading real ML models.
"""

import json
import os
import sys
import tempfile

import pytest

# Make sure the project root is on the path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


# ---------------------------------------------------------------------------
# Minimal mock dictionary (subset of real data, no ML models needed)
# ---------------------------------------------------------------------------

MOCK_GLOSS_DICT = {
    "hello": {
        "gloss": "HELLO",
        "category": "greeting",
        "video_url": "https://example.com/hello.mp4",
        "description": "A greeting sign",
        "synonyms": ["hi", "greetings"],
    },
    "help": {
        "gloss": "HELP",
        "category": "basic_needs",
        "video_url": "https://example.com/help.mp4",
        "description": "Asking for assistance",
        "synonyms": ["assist", "aid"],
    },
    "happy": {
        "gloss": "HAPPY",
        "category": "emotions",
        "video_url": "https://example.com/happy.mp4",
        "description": "Feeling joy",
        "synonyms": ["glad", "joyful"],
    },
    "exercise": {
        "gloss": "EXERCISE",
        "category": "fitness_actions",
        "video_url": "https://example.com/exercise.mp4",
        "description": "Physical activity",
        "synonyms": ["workout", "training"],
    },
    "water": {
        "gloss": "WATER",
        "category": "basic_needs",
        "video_url": "https://example.com/water.mp4",
        "description": "Drinking water",
        "synonyms": ["drink", "hydrate"],
    },
    "goodbye": {
        "gloss": "GOODBYE",
        "category": "greeting",
        "video_url": "https://example.com/goodbye.mp4",
        "description": "A farewell sign",
        "synonyms": ["bye", "farewell"],
    },
}

MOCK_TARGET_WORDS = {
    "target_words": [
        {"word": "hello", "synonyms": ["hi", "greetings", "howdy"]},
        {"word": "help",  "synonyms": ["assist", "aid", "support"]},
        {"word": "happy", "synonyms": ["glad", "joyful", "pleased"]},
        {"word": "exercise", "synonyms": ["workout", "training", "drill"]},
    ]
}

MOCK_SYNONYM_MAPPING = {
    "hi": "hello",
    "greetings": "hello",
    "assist": "help",
    "aid": "help",
    "support": "help",
    "glad": "happy",
    "joyful": "happy",
    "workout": "exercise",
    "training": "exercise",
}


@pytest.fixture(scope="session")
def tmp_data_dir():
    """Create a temporary directory with mock JSON data files."""
    with tempfile.TemporaryDirectory() as tmpdir:
        # Write mock files
        gloss_path = os.path.join(tmpdir, "auslan_dictionary.json")
        words_path = os.path.join(tmpdir, "target_words.json")
        syn_path   = os.path.join(tmpdir, "synonym_mapping.json")

        with open(gloss_path, "w") as f:
            json.dump(MOCK_GLOSS_DICT, f)
        with open(words_path, "w") as f:
            json.dump(MOCK_TARGET_WORDS, f)
        with open(syn_path, "w") as f:
            json.dump(MOCK_SYNONYM_MAPPING, f)

        yield {
            "gloss_dict": gloss_path,
            "target_words": words_path,
            "synonym_mapping": syn_path,
            "tmpdir": tmpdir,
        }


@pytest.fixture(scope="session")
def matcher(tmp_data_dir):
    """A SignMatcher instance using mock data and NO semantic model."""
    from src.matcher import SignMatcher
    return SignMatcher(
        gloss_dict_path=tmp_data_dir["gloss_dict"],
        target_words_path=tmp_data_dir["target_words"],
        synonym_mapping_path=tmp_data_dir["synonym_mapping"],
        semantic_model_name="all-MiniLM-L6-v2",  # won't load; no shared model provided
        shared_semantic_model=None,
        embedding_cache_dir=None,
    )


@pytest.fixture(scope="session")
def preprocessor():
    """A TextPreprocessor with no spaCy (pure tokenizer)."""
    from src.preprocessing import TextPreprocessor
    return TextPreprocessor(shared_spacy_model=None)


@pytest.fixture
def mock_dict():
    """Return the mock dictionary directly (for unit tests that don't need files)."""
    return MOCK_GLOSS_DICT
