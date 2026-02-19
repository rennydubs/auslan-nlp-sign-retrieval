"""
Text preprocessing module for Auslan sign retrieval system.
Handles tokenization, cleaning, normalization, and lemmatization of user input.
"""

import logging
import re
from typing import List, Optional

from nltk.stem import PorterStemmer

logger = logging.getLogger(__name__)

try:
    import spacy
    _SPACY_AVAILABLE = True
except ImportError:
    _SPACY_AVAILABLE = False


class TextPreprocessor:
    """
    Handles text preprocessing for sign language retrieval.
    Includes tokenization, normalization, lemmatization, and cleaning operations.
    """

    def __init__(self, spacy_model_name: str = "en_core_web_sm",
                 shared_spacy_model=None):
        """Initialize the preprocessor with stop words, contractions, and optional spaCy model."""
        self.stop_words = {
            'a', 'an', 'and', 'are', 'as', 'at', 'be', 'by', 'for', 'from',
            'has', 'he', 'in', 'is', 'it', 'its', 'of', 'on', 'that', 'the',
            'to', 'was', 'will', 'with', 'you', 'your'
        }

        self.contractions = {
            "don't": "do not",
            "won't": "will not",
            "can't": "cannot",
            "n't": " not",
            "'re": " are",
            "'ve": " have",
            "'ll": " will",
            "'d": " would",
            "'m": " am"
        }

        self.stemmer = PorterStemmer()

        # spaCy model for lemmatization (optional, loaded lazily)
        self._nlp: Optional[object] = None
        if shared_spacy_model is not None:
            self._nlp = shared_spacy_model
        elif _SPACY_AVAILABLE:
            try:
                self._nlp = spacy.load(spacy_model_name)
                logger.info("Preprocessor loaded spaCy model: %s", spacy_model_name)
            except OSError:
                logger.warning("spaCy model '%s' not found. Lemmatization unavailable.", spacy_model_name)

    def clean_text(self, text: str) -> str:
        """Clean input text by removing special characters and normalizing."""
        if not isinstance(text, str):
            return ""

        text = text.lower()

        for contraction, expansion in self.contractions.items():
            text = text.replace(contraction, expansion)

        # Remove punctuation except apostrophes
        text = re.sub(r'[^\w\s\']', ' ', text)

        # Remove extra whitespace
        text = ' '.join(text.split())

        return text

    def tokenize(self, text: str) -> List[str]:
        """Tokenize text into individual words."""
        cleaned_text = self.clean_text(text)
        tokens = cleaned_text.split()

        # Filter out empty tokens and single characters (except 'a', 'i')
        return [t for t in tokens if len(t) > 1 or t in ['a', 'i']]

    def remove_stop_words(self, tokens: List[str]) -> List[str]:
        """Remove common stop words from token list."""
        return [token for token in tokens if token not in self.stop_words]

    def stem_tokens(self, tokens: List[str]) -> List[str]:
        """Apply Porter stemming to tokens."""
        return [self.stemmer.stem(t) for t in tokens]

    def lemmatize_tokens(self, tokens: List[str]) -> List[str]:
        """Apply spaCy lemmatization to tokens.

        Lemmatization is more accurate than stemming because it uses
        vocabulary and morphological analysis to return the base form:
        "running" -> "run", "exercises" -> "exercise", "happier" -> "happy"

        Falls back to returning tokens unchanged if spaCy is not available.
        """
        if not self._nlp:
            return tokens

        # Process as a single string so spaCy can use POS context
        text = ' '.join(tokens)
        doc = self._nlp(text)

        lemmatized = []
        for token in doc:
            lemma = token.lemma_.lower()
            # spaCy sometimes returns -PRON- for pronouns; keep original in that case
            if lemma.startswith('-') and lemma.endswith('-'):
                lemmatized.append(token.text.lower())
            else:
                lemmatized.append(lemma)

        return lemmatized

    def preprocess(self, text: str, remove_stops: bool = False,
                   use_stemming: bool = False,
                   use_lemmatization: bool = False) -> List[str]:
        """Complete preprocessing pipeline.

        Args:
            text: Raw input text
            remove_stops: Whether to remove stop words
            use_stemming: Whether to apply Porter stemming
            use_lemmatization: Whether to apply spaCy lemmatization (preferred over stemming)

        Returns:
            Processed tokens ready for matching
        """
        tokens = self.tokenize(text)

        if remove_stops:
            tokens = self.remove_stop_words(tokens)

        # Lemmatization and stemming are mutually exclusive; prefer lemmatization
        if use_lemmatization:
            tokens = self.lemmatize_tokens(tokens)
        elif use_stemming:
            tokens = self.stem_tokens(tokens)

        return tokens

    def preprocess_batch(self, texts: List[str], remove_stops: bool = False,
                         use_stemming: bool = False,
                         use_lemmatization: bool = False) -> List[List[str]]:
        """Process multiple texts in batch."""
        return [self.preprocess(text, remove_stops, use_stemming, use_lemmatization)
                for text in texts]


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

    preprocessor = TextPreprocessor()

    test_texts = [
        "Hello, how are you today?",
        "I can't find the toilet.",
        "Let's go buy some food and eat together!",
        "Where is your house? I'd like to visit.",
        "My friend speaks very well.",
        "She was running exercises happily.",
    ]

    print("Testing Text Preprocessor:")
    print("=" * 50)

    for text in test_texts:
        tokens = preprocessor.preprocess(text)
        tokens_lemma = preprocessor.preprocess(text, use_lemmatization=True)
        tokens_stem = preprocessor.preprocess(text, use_stemming=True)

        print(f"Original:  {text}")
        print(f"Tokens:    {tokens}")
        print(f"Lemmatized:{tokens_lemma}")
        print(f"Stemmed:   {tokens_stem}")
        print("-" * 30)
