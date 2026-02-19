"""
Enhanced NLP features for Auslan sign retrieval system.
Includes sentiment analysis, named entity recognition, and context understanding.
"""

import logging
import re
from collections import defaultdict
from dataclasses import dataclass
from typing import Any, Dict, List, Tuple

logger = logging.getLogger(__name__)

# Check library availability at import time (no model loading)
try:
    import spacy
    import torch  # noqa: F401
    from sentence_transformers import SentenceTransformer  # noqa: F401
    from textblob import TextBlob  # noqa: F401
    from transformers import pipeline

    NLP_ENHANCED = True
except ImportError:
    NLP_ENHANCED = False
    logger.info("Advanced NLP libraries not installed. Using basic NLP features.")


@dataclass
class NLPAnalysis:
    """Container for comprehensive NLP analysis results."""

    sentiment_score: float  # -1 to 1
    sentiment_label: str  # positive, negative, neutral
    confidence: float  # 0 to 1
    entities: List[Dict[str, Any]]
    key_phrases: List[str]
    intent: str
    emotion: str
    formality_level: str
    complexity_score: float
    readability_score: float


class EnhancedNLPProcessor:
    """
    Advanced NLP processor with sentiment analysis, entity recognition,
    and context understanding capabilities.

    Models are loaded lazily on first use inside __init__, not at module import.
    A shared semantic model can be injected to avoid duplicate loading.
    """

    def __init__(
        self,
        spacy_model_name: str = "en_core_web_sm",
        sentiment_model_name: str = "distilbert-base-uncased-finetuned-sst-2-english",
        emotion_model_name: str = "j-hartmann/emotion-english-distilroberta-base",
        shared_semantic_model=None,
    ):
        """Initialize the NLP processor, loading models lazily."""
        self.nlp = None
        self.semantic_model = shared_semantic_model
        self.sentiment_model = None
        self.emotion_model = None
        self._nlp_enhanced = NLP_ENHANCED

        if NLP_ENHANCED:
            # Load spaCy
            try:
                self.nlp = spacy.load(spacy_model_name)
                logger.info("Loaded spaCy model: %s", spacy_model_name)
            except OSError:
                logger.warning(
                    "spaCy model '%s' not found. Some features limited.",
                    spacy_model_name,
                )

            # Load sentiment model
            try:
                self.sentiment_model = pipeline(
                    "sentiment-analysis",
                    model=sentiment_model_name,
                    tokenizer=sentiment_model_name,
                )
                logger.info("Loaded sentiment model: %s", sentiment_model_name)
            except Exception as e:
                logger.warning("Sentiment model not available: %s", e)

            # Load emotion model
            try:
                self.emotion_model = pipeline(
                    "text-classification",
                    model=emotion_model_name,
                    tokenizer=emotion_model_name,
                )
                logger.info("Loaded emotion model: %s", emotion_model_name)
            except Exception as e:
                logger.warning("Emotion model not available: %s", e)

        # Emotion lexicon for fine-grained emotion detection
        self.emotion_lexicon = {
            "joy": [
                "happy",
                "joyful",
                "elated",
                "cheerful",
                "pleased",
                "content",
                "glad",
            ],
            "sadness": ["sad", "depressed", "unhappy", "gloomy", "melancholy", "down"],
            "anger": ["angry", "furious", "mad", "irritated", "annoyed", "rage"],
            "fear": ["afraid", "scared", "anxious", "worried", "nervous", "frightened"],
            "surprise": ["surprised", "amazed", "astonished", "shocked", "stunned"],
            "disgust": ["disgusted", "revolted", "sick", "nauseated"],
            "trust": ["trust", "confident", "secure", "safe", "assured"],
            "anticipation": ["excited", "eager", "hopeful", "optimistic"],
        }

        # Intent patterns
        self.intent_patterns = {
            "instruction": [
                r"\b(do|perform|execute|practice|try)\b",
                r"\b(let\'s|let us)\b",
                r"\b(start|begin|commence)\b",
            ],
            "request": [
                r"\b(please|can you|could you|would you)\b",
                r"\b(need|want|require)\b",
                r"\b(help|assist|aid)\b",
            ],
            "question": [
                r"\b(how|what|where|when|why|who)\b",
                r"\?",
                r"\b(is|are|do|does|can|will)\b.*\?",
            ],
            "information": [
                r"\b(tell me|show me|explain)\b",
                r"\b(information|details|facts)\b",
            ],
            "greeting": [
                r"\b(hello|hi|hey|good morning|good afternoon|good evening)\b",
                r"\b(goodbye|bye|see you|farewell)\b",
            ],
            "appreciation": [
                r"\b(thank|thanks|appreciate|grateful)\b",
                r"\b(good job|well done|excellent)\b",
            ],
        }

        # Formality indicators
        self.formality_indicators = {
            "formal": [
                "please",
                "could you",
                "would you",
                "might I",
                "I would like",
                "excuse me",
                "pardon me",
                "sir",
                "madam",
            ],
            "informal": [
                "hey",
                "yo",
                "gonna",
                "wanna",
                "gotta",
                "yeah",
                "ok",
                "cool",
                "awesome",
                "dude",
                "buddy",
            ],
        }

    def analyze_text(self, text: str) -> NLPAnalysis:
        """Perform comprehensive NLP analysis on input text."""
        sentiment_score, sentiment_label, confidence = self._analyze_sentiment(text)
        entities = self._extract_entities(text)
        key_phrases = self._extract_key_phrases(text)
        intent = self._detect_intent(text)
        emotion = self._detect_emotion(text)
        formality_level = self._assess_formality(text)
        complexity_score = self._calculate_complexity(text)
        readability_score = self._calculate_readability(text)

        return NLPAnalysis(
            sentiment_score=sentiment_score,
            sentiment_label=sentiment_label,
            confidence=confidence,
            entities=entities,
            key_phrases=key_phrases,
            intent=intent,
            emotion=emotion,
            formality_level=formality_level,
            complexity_score=complexity_score,
            readability_score=readability_score,
        )

    def _analyze_sentiment(self, text: str) -> Tuple[float, str, float]:
        """Analyze sentiment using DistilBERT with TextBlob and lexicon fallbacks."""
        # Try DistilBERT first
        if self.sentiment_model:
            try:
                result = self.sentiment_model(text)[0]
                label = result["label"].lower()
                confidence = result["score"]
                sentiment_score = confidence if label == "positive" else -confidence
                final_label = label if label in ("positive", "negative") else "neutral"
                logger.debug(
                    "Sentiment: %s (%.3f, conf=%.3f)",
                    final_label,
                    sentiment_score,
                    confidence,
                )
                return sentiment_score, final_label, confidence
            except Exception as e:
                logger.warning("DistilBERT sentiment failed: %s", e)

        # Fallback to TextBlob
        if self._nlp_enhanced:
            try:
                from textblob import TextBlob

                blob = TextBlob(text)
                polarity = blob.sentiment.polarity
                subjectivity = blob.sentiment.subjectivity
                if polarity > 0.1:
                    label = "positive"
                elif polarity < -0.1:
                    label = "negative"
                else:
                    label = "neutral"
                return polarity, label, subjectivity
            except Exception as e:
                logger.warning("TextBlob analysis failed: %s", e)

        return self._lexicon_sentiment(text)

    def _lexicon_sentiment(self, text: str) -> Tuple[float, str, float]:
        """Lexicon-based sentiment analysis fallback."""
        positive_words = {
            "good",
            "great",
            "excellent",
            "amazing",
            "wonderful",
            "fantastic",
            "happy",
            "joy",
            "love",
            "like",
            "awesome",
            "perfect",
            "best",
        }
        negative_words = {
            "bad",
            "terrible",
            "awful",
            "horrible",
            "hate",
            "dislike",
            "sad",
            "angry",
            "upset",
            "disappointed",
            "frustrated",
            "worst",
        }

        words = text.lower().split()
        if not words:
            return 0.0, "neutral", 0.5

        positive_count = sum(1 for word in words if word in positive_words)
        negative_count = sum(1 for word in words if word in negative_words)
        total = positive_count + negative_count

        if total == 0:
            return 0.0, "neutral", 0.5

        score = (positive_count - negative_count) / len(words)
        confidence = min(total / len(words), 1.0)
        label = "positive" if score > 0 else ("negative" if score < 0 else "neutral")
        return score, label, confidence

    def _extract_entities(self, text: str) -> List[Dict[str, Any]]:
        """Extract named entities from text."""
        if self.nlp:
            try:
                doc = self.nlp(text)
                return [
                    {
                        "text": ent.text,
                        "label": ent.label_,
                        "start": ent.start_char,
                        "end": ent.end_char,
                        "description": spacy.explain(ent.label_) or ent.label_,
                    }
                    for ent in doc.ents
                ]
            except Exception as e:
                logger.warning("spaCy entity extraction failed: %s", e)

        return self._pattern_entity_extraction(text)

    def _pattern_entity_extraction(self, text: str) -> List[Dict[str, Any]]:
        """Pattern-based entity extraction fallback."""
        entities = []
        patterns = [
            (r"\b(today|tomorrow|yesterday)\b", "TIME"),
            (r"\b(\d{1,2}:\d{2})\b", "TIME"),
            (r"\b(morning|afternoon|evening|night)\b", "TIME"),
            (r"\b(Monday|Tuesday|Wednesday|Thursday|Friday|Saturday|Sunday)\b", "DATE"),
            (r"\b(arms?|legs?|chest|back|shoulders?|muscles?)\b", "BODY_PART"),
        ]
        for pattern, label in patterns:
            for match in re.finditer(pattern, text, re.IGNORECASE):
                entities.append(
                    {
                        "text": match.group(),
                        "label": label,
                        "start": match.start(),
                        "end": match.end(),
                        "description": label.replace("_", " ").title(),
                    }
                )
        return entities

    def _extract_key_phrases(self, text: str) -> List[str]:
        """Extract key phrases from text."""
        if self.nlp:
            try:
                doc = self.nlp(text)
                phrases = []
                for chunk in doc.noun_chunks:
                    if len(chunk.text.split()) > 1:
                        phrases.append(chunk.text)
                for token in doc:
                    if token.pos_ == "VERB":
                        parts = [token.text]
                        for child in token.children:
                            if child.dep_ in ["dobj", "attr", "prep"]:
                                parts.append(child.text)
                        if len(parts) > 1:
                            phrases.append(" ".join(parts))
                return list(set(phrases))
            except Exception as e:
                logger.warning("Key phrase extraction failed: %s", e)

        # Fallback: simple n-grams
        words = text.lower().split()
        phrases = []
        for i in range(len(words) - 1):
            phrases.append(" ".join(words[i : i + 2]))
            if i < len(words) - 2:
                phrases.append(" ".join(words[i : i + 3]))
        return phrases

    def _detect_intent(self, text: str) -> str:
        """Detect the intent/purpose of the text."""
        text_lower = text.lower()
        intent_scores = defaultdict(int)
        for intent, patterns in self.intent_patterns.items():
            for pattern in patterns:
                intent_scores[intent] += len(re.findall(pattern, text_lower))
        if not intent_scores:
            return "statement"
        return max(intent_scores.items(), key=lambda x: x[1])[0]

    def _detect_emotion(self, text: str) -> str:
        """Detect fine-grained emotions using DistilRoBERTa with lexicon fallback."""
        if self.emotion_model:
            try:
                result = self.emotion_model(text)[0]
                emotion = result["label"].lower()
                logger.debug("Emotion: %s (conf=%.3f)", emotion, result["score"])
                mapping = {
                    "joy": "joy",
                    "sadness": "sadness",
                    "anger": "anger",
                    "fear": "fear",
                    "surprise": "surprise",
                    "disgust": "disgust",
                    "neutral": "neutral",
                }
                return mapping.get(emotion, emotion)
            except Exception as e:
                logger.warning("Emotion detection failed: %s", e)

        # Lexicon fallback
        words = text.lower().split()
        scores = defaultdict(int)
        for emotion, emotion_words in self.emotion_lexicon.items():
            for word in words:
                if word in emotion_words:
                    scores[emotion] += 1
        if not scores:
            return "neutral"
        return max(scores.items(), key=lambda x: x[1])[0]

    def _assess_formality(self, text: str) -> str:
        """Assess the formality level of the text."""
        text_lower = text.lower()
        formal = sum(1 for w in self.formality_indicators["formal"] if w in text_lower)
        informal = sum(
            1 for w in self.formality_indicators["informal"] if w in text_lower
        )
        if formal > informal:
            return "formal"
        elif informal > formal:
            return "informal"
        return "neutral"

    def _calculate_complexity(self, text: str) -> float:
        """Calculate text complexity score (0-1)."""
        words = text.split()
        sentences = re.split(r"[.!?]+", text)
        if not words:
            return 0.0
        avg_word_len = sum(len(w) for w in words) / len(words)
        avg_sent_len = len(words) / max(len(sentences), 1)
        return min((avg_word_len / 10.0 + avg_sent_len / 20.0) / 2, 1.0)

    def _calculate_readability(self, text: str) -> float:
        """Calculate readability score using Flesch Reading Ease formula."""
        words = text.split()
        sentences = re.split(r"[.!?]+", text)
        syllables = self._count_syllables(text)
        if not words or not sentences:
            return 0.5
        avg_sent_len = len(words) / len(sentences)
        avg_syl_per_word = syllables / len(words)
        flesch = 206.835 - (1.015 * avg_sent_len) - (84.6 * avg_syl_per_word)
        return max(0, min(100, flesch)) / 100

    def _count_syllables(self, text: str) -> int:
        """Count syllables in text (approximate)."""
        vowels = "aeiouy"
        total_syllables = 0

        for word in text.lower().split():
            word = re.sub(r"[^a-z]", "", word)
            if not word:
                continue
            word_syllables = 0
            prev_was_vowel = False
            for char in word:
                is_vowel = char in vowels
                if is_vowel and not prev_was_vowel:
                    word_syllables += 1
                prev_was_vowel = is_vowel
            if word.endswith("e") and word_syllables > 1:
                word_syllables -= 1
            total_syllables += max(word_syllables, 1)

        return total_syllables

    def generate_context_aware_suggestions(
        self, analysis: NLPAnalysis, available_signs: List[str]
    ) -> List[str]:
        """Generate context-aware suggestions based on NLP analysis."""
        suggestions = []

        if analysis.intent == "greeting":
            suggestions.extend(
                s
                for s in available_signs
                if s in ["hello", "goodbye", "thank", "please"]
            )
        elif analysis.intent == "instruction":
            suggestions.extend(
                s for s in available_signs if s in ["do", "try", "start", "go", "come"]
            )
        elif analysis.intent == "request":
            suggestions.extend(
                s for s in available_signs if s in ["help", "please", "need", "want"]
            )

        if analysis.emotion == "joy":
            suggestions.extend(
                s for s in available_signs if s in ["happy", "good", "excellent"]
            )
        elif analysis.emotion in ("sadness", "anger"):
            suggestions.extend(
                s for s in available_signs if s in ["sad", "angry", "help"]
            )

        for entity in analysis.entities:
            if entity["label"] == "TIME":
                suggestions.extend(
                    s for s in available_signs if s in ["today", "tomorrow", "time"]
                )
            elif entity["label"] == "BODY_PART":
                suggestions.extend(
                    s
                    for s in available_signs
                    if s in ["arms", "legs", "chest", "muscle"]
                )

        return list(set(suggestions))
