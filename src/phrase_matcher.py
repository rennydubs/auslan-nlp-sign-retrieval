"""
Advanced phrase matching system with NLP features for Auslan sign retrieval.
Implements intelligent phrase segmentation, context analysis, and sign ordering.
"""

import json
import logging
import re
from dataclasses import dataclass
from typing import Any, Dict, List

logger = logging.getLogger(__name__)

try:
    import spacy

    SPACY_AVAILABLE = True
except ImportError:
    SPACY_AVAILABLE = False
    logger.info("spaCy not available. Advanced NLP features disabled.")


@dataclass
class PhraseMatch:
    """Represents a matched phrase with context information."""

    original_phrase: str
    matched_signs: List[Dict[str, Any]]
    confidence: float
    phrase_type: str
    sentiment: str
    entities: List[Dict[str, Any]]
    grammar_structure: str


class IntelligentPhraseMatcher:
    """
    Advanced phrase matching system that understands context, grammar, and semantics.
    """

    def __init__(
        self,
        gloss_dict_path: str,
        target_words_path: str,
        sign_matcher=None,
        spacy_model_name: str = "en_core_web_sm",
    ):
        """Initialize the phrase matcher with dictionaries and NLP models."""
        self.gloss_dict = self._load_dictionary(gloss_dict_path)
        self.target_words = self._load_target_words(target_words_path)
        self.matcher = sign_matcher  # SignMatcher instance for semantic matching

        # Load spaCy lazily
        self.nlp = None
        if SPACY_AVAILABLE:
            try:
                self.nlp = spacy.load(spacy_model_name)
                logger.info("Phrase matcher loaded spaCy model: %s", spacy_model_name)
            except OSError:
                logger.warning(
                    "spaCy model '%s' not found for phrase matcher.", spacy_model_name
                )

        # Define common phrase patterns for sign language
        self.phrase_patterns = {
            "greeting": [
                "hello",
                "hi",
                "good morning",
                "good afternoon",
                "goodbye",
                "bye",
            ],
            "question": ["how", "what", "where", "when", "why", "who"],
            "instruction": ["please", "do", "can you", "let's", "try to"],
            "request": ["need", "want", "help", "please"],
            "fitness_command": [
                "warm up",
                "cool down",
                "lift",
                "exercise",
                "stretch",
                "breathe",
            ],
            "emotional": ["happy", "sad", "angry", "excited", "tired"],
            "temporal": ["today", "tomorrow", "now", "later", "time"],
        }

        # Common sign language grammar structures
        self.grammar_rules = {
            "topic_comment": ["noun verb", "noun adjective"],
            "question_formation": ["wh-word noun verb", "verb noun"],
            "imperative": ["verb", "verb noun"],
            "description": ["noun adjective", "adjective noun"],
        }

    def _load_dictionary(self, path: str) -> Dict[str, Any]:
        """Load the Auslan dictionary."""
        try:
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)
        except FileNotFoundError:
            logger.warning("Dictionary file not found at %s", path)
            return {}

    def _load_target_words(self, path: str) -> Dict[str, List[str]]:
        """Load target words with synonyms."""
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
                word_dict = {}
                for item in data["target_words"]:
                    word_dict[item["word"]] = item["synonyms"]
                return word_dict
        except FileNotFoundError:
            logger.warning("Target words file not found at %s", path)
            return {}

    def analyze_phrase(self, text: str) -> Dict[str, Any]:
        """
        Perform comprehensive NLP analysis on the input phrase.

        Args:
            text: Input text to analyze

        Returns:
            Dict containing analysis results
        """
        analysis = {
            "original_text": text,
            "entities": [],
            "sentiment": "neutral",
            "phrase_type": "statement",
            "grammar_structure": "unknown",
            "key_concepts": [],
            "action_words": [],
            "descriptors": [],
        }

        if not self.nlp:
            return self._basic_analysis(text)

        doc = self.nlp(text)

        # Named Entity Recognition
        analysis["entities"] = [
            {
                "text": ent.text,
                "label": ent.label_,
                "description": spacy.explain(ent.label_),
            }
            for ent in doc.ents
        ]

        # Sentiment Analysis (basic lexicon)
        sentiment_indicators = {
            "positive": [
                "happy",
                "good",
                "great",
                "excellent",
                "love",
                "like",
                "wonderful",
            ],
            "negative": ["sad", "bad", "terrible", "hate", "angry", "upset", "painful"],
        }

        text_lower = text.lower()
        for sentiment, indicators in sentiment_indicators.items():
            if any(indicator in text_lower for indicator in indicators):
                analysis["sentiment"] = sentiment
                break

        # Phrase type detection
        if any(
            word in text_lower
            for word in ["?", "how", "what", "where", "when", "why", "who"]
        ):
            analysis["phrase_type"] = "question"
        elif any(word in text_lower for word in ["please", "can you", "could you"]):
            analysis["phrase_type"] = "request"
        elif any(word in text_lower for word in ["do", "let's", "try"]):
            analysis["phrase_type"] = "instruction"
        elif any(word in text_lower for word in self.phrase_patterns["greeting"]):
            analysis["phrase_type"] = "greeting"

        # Extract grammatical information
        pos_sequence = [token.pos_ for token in doc]
        analysis["grammar_structure"] = " ".join(pos_sequence)

        # Extract key concepts, actions, and descriptors
        for token in doc:
            if token.pos_ == "VERB":
                analysis["action_words"].append(token.lemma_)
            elif token.pos_ in ["NOUN", "PROPN"]:
                analysis["key_concepts"].append(token.lemma_)
            elif token.pos_ == "ADJ":
                analysis["descriptors"].append(token.lemma_)

        return analysis

    def _basic_analysis(self, text: str) -> Dict[str, Any]:
        """Basic analysis without spaCy for fallback."""
        analysis = {
            "original_text": text,
            "entities": [],
            "sentiment": "neutral",
            "phrase_type": "statement",
            "grammar_structure": "basic",
            "key_concepts": [],
            "action_words": [],
            "descriptors": [],
        }

        text_lower = text.lower()

        if "?" in text or any(word in text_lower for word in ["how", "what", "where"]):
            analysis["phrase_type"] = "question"
        elif any(word in text_lower for word in ["please", "can you"]):
            analysis["phrase_type"] = "request"
        elif any(word in text_lower for word in self.phrase_patterns["greeting"]):
            analysis["phrase_type"] = "greeting"

        if any(word in text_lower for word in ["happy", "good", "great"]):
            analysis["sentiment"] = "positive"
        elif any(word in text_lower for word in ["sad", "bad", "angry"]):
            analysis["sentiment"] = "negative"

        return analysis

    def segment_phrase(self, text: str) -> List[str]:
        """
        Intelligently segment a phrase into meaningful chunks for sign language.

        Args:
            text: Input text to segment

        Returns:
            List of phrase segments
        """
        if not self.nlp:
            return self._simple_segmentation(text)

        doc = self.nlp(text)
        segments = []
        current_segment = []

        for token in doc:
            if (
                token.pos_ in ["VERB"]
                and current_segment
                and any(t.pos_ in ["NOUN", "PROPN"] for t in current_segment)
            ):
                segments.append(" ".join([t.text for t in current_segment]))
                current_segment = [token]
            elif token.pos_ in ["PUNCT"] and token.text in [".", "!", "?"]:
                if current_segment:
                    segments.append(" ".join([t.text for t in current_segment]))
                    current_segment = []
            else:
                current_segment.append(token)

        if current_segment:
            segments.append(" ".join([t.text for t in current_segment]))

        return [seg.strip() for seg in segments if seg.strip()]

    def _simple_segmentation(self, text: str) -> List[str]:
        """Simple segmentation fallback."""
        segments = re.split(r"[.!?;,]|\s+and\s+|\s+then\s+", text)
        return [seg.strip() for seg in segments if seg.strip()]

    def match_phrase_intelligently(
        self, text: str, use_semantic: bool = True, threshold: float = 0.6
    ) -> PhraseMatch:
        """
        Perform intelligent phrase matching with context awareness.

        Args:
            text: Input phrase to match
            use_semantic: Whether to use semantic matching
            threshold: Semantic similarity threshold

        Returns:
            PhraseMatch object with comprehensive results
        """
        analysis = self.analyze_phrase(text)
        segments = self.segment_phrase(text)

        all_matches = []
        total_confidence = 0

        for segment in segments:
            segment_matches = self._match_segment(segment, use_semantic, threshold)
            all_matches.extend(segment_matches)

            if segment_matches:
                segment_confidence = sum(
                    m["confidence"] for m in segment_matches
                ) / len(segment_matches)
                total_confidence += segment_confidence

        overall_confidence = total_confidence / len(segments) if segments else 0

        optimized_matches = self._optimize_sign_order(all_matches, analysis)

        return PhraseMatch(
            original_phrase=text,
            matched_signs=optimized_matches,
            confidence=overall_confidence,
            phrase_type=analysis["phrase_type"],
            sentiment=analysis["sentiment"],
            entities=analysis["entities"],
            grammar_structure=analysis["grammar_structure"],
        )

    def _match_segment(
        self, segment: str, use_semantic: bool, threshold: float
    ) -> List[Dict[str, Any]]:
        """Match a single segment to signs."""
        tokens = segment.lower().split()
        matches = []

        for token in tokens:
            # Try exact match first
            if token in self.gloss_dict:
                matches.append(
                    {
                        "word": token,
                        "sign_data": self.gloss_dict[token],
                        "match_type": "exact",
                        "confidence": 1.0,
                    }
                )
            else:
                # Try synonym matching via target_words
                synonym_found = False
                for word, synonyms in self.target_words.items():
                    if token in synonyms and word in self.gloss_dict:
                        matches.append(
                            {
                                "word": token,
                                "sign_data": self.gloss_dict[word],
                                "match_type": "synonym",
                                "confidence": 0.9,
                            }
                        )
                        synonym_found = True
                        break

                # Try semantic matching via injected SignMatcher
                if not synonym_found and use_semantic and self.matcher:
                    semantic_match = self.matcher.semantic_match(token, threshold)
                    if semantic_match:
                        matches.append(semantic_match)

        return matches

    def _optimize_sign_order(
        self, matches: List[Dict[str, Any]], analysis: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """
        Optimize the order of signs based on ASL/Auslan grammar rules.
        Generally: Time -> Topic -> Comment -> Question marker

        Args:
            matches: List of matched signs
            analysis: Phrase analysis results

        Returns:
            Optimized list of signs
        """
        if not matches:
            return matches

        priority_order = {
            "time": 0,
            "temporal": 0,
            "greeting": 1,
            "fitness_core": 2,
            "basic_needs": 3,
            "actions": 4,
            "emotions": 5,
            "descriptive": 6,
            "fitness_anatomy": 7,
            "fitness_cardio": 7,
            "fitness_health": 7,
            "places": 8,
            "social": 8,
            "communication": 8,
            "fitness_actions": 9,
            "fitness_equipment": 10,
        }

        def sort_key(match):
            sign_data = match.get("sign_data") or {}
            category = sign_data.get("category", "unknown")
            priority = priority_order.get(category, 10)
            confidence = match.get("confidence", 0)
            return (priority, -confidence)

        sorted_matches = sorted(matches, key=sort_key)

        # For questions, move wh-words to the end (Auslan structure)
        if analysis["phrase_type"] == "question":
            question_words = []
            other_words = []

            for match in sorted_matches:
                word = match["word"].lower()
                if word in ["how", "what", "where", "when", "why", "who"]:
                    question_words.append(match)
                else:
                    other_words.append(match)

            sorted_matches = other_words + question_words

        return sorted_matches

    def get_phrase_suggestions(self, partial_text: str, limit: int = 5) -> List[str]:
        """
        Get intelligent phrase suggestions based on partial input.

        Args:
            partial_text: Partial text input
            limit: Maximum number of suggestions

        Returns:
            List of suggested phrases
        """
        suggestions = []
        text_lower = partial_text.lower().strip()

        if len(text_lower) < 2:
            return suggestions

        templates = [
            "I need {word}",
            "Let's {verb}",
            "How to {verb}",
            "Please help me {verb}",
            "I want to {verb}",
            "Time to {verb}",
            "I feel {emotion}",
            "Can you {verb}",
            "Where is the {noun}",
            "Thank you for {noun}",
        ]

        available_words = list(self.gloss_dict.keys())

        for template in templates:
            if "{verb}" in template:
                verbs = [
                    word
                    for word in available_words
                    if self.gloss_dict[word].get("category")
                    in ["actions", "fitness_actions"]
                ]
                for verb in verbs[:3]:
                    suggestion = template.replace("{verb}", verb)
                    if text_lower in suggestion.lower():
                        suggestions.append(suggestion)

            elif "{emotion}" in template:
                emotions = [
                    word
                    for word in available_words
                    if self.gloss_dict[word].get("category") == "emotions"
                ]
                for emotion in emotions[:2]:
                    suggestion = template.replace("{emotion}", emotion)
                    if text_lower in suggestion.lower():
                        suggestions.append(suggestion)

            elif "{noun}" in template:
                nouns = [
                    word
                    for word in available_words
                    if self.gloss_dict[word].get("category")
                    in ["basic_needs", "places"]
                ]
                for noun in nouns[:2]:
                    suggestion = template.replace("{noun}", noun)
                    if text_lower in suggestion.lower():
                        suggestions.append(suggestion)

        return suggestions[:limit]
