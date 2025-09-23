"""
Advanced phrase matching system with NLP features for Auslan sign retrieval.
Implements intelligent phrase segmentation, context analysis, and sign ordering.
"""

import re
import json
from typing import List, Dict, Any, Tuple, Optional
from collections import defaultdict
import spacy
from dataclasses import dataclass

try:
    import spacy
    from spacy import displacy
    NLP_AVAILABLE = True
    # Download the model if not available
    try:
        nlp = spacy.load("en_core_web_sm")
    except OSError:
        print("Downloading spaCy English model...")
        spacy.cli.download("en_core_web_sm")
        nlp = spacy.load("en_core_web_sm")
except ImportError:
    NLP_AVAILABLE = False
    print("Warning: spaCy not available. Advanced NLP features disabled.")
    nlp = None

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

    def __init__(self, gloss_dict_path: str, target_words_path: str):
        """Initialize the phrase matcher with dictionaries and NLP models."""
        self.gloss_dict = self._load_dictionary(gloss_dict_path)
        self.target_words = self._load_target_words(target_words_path)
        self.nlp = nlp if NLP_AVAILABLE else None

        # Define common phrase patterns for sign language
        self.phrase_patterns = {
            'greeting': ['hello', 'hi', 'good morning', 'good afternoon', 'goodbye', 'bye'],
            'question': ['how', 'what', 'where', 'when', 'why', 'who'],
            'instruction': ['please', 'do', 'can you', 'let\'s', 'try to'],
            'request': ['need', 'want', 'help', 'please'],
            'fitness_command': ['warm up', 'cool down', 'lift', 'exercise', 'stretch', 'breathe'],
            'emotional': ['happy', 'sad', 'angry', 'excited', 'tired'],
            'temporal': ['today', 'tomorrow', 'now', 'later', 'time']
        }

        # Common sign language grammar structures
        self.grammar_rules = {
            'topic_comment': ['noun verb', 'noun adjective'],
            'question_formation': ['wh-word noun verb', 'verb noun'],
            'imperative': ['verb', 'verb noun'],
            'description': ['noun adjective', 'adjective noun']
        }

    def _load_dictionary(self, path: str) -> Dict[str, Any]:
        """Load the Auslan dictionary."""
        try:
            with open(path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except FileNotFoundError:
            print(f"Warning: Dictionary file not found at {path}")
            return {}

    def _load_target_words(self, path: str) -> Dict[str, List[str]]:
        """Load target words with synonyms."""
        try:
            with open(path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                word_dict = {}
                for item in data['target_words']:
                    word_dict[item['word']] = item['synonyms']
                return word_dict
        except FileNotFoundError:
            print(f"Warning: Target words file not found at {path}")
            return {}

    def analyze_phrase(self, text: str) -> Dict[str, Any]:
        """
        Perform comprehensive NLP analysis on the input phrase.

        Args:
            text (str): Input text to analyze

        Returns:
            Dict containing analysis results
        """
        analysis = {
            'original_text': text,
            'entities': [],
            'sentiment': 'neutral',
            'phrase_type': 'statement',
            'grammar_structure': 'unknown',
            'key_concepts': [],
            'action_words': [],
            'descriptors': []
        }

        if not self.nlp:
            # Fallback to basic analysis without spaCy
            return self._basic_analysis(text)

        doc = self.nlp(text)

        # Named Entity Recognition
        analysis['entities'] = [
            {
                'text': ent.text,
                'label': ent.label_,
                'description': spacy.explain(ent.label_)
            }
            for ent in doc.ents
        ]

        # Sentiment Analysis (basic)
        sentiment_indicators = {
            'positive': ['happy', 'good', 'great', 'excellent', 'love', 'like', 'wonderful'],
            'negative': ['sad', 'bad', 'terrible', 'hate', 'angry', 'upset', 'painful'],
            'neutral': []
        }

        text_lower = text.lower()
        for sentiment, indicators in sentiment_indicators.items():
            if any(indicator in text_lower for indicator in indicators):
                analysis['sentiment'] = sentiment
                break

        # Phrase type detection
        if any(word in text_lower for word in ['?', 'how', 'what', 'where', 'when', 'why', 'who']):
            analysis['phrase_type'] = 'question'
        elif any(word in text_lower for word in ['please', 'can you', 'could you']):
            analysis['phrase_type'] = 'request'
        elif any(word in text_lower for word in ['do', 'let\'s', 'try']):
            analysis['phrase_type'] = 'instruction'
        elif any(word in text_lower for word in self.phrase_patterns['greeting']):
            analysis['phrase_type'] = 'greeting'

        # Extract grammatical information
        pos_sequence = [token.pos_ for token in doc]
        analysis['grammar_structure'] = ' '.join(pos_sequence)

        # Extract key concepts, actions, and descriptors
        for token in doc:
            if token.pos_ == 'VERB':
                analysis['action_words'].append(token.lemma_)
            elif token.pos_ in ['NOUN', 'PROPN']:
                analysis['key_concepts'].append(token.lemma_)
            elif token.pos_ == 'ADJ':
                analysis['descriptors'].append(token.lemma_)

        return analysis

    def _basic_analysis(self, text: str) -> Dict[str, Any]:
        """Basic analysis without spaCy for fallback."""
        analysis = {
            'original_text': text,
            'entities': [],
            'sentiment': 'neutral',
            'phrase_type': 'statement',
            'grammar_structure': 'basic',
            'key_concepts': [],
            'action_words': [],
            'descriptors': []
        }

        text_lower = text.lower()

        # Basic phrase type detection
        if '?' in text or any(word in text_lower for word in ['how', 'what', 'where']):
            analysis['phrase_type'] = 'question'
        elif any(word in text_lower for word in ['please', 'can you']):
            analysis['phrase_type'] = 'request'
        elif any(word in text_lower for word in self.phrase_patterns['greeting']):
            analysis['phrase_type'] = 'greeting'

        # Basic sentiment
        if any(word in text_lower for word in ['happy', 'good', 'great']):
            analysis['sentiment'] = 'positive'
        elif any(word in text_lower for word in ['sad', 'bad', 'angry']):
            analysis['sentiment'] = 'negative'

        return analysis

    def segment_phrase(self, text: str) -> List[str]:
        """
        Intelligently segment a phrase into meaningful chunks for sign language.

        Args:
            text (str): Input text to segment

        Returns:
            List of phrase segments
        """
        if not self.nlp:
            # Simple fallback segmentation
            return self._simple_segmentation(text)

        doc = self.nlp(text)
        segments = []
        current_segment = []

        for token in doc:
            # Start a new segment for certain conditions
            if (token.pos_ in ['VERB'] and current_segment and
                any(t.pos_ in ['NOUN', 'PROPN'] for t in current_segment)):
                segments.append(' '.join([t.text for t in current_segment]))
                current_segment = [token]
            elif token.pos_ in ['PUNCT'] and token.text in ['.', '!', '?']:
                if current_segment:
                    segments.append(' '.join([t.text for t in current_segment]))
                    current_segment = []
            else:
                current_segment.append(token)

        if current_segment:
            segments.append(' '.join([t.text for t in current_segment]))

        return [seg.strip() for seg in segments if seg.strip()]

    def _simple_segmentation(self, text: str) -> List[str]:
        """Simple segmentation fallback."""
        # Split by punctuation and common phrase boundaries
        segments = re.split(r'[.!?;,]|\s+and\s+|\s+then\s+', text)
        return [seg.strip() for seg in segments if seg.strip()]

    def match_phrase_intelligently(self, text: str, use_semantic: bool = True,
                                 threshold: float = 0.6) -> PhraseMatch:
        """
        Perform intelligent phrase matching with context awareness.

        Args:
            text (str): Input phrase to match
            use_semantic (bool): Whether to use semantic matching
            threshold (float): Semantic similarity threshold

        Returns:
            PhraseMatch object with comprehensive results
        """
        # Analyze the phrase first
        analysis = self.analyze_phrase(text)

        # Segment the phrase
        segments = self.segment_phrase(text)

        all_matches = []
        total_confidence = 0

        for segment in segments:
            # For each segment, find the best matches
            segment_matches = self._match_segment(segment, use_semantic, threshold)
            all_matches.extend(segment_matches)

            # Calculate confidence based on match quality
            if segment_matches:
                segment_confidence = sum(match['confidence'] for match in segment_matches) / len(segment_matches)
                total_confidence += segment_confidence

        # Calculate overall confidence
        overall_confidence = total_confidence / len(segments) if segments else 0

        # Optimize sign order for sign language grammar
        optimized_matches = self._optimize_sign_order(all_matches, analysis)

        return PhraseMatch(
            original_phrase=text,
            matched_signs=optimized_matches,
            confidence=overall_confidence,
            phrase_type=analysis['phrase_type'],
            sentiment=analysis['sentiment'],
            entities=analysis['entities'],
            grammar_structure=analysis['grammar_structure']
        )

    def _match_segment(self, segment: str, use_semantic: bool, threshold: float) -> List[Dict[str, Any]]:
        """Match a single segment to signs."""
        # Direct matching without creating new SignMatcher instance
        tokens = segment.lower().split()
        matches = []

        for token in tokens:
            # Try exact match first
            if token in self.gloss_dict:
                match = {
                    'word': token,
                    'sign_data': self.gloss_dict[token],
                    'match_type': 'exact',
                    'confidence': 1.0
                }
                matches.append(match)
            else:
                # Try synonym matching
                synonym_found = False
                for word, synonyms in self.target_words.items():
                    if token in synonyms and word in self.gloss_dict:
                        match = {
                            'word': token,
                            'sign_data': self.gloss_dict[word],
                            'match_type': 'synonym',
                            'confidence': 0.9
                        }
                        matches.append(match)
                        synonym_found = True
                        break

                # Try semantic matching if enabled and no synonym found
                if not synonym_found and use_semantic and hasattr(self, 'matcher') and self.matcher:
                    semantic_match = self.matcher.semantic_match(token, threshold)
                    if semantic_match:
                        matches.append(semantic_match)

        return matches

    def _optimize_sign_order(self, matches: List[Dict[str, Any]], analysis: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Optimize the order of signs based on ASL/Auslan grammar rules.

        Args:
            matches: List of matched signs
            analysis: Phrase analysis results

        Returns:
            Optimized list of signs
        """
        if not matches:
            return matches

        # Basic ASL/Auslan grammar: Topic-Comment structure
        # Generally: Time -> Topic -> Comment -> Question marker

        priority_order = {
            'temporal': 0,    # time words first
            'greeting': 1,    # greetings early
            'fitness_core': 2,  # main fitness concepts
            'basic_needs': 3,   # basic needs
            'actions': 4,       # action verbs
            'emotions': 5,      # emotional states
            'descriptive': 6,   # descriptors
            'fitness_anatomy': 7,  # body parts
            'places': 8,        # locations
            'fitness_equipment': 9  # equipment last
        }

        # Sort matches by category priority and confidence
        def sort_key(match):
            category = match['sign_data'].get('category', 'unknown')
            priority = priority_order.get(category, 10)
            confidence = match.get('confidence', 0)
            return (priority, -confidence)  # negative confidence for descending order

        sorted_matches = sorted(matches, key=sort_key)

        # Additional grammar-based reordering for questions
        if analysis['phrase_type'] == 'question':
            # Move question words to the end (ASL structure)
            question_words = []
            other_words = []

            for match in sorted_matches:
                word = match['word'].lower()
                if word in ['how', 'what', 'where', 'when', 'why', 'who']:
                    question_words.append(match)
                else:
                    other_words.append(match)

            sorted_matches = other_words + question_words

        return sorted_matches

    def get_phrase_suggestions(self, partial_text: str, limit: int = 5) -> List[str]:
        """
        Get intelligent phrase suggestions based on partial input.

        Args:
            partial_text (str): Partial text input
            limit (int): Maximum number of suggestions

        Returns:
            List of suggested phrases
        """
        suggestions = []
        text_lower = partial_text.lower().strip()

        if len(text_lower) < 2:
            return suggestions

        # Common phrase templates
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
            "Thank you for {noun}"
        ]

        # Context-aware suggestions based on available signs
        available_words = list(self.gloss_dict.keys())

        for template in templates:
            if '{verb}' in template:
                verbs = [word for word in available_words
                        if self.gloss_dict[word].get('category') in ['actions', 'fitness_actions']]
                for verb in verbs[:3]:
                    suggestion = template.replace('{verb}', verb)
                    if text_lower in suggestion.lower():
                        suggestions.append(suggestion)

            elif '{emotion}' in template:
                emotions = [word for word in available_words
                           if self.gloss_dict[word].get('category') == 'emotions']
                for emotion in emotions[:2]:
                    suggestion = template.replace('{emotion}', emotion)
                    if text_lower in suggestion.lower():
                        suggestions.append(suggestion)

            elif '{noun}' in template:
                nouns = [word for word in available_words
                        if self.gloss_dict[word].get('category') in ['basic_needs', 'places']]
                for noun in nouns[:2]:
                    suggestion = template.replace('{noun}', noun)
                    if text_lower in suggestion.lower():
                        suggestions.append(suggestion)

        return suggestions[:limit]