"""
Enhanced NLP features for Auslan sign retrieval system.
Includes sentiment analysis, named entity recognition, and context understanding.
"""

import re
import json
from typing import List, Dict, Any, Tuple, Optional
from collections import defaultdict, Counter
from dataclasses import dataclass
import math

try:
    import spacy
    from textblob import TextBlob
    from sentence_transformers import SentenceTransformer
    from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification
    import torch
    NLP_ENHANCED = True

    # Try to load spaCy model
    try:
        nlp = spacy.load("en_core_web_sm")
    except OSError:
        print("Note: spaCy model not found. Some advanced features may be limited.")
        nlp = None

    # Load transformer models
    try:
        # DistilBERT for semantic similarity (same as existing matcher)
        semantic_model = SentenceTransformer('all-MiniLM-L6-v2')
        print("Loaded all-MiniLM-L6-v2 for semantic similarity")

        # DistilBERT for sentiment analysis (more accurate than TextBlob)
        sentiment_model = pipeline(
            "sentiment-analysis",
            model="distilbert-base-uncased-finetuned-sst-2-english",
            tokenizer="distilbert-base-uncased-finetuned-sst-2-english"
        )
        print("Loaded DistilBERT for sentiment analysis")

        # RoBERTa for emotion classification
        try:
            emotion_model = pipeline(
                "text-classification",
                model="j-hartmann/emotion-english-distilroberta-base",
                tokenizer="j-hartmann/emotion-english-distilroberta-base"
            )
            print("Loaded DistilRoBERTa for emotion classification")
            EMOTION_MODEL_AVAILABLE = True
        except Exception as e:
            print(f"Note: Advanced emotion model not available: {e}")
            emotion_model = None
            EMOTION_MODEL_AVAILABLE = False

    except ImportError as e:
        print(f"Note: Transformer models not available: {e}")
        semantic_model = None
        sentiment_model = None
        emotion_model = None
        EMOTION_MODEL_AVAILABLE = False

except ImportError:
    print("Note: Advanced NLP libraries not installed. Using basic NLP features.")
    NLP_ENHANCED = False
    nlp = None
    semantic_model = None
    sentiment_model = None
    emotion_model = None
    EMOTION_MODEL_AVAILABLE = False

@dataclass
class NLPAnalysis:
    """Container for comprehensive NLP analysis results."""
    sentiment_score: float  # -1 to 1
    sentiment_label: str    # positive, negative, neutral
    confidence: float       # 0 to 1
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
    """

    def __init__(self):
        """Initialize the NLP processor with models and dictionaries."""
        self.nlp = nlp
        self.semantic_model = semantic_model
        self.sentiment_model = sentiment_model
        self.emotion_model = emotion_model

        # Emotion lexicon for fine-grained emotion detection
        self.emotion_lexicon = {
            'joy': ['happy', 'joyful', 'elated', 'cheerful', 'pleased', 'content', 'glad'],
            'sadness': ['sad', 'depressed', 'unhappy', 'gloomy', 'melancholy', 'down'],
            'anger': ['angry', 'furious', 'mad', 'irritated', 'annoyed', 'rage'],
            'fear': ['afraid', 'scared', 'anxious', 'worried', 'nervous', 'frightened'],
            'surprise': ['surprised', 'amazed', 'astonished', 'shocked', 'stunned'],
            'disgust': ['disgusted', 'revolted', 'sick', 'nauseated'],
            'trust': ['trust', 'confident', 'secure', 'safe', 'assured'],
            'anticipation': ['excited', 'eager', 'hopeful', 'optimistic']
        }

        # Intent patterns for fitness/coaching context
        self.intent_patterns = {
            'instruction': [
                r'\b(do|perform|execute|practice|try)\b',
                r'\b(let\'s|let us)\b',
                r'\b(start|begin|commence)\b'
            ],
            'request': [
                r'\b(please|can you|could you|would you)\b',
                r'\b(need|want|require)\b',
                r'\b(help|assist|aid)\b'
            ],
            'question': [
                r'\b(how|what|where|when|why|who)\b',
                r'\?',
                r'\b(is|are|do|does|can|will)\b.*\?'
            ],
            'information': [
                r'\b(tell me|show me|explain)\b',
                r'\b(information|details|facts)\b'
            ],
            'greeting': [
                r'\b(hello|hi|hey|good morning|good afternoon|good evening)\b',
                r'\b(goodbye|bye|see you|farewell)\b'
            ],
            'appreciation': [
                r'\b(thank|thanks|appreciate|grateful)\b',
                r'\b(good job|well done|excellent)\b'
            ]
        }

        # Formality indicators
        self.formality_indicators = {
            'formal': [
                'please', 'could you', 'would you', 'might I', 'I would like',
                'excuse me', 'pardon me', 'sir', 'madam'
            ],
            'informal': [
                'hey', 'yo', 'gonna', 'wanna', 'gotta', 'yeah', 'ok', 'cool',
                'awesome', 'dude', 'buddy'
            ]
        }

    def analyze_text(self, text: str) -> NLPAnalysis:
        """
        Perform comprehensive NLP analysis on input text.

        Args:
            text (str): Input text to analyze

        Returns:
            NLPAnalysis: Comprehensive analysis results
        """
        # Basic sentiment analysis
        sentiment_score, sentiment_label, confidence = self._analyze_sentiment(text)

        # Named entity recognition
        entities = self._extract_entities(text)

        # Key phrase extraction
        key_phrases = self._extract_key_phrases(text)

        # Intent detection
        intent = self._detect_intent(text)

        # Emotion detection
        emotion = self._detect_emotion(text)

        # Formality level
        formality_level = self._assess_formality(text)

        # Complexity and readability
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
            readability_score=readability_score
        )

    def _analyze_sentiment(self, text: str) -> Tuple[float, str, float]:
        """
        Analyze sentiment using DistilBERT transformer model with TextBlob fallback.

        Returns:
            Tuple of (sentiment_score, sentiment_label, confidence)
        """
        # Try DistilBERT first (most accurate)
        if self.sentiment_model and NLP_ENHANCED:
            try:
                result = self.sentiment_model(text)[0]
                label = result['label'].lower()  # 'positive' or 'negative'
                confidence = result['score']

                # Convert to sentiment score (-1 to 1)
                if label == 'positive':
                    sentiment_score = confidence
                else:  # negative
                    sentiment_score = -confidence

                # Map labels for consistency
                if label in ['positive', 'negative']:
                    final_label = label
                else:
                    final_label = 'neutral'

                print(f"DistilBERT Sentiment: {final_label} (score: {sentiment_score:.3f}, confidence: {confidence:.3f})")
                return sentiment_score, final_label, confidence

            except Exception as e:
                print(f"DistilBERT sentiment analysis failed: {e}")

        # Fallback to TextBlob
        if NLP_ENHANCED:
            try:
                blob = TextBlob(text)
                polarity = blob.sentiment.polarity  # -1 to 1
                subjectivity = blob.sentiment.subjectivity  # 0 to 1

                # Convert to label
                if polarity > 0.1:
                    label = 'positive'
                elif polarity < -0.1:
                    label = 'negative'
                else:
                    label = 'neutral'

                # Use subjectivity as confidence indicator
                confidence = subjectivity

                return polarity, label, confidence

            except Exception as e:
                print(f"TextBlob analysis failed: {e}")

        # Final fallback to lexicon-based sentiment analysis
        return self._lexicon_sentiment(text)

    def _lexicon_sentiment(self, text: str) -> Tuple[float, str, float]:
        """Lexicon-based sentiment analysis fallback."""
        positive_words = [
            'good', 'great', 'excellent', 'amazing', 'wonderful', 'fantastic',
            'happy', 'joy', 'love', 'like', 'awesome', 'perfect', 'best'
        ]
        negative_words = [
            'bad', 'terrible', 'awful', 'horrible', 'hate', 'dislike',
            'sad', 'angry', 'upset', 'disappointed', 'frustrated', 'worst'
        ]

        words = text.lower().split()
        positive_count = sum(1 for word in words if word in positive_words)
        negative_count = sum(1 for word in words if word in negative_words)

        total_sentiment_words = positive_count + negative_count

        if total_sentiment_words == 0:
            return 0.0, 'neutral', 0.5

        sentiment_score = (positive_count - negative_count) / len(words)
        confidence = total_sentiment_words / len(words)

        if sentiment_score > 0:
            label = 'positive'
        elif sentiment_score < 0:
            label = 'negative'
        else:
            label = 'neutral'

        return sentiment_score, label, min(confidence, 1.0)

    def _extract_entities(self, text: str) -> List[Dict[str, Any]]:
        """Extract named entities from text."""
        entities = []

        if self.nlp:
            try:
                doc = self.nlp(text)
                for ent in doc.ents:
                    entities.append({
                        'text': ent.text,
                        'label': ent.label_,
                        'start': ent.start_char,
                        'end': ent.end_char,
                        'description': spacy.explain(ent.label_) if hasattr(spacy, 'explain') else ent.label_
                    })
                return entities
            except Exception as e:
                print(f"spaCy entity extraction failed: {e}")

        # Fallback to pattern-based entity extraction
        return self._pattern_entity_extraction(text)

    def _pattern_entity_extraction(self, text: str) -> List[Dict[str, Any]]:
        """Pattern-based entity extraction fallback."""
        entities = []

        # Time expressions
        time_patterns = [
            (r'\b(today|tomorrow|yesterday)\b', 'TIME'),
            (r'\b(\d{1,2}:\d{2})\b', 'TIME'),
            (r'\b(morning|afternoon|evening|night)\b', 'TIME'),
            (r'\b(Monday|Tuesday|Wednesday|Thursday|Friday|Saturday|Sunday)\b', 'DATE')
        ]

        # Body parts (relevant for fitness)
        body_patterns = [
            (r'\b(arms?|legs?|chest|back|shoulders?|muscles?)\b', 'BODY_PART')
        ]

        all_patterns = time_patterns + body_patterns

        for pattern, label in all_patterns:
            matches = re.finditer(pattern, text, re.IGNORECASE)
            for match in matches:
                entities.append({
                    'text': match.group(),
                    'label': label,
                    'start': match.start(),
                    'end': match.end(),
                    'description': label.replace('_', ' ').title()
                })

        return entities

    def _extract_key_phrases(self, text: str) -> List[str]:
        """Extract key phrases from text."""
        if self.nlp:
            try:
                doc = self.nlp(text)
                # Extract noun phrases and important verb phrases
                phrases = []

                # Noun phrases
                for chunk in doc.noun_chunks:
                    if len(chunk.text.split()) > 1:  # Multi-word phrases
                        phrases.append(chunk.text)

                # Important verb phrases (verb + object)
                for token in doc:
                    if token.pos_ == 'VERB':
                        # Get verb and its direct objects/complements
                        verb_phrase_parts = [token.text]
                        for child in token.children:
                            if child.dep_ in ['dobj', 'attr', 'prep']:
                                verb_phrase_parts.append(child.text)

                        if len(verb_phrase_parts) > 1:
                            phrases.append(' '.join(verb_phrase_parts))

                return list(set(phrases))  # Remove duplicates

            except Exception as e:
                print(f"Key phrase extraction failed: {e}")

        # Fallback to simple n-gram extraction
        return self._simple_phrase_extraction(text)

    def _simple_phrase_extraction(self, text: str) -> List[str]:
        """Simple phrase extraction fallback."""
        words = text.lower().split()
        phrases = []

        # Extract 2-grams and 3-grams
        for i in range(len(words) - 1):
            bigram = ' '.join(words[i:i+2])
            phrases.append(bigram)

            if i < len(words) - 2:
                trigram = ' '.join(words[i:i+3])
                phrases.append(trigram)

        return phrases

    def _detect_intent(self, text: str) -> str:
        """Detect the intent/purpose of the text."""
        text_lower = text.lower()

        intent_scores = defaultdict(int)

        for intent, patterns in self.intent_patterns.items():
            for pattern in patterns:
                matches = len(re.findall(pattern, text_lower))
                intent_scores[intent] += matches

        if not intent_scores:
            return 'statement'  # Default intent

        # Return the intent with the highest score
        return max(intent_scores.items(), key=lambda x: x[1])[0]

    def _detect_emotion(self, text: str) -> str:
        """Detect fine-grained emotions using DistilRoBERTa transformer model."""
        # Try DistilRoBERTa emotion model first (most accurate)
        if self.emotion_model and EMOTION_MODEL_AVAILABLE:
            try:
                result = self.emotion_model(text)[0]
                emotion = result['label'].lower()
                confidence = result['score']

                print(f"DistilRoBERTa Emotion: {emotion} (confidence: {confidence:.3f})")

                # Map model outputs to our emotion categories
                emotion_mapping = {
                    'joy': 'joy',
                    'sadness': 'sadness',
                    'anger': 'anger',
                    'fear': 'fear',
                    'surprise': 'surprise',
                    'disgust': 'disgust',
                    'love': 'joy',  # Map love to joy
                    'optimism': 'anticipation',
                    'pessimism': 'sadness'
                }

                return emotion_mapping.get(emotion, emotion)

            except Exception as e:
                print(f"DistilRoBERTa emotion detection failed: {e}")

        # Fallback to lexicon-based emotion detection
        text_lower = text.lower()
        words = text_lower.split()

        emotion_scores = defaultdict(int)

        for emotion, emotion_words in self.emotion_lexicon.items():
            for word in words:
                if word in emotion_words:
                    emotion_scores[emotion] += 1

        if not emotion_scores:
            return 'neutral'

        # Return the emotion with the highest score
        return max(emotion_scores.items(), key=lambda x: x[1])[0]

    def _assess_formality(self, text: str) -> str:
        """Assess the formality level of the text."""
        text_lower = text.lower()

        formal_count = sum(1 for word in self.formality_indicators['formal']
                          if word in text_lower)
        informal_count = sum(1 for word in self.formality_indicators['informal']
                            if word in text_lower)

        # Consider punctuation and structure
        if '?' in text or '!' in text:
            informal_count += 1

        if formal_count > informal_count:
            return 'formal'
        elif informal_count > formal_count:
            return 'informal'
        else:
            return 'neutral'

    def _calculate_complexity(self, text: str) -> float:
        """Calculate text complexity score (0-1)."""
        words = text.split()
        sentences = re.split(r'[.!?]+', text)

        if not words:
            return 0.0

        # Average word length
        avg_word_length = sum(len(word) for word in words) / len(words)

        # Average sentence length
        avg_sentence_length = len(words) / max(len(sentences), 1)

        # Complexity based on word and sentence length
        complexity = (avg_word_length / 10.0 + avg_sentence_length / 20.0) / 2

        return min(complexity, 1.0)

    def _calculate_readability(self, text: str) -> float:
        """Calculate readability score using Flesch Reading Ease formula."""
        words = text.split()
        sentences = re.split(r'[.!?]+', text)
        syllables = self._count_syllables(text)

        if not words or not sentences:
            return 0.5  # Default medium readability

        avg_sentence_length = len(words) / len(sentences)
        avg_syllables_per_word = syllables / len(words)

        # Simplified Flesch Reading Ease (scaled to 0-1)
        flesch_score = 206.835 - (1.015 * avg_sentence_length) - (84.6 * avg_syllables_per_word)

        # Normalize to 0-1 scale (higher = more readable)
        normalized_score = max(0, min(100, flesch_score)) / 100

        return normalized_score

    def _count_syllables(self, text: str) -> int:
        """Count syllables in text (approximate)."""
        vowels = 'aeiouy'
        syllable_count = 0

        for word in text.lower().split():
            word = re.sub(r'[^a-z]', '', word)
            if not word:
                continue

            # Count vowel groups
            prev_was_vowel = False
            for char in word:
                is_vowel = char in vowels
                if is_vowel and not prev_was_vowel:
                    syllable_count += 1
                prev_was_vowel = is_vowel

            # Adjust for silent 'e'
            if word.endswith('e') and syllable_count > 1:
                syllable_count -= 1

            # Every word has at least one syllable
            if syllable_count == 0:
                syllable_count = 1

        return syllable_count

    def generate_context_aware_suggestions(self, analysis: NLPAnalysis,
                                         available_signs: List[str]) -> List[str]:
        """
        Generate context-aware suggestions based on NLP analysis.

        Args:
            analysis: NLP analysis results
            available_signs: List of available sign words

        Returns:
            List of contextually relevant suggestions
        """
        suggestions = []

        # Intent-based suggestions
        if analysis.intent == 'greeting':
            greetings = [sign for sign in available_signs
                        if sign in ['hello', 'goodbye', 'thank', 'please']]
            suggestions.extend(greetings)

        elif analysis.intent == 'instruction':
            actions = [sign for sign in available_signs
                      if sign in ['do', 'try', 'start', 'go', 'come']]
            suggestions.extend(actions)

        elif analysis.intent == 'request':
            helpers = [sign for sign in available_signs
                      if sign in ['help', 'please', 'need', 'want']]
            suggestions.extend(helpers)

        # Emotion-based suggestions
        if analysis.emotion in ['joy', 'happiness']:
            positive_signs = [sign for sign in available_signs
                            if sign in ['happy', 'good', 'excellent']]
            suggestions.extend(positive_signs)

        elif analysis.emotion in ['sadness', 'anger']:
            emotion_signs = [sign for sign in available_signs
                           if sign in ['sad', 'angry', 'help']]
            suggestions.extend(emotion_signs)

        # Entity-based suggestions
        for entity in analysis.entities:
            if entity['label'] == 'TIME':
                time_signs = [sign for sign in available_signs
                            if sign in ['today', 'tomorrow', 'time']]
                suggestions.extend(time_signs)

            elif entity['label'] == 'BODY_PART':
                body_signs = [sign for sign in available_signs
                            if sign in ['arms', 'legs', 'chest', 'muscle']]
                suggestions.extend(body_signs)

        return list(set(suggestions))  # Remove duplicates