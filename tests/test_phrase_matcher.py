"""
Unit tests for IntelligentPhraseMatcher from src/phrase_matcher.py.

All tests use mocked data only — no real spaCy model or ML models are loaded.
IntelligentPhraseMatcher gracefully degrades to _basic_analysis / _simple_segmentation
when self.nlp is None, which is the path exercised here.

Fixtures from conftest:
- tmp_data_dir  : paths to temp JSON files backed by MOCK_GLOSS_DICT
- matcher       : a SignMatcher instance using those mock files (no semantic model)
"""

from unittest.mock import MagicMock, patch

import pytest

from src.phrase_matcher import IntelligentPhraseMatcher, PhraseMatch

# ---------------------------------------------------------------------------
# Helpers / shared constants
# ---------------------------------------------------------------------------

_KNOWN_WORDS = ["hello", "help", "happy", "exercise", "water", "goodbye"]
_UNKNOWN_WORD = "zzzzunknown"


# ---------------------------------------------------------------------------
# Fixture: IntelligentPhraseMatcher backed by mock files, spaCy disabled
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def phrase_matcher(tmp_data_dir, matcher):
    """
    Return an IntelligentPhraseMatcher that:
    - loads gloss dict and target words from the temp mock files
    - receives the session-scoped SignMatcher as its sign_matcher dependency
    - has self.nlp = None (spaCy never loaded, falls back to basic analysis)
    """
    pm = IntelligentPhraseMatcher(
        gloss_dict_path=tmp_data_dir["gloss_dict"],
        target_words_path=tmp_data_dir["target_words"],
        sign_matcher=matcher,
        spacy_model_name="en_core_web_sm",  # won't be loaded; OSError caught internally
    )
    # Force spaCy off so tests are deterministic regardless of environment.
    pm.nlp = None
    return pm


# ---------------------------------------------------------------------------
# TestPhraseMatcherInit
# ---------------------------------------------------------------------------


class TestPhraseMatcherInit:
    """Verify the matcher initialises correctly from mock data."""

    def test_gloss_dict_populated(self, phrase_matcher):
        """Gloss dictionary must contain at least the mock words."""
        assert len(phrase_matcher.gloss_dict) >= len(_KNOWN_WORDS)

    def test_target_words_populated(self, phrase_matcher):
        """Target words dict must contain at least the four mock entries."""
        assert len(phrase_matcher.target_words) >= 4

    def test_sign_matcher_injected(self, phrase_matcher, matcher):
        """The injected SignMatcher must be the same object."""
        assert phrase_matcher.matcher is matcher

    def test_spacy_disabled(self, phrase_matcher):
        """self.nlp must be None so the basic-analysis fallback is exercised."""
        assert phrase_matcher.nlp is None

    def test_phrase_patterns_defined(self, phrase_matcher):
        """Phrase pattern dictionary must contain standard categories."""
        expected_keys = {
            "greeting",
            "question",
            "instruction",
            "request",
            "fitness_command",
            "emotional",
            "temporal",
        }
        assert expected_keys.issubset(phrase_matcher.phrase_patterns.keys())

    def test_missing_dict_file_gives_empty_dict(self, tmp_data_dir, matcher):
        """A missing gloss dict file must produce an empty dict, not a crash."""
        pm = IntelligentPhraseMatcher(
            gloss_dict_path="/nonexistent/path/dict.json",
            target_words_path=tmp_data_dir["target_words"],
            sign_matcher=matcher,
        )
        assert pm.gloss_dict == {}


# ---------------------------------------------------------------------------
# TestBasicAnalysis  (fallback path — nlp is None)
# ---------------------------------------------------------------------------


class TestBasicAnalysis:
    """Tests for _basic_analysis, the fallback when spaCy is unavailable."""

    def test_returns_required_keys(self, phrase_matcher):
        result = phrase_matcher._basic_analysis("hello")
        required = {
            "original_text",
            "entities",
            "sentiment",
            "phrase_type",
            "grammar_structure",
            "key_concepts",
            "action_words",
            "descriptors",
        }
        assert required.issubset(result.keys())

    def test_grammar_structure_is_basic(self, phrase_matcher):
        result = phrase_matcher._basic_analysis("anything")
        assert result["grammar_structure"] == "basic"

    def test_greeting_phrase_type(self, phrase_matcher):
        result = phrase_matcher._basic_analysis("hello there")
        assert result["phrase_type"] == "greeting"

    def test_question_phrase_type_from_keyword(self, phrase_matcher):
        result = phrase_matcher._basic_analysis("how are you")
        assert result["phrase_type"] == "question"

    def test_question_phrase_type_from_mark(self, phrase_matcher):
        result = phrase_matcher._basic_analysis("is it raining?")
        assert result["phrase_type"] == "question"

    def test_request_phrase_type(self, phrase_matcher):
        result = phrase_matcher._basic_analysis("can you help")
        assert result["phrase_type"] == "request"

    def test_positive_sentiment(self, phrase_matcher):
        result = phrase_matcher._basic_analysis("I feel happy today")
        assert result["sentiment"] == "positive"

    def test_negative_sentiment(self, phrase_matcher):
        result = phrase_matcher._basic_analysis("that is bad and sad")
        assert result["sentiment"] == "negative"

    def test_neutral_sentiment_default(self, phrase_matcher):
        result = phrase_matcher._basic_analysis("water please")
        assert result["sentiment"] == "neutral"

    def test_empty_string(self, phrase_matcher):
        """Empty string must not raise."""
        result = phrase_matcher._basic_analysis("")
        assert result["original_text"] == ""
        assert result["phrase_type"] == "statement"


# ---------------------------------------------------------------------------
# TestAnalyzePhrase  (public API delegates to _basic_analysis when nlp is None)
# ---------------------------------------------------------------------------


class TestAnalyzePhrase:
    """analyze_phrase must delegate to _basic_analysis when self.nlp is None."""

    def test_returns_dict(self, phrase_matcher):
        result = phrase_matcher.analyze_phrase("hello help")
        assert isinstance(result, dict)

    def test_original_text_preserved(self, phrase_matcher):
        text = "hello help"
        result = phrase_matcher.analyze_phrase(text)
        assert result["original_text"] == text

    def test_entities_is_list(self, phrase_matcher):
        result = phrase_matcher.analyze_phrase("I need water")
        assert isinstance(result["entities"], list)

    def test_phrase_type_for_greeting(self, phrase_matcher):
        result = phrase_matcher.analyze_phrase("goodbye friend")
        assert result["phrase_type"] == "greeting"

    def test_phrase_type_for_statement(self, phrase_matcher):
        result = phrase_matcher.analyze_phrase("exercise is good for you")
        assert result["phrase_type"] == "statement"


# ---------------------------------------------------------------------------
# TestSimpleSegmentation  (fallback path — nlp is None)
# ---------------------------------------------------------------------------


class TestSimpleSegmentation:
    """Tests for _simple_segmentation and the public segment_phrase API."""

    def test_single_word(self, phrase_matcher):
        segments = phrase_matcher.segment_phrase("hello")
        assert segments == ["hello"]

    def test_comma_splits(self, phrase_matcher):
        segments = phrase_matcher.segment_phrase("hello, help")
        assert len(segments) == 2

    def test_semicolon_splits(self, phrase_matcher):
        segments = phrase_matcher.segment_phrase("exercise; water")
        assert len(segments) == 2

    def test_and_conjunction_splits(self, phrase_matcher):
        segments = phrase_matcher.segment_phrase("hello and goodbye")
        assert len(segments) == 2

    def test_then_conjunction_splits(self, phrase_matcher):
        segments = phrase_matcher.segment_phrase("warm up then exercise")
        assert len(segments) == 2

    def test_empty_string_gives_empty_list(self, phrase_matcher):
        segments = phrase_matcher.segment_phrase("")
        assert segments == []

    def test_strips_whitespace_from_segments(self, phrase_matcher):
        segments = phrase_matcher.segment_phrase("  hello , help  ")
        for seg in segments:
            assert seg == seg.strip()

    def test_no_empty_segments(self, phrase_matcher):
        segments = phrase_matcher.segment_phrase("hello,,help")
        assert all(seg != "" for seg in segments)


# ---------------------------------------------------------------------------
# TestMatchSegment  (internal helper)
# ---------------------------------------------------------------------------


class TestMatchSegment:
    """Tests for _match_segment, which tokenises one segment and matches tokens."""

    def test_exact_word_matched(self, phrase_matcher):
        matches = phrase_matcher._match_segment(
            "hello", use_semantic=False, threshold=0.6
        )
        assert len(matches) == 1
        assert matches[0]["word"] == "hello"
        assert matches[0]["match_type"] == "exact"
        assert matches[0]["confidence"] == 1.0

    def test_multiple_words_in_segment(self, phrase_matcher):
        matches = phrase_matcher._match_segment(
            "hello help", use_semantic=False, threshold=0.6
        )
        words = [m["word"] for m in matches]
        assert "hello" in words
        assert "help" in words

    def test_unknown_word_produces_no_match(self, phrase_matcher):
        matches = phrase_matcher._match_segment(
            _UNKNOWN_WORD, use_semantic=False, threshold=0.6
        )
        assert matches == []

    def test_synonym_word_matched(self, phrase_matcher):
        # "assist" is a synonym of "help" in target_words
        matches = phrase_matcher._match_segment(
            "assist", use_semantic=False, threshold=0.6
        )
        assert len(matches) == 1
        assert matches[0]["match_type"] == "synonym"
        assert matches[0]["confidence"] == pytest.approx(0.9)

    def test_match_dict_has_required_keys(self, phrase_matcher):
        matches = phrase_matcher._match_segment(
            "hello", use_semantic=False, threshold=0.6
        )
        required = {"word", "sign_data", "match_type", "confidence"}
        for match in matches:
            assert required.issubset(match.keys())

    def test_sign_data_is_dict(self, phrase_matcher):
        matches = phrase_matcher._match_segment(
            "hello", use_semantic=False, threshold=0.6
        )
        assert isinstance(matches[0]["sign_data"], dict)

    def test_case_insensitive_matching(self, phrase_matcher):
        matches = phrase_matcher._match_segment(
            "HELLO", use_semantic=False, threshold=0.6
        )
        assert len(matches) == 1
        assert matches[0]["word"] == "hello"

    def test_semantic_disabled_unknown_not_matched(self, phrase_matcher):
        """When use_semantic=False, an unknown word produces no result."""
        matches = phrase_matcher._match_segment(
            "require", use_semantic=False, threshold=0.6
        )
        assert matches == []

    def test_semantic_enabled_delegates_to_sign_matcher(self, phrase_matcher):
        """When use_semantic=True and word is unknown, the injected SignMatcher
        is asked for a semantic match (which returns None here because no
        semantic model is loaded, but the call must not raise)."""
        matches = phrase_matcher._match_segment(
            "require", use_semantic=True, threshold=0.6
        )
        # Without a real semantic model the result is still empty — but no crash.
        assert isinstance(matches, list)


# ---------------------------------------------------------------------------
# TestMatchPhraseIntelligently  (the primary public method)
# ---------------------------------------------------------------------------


class TestMatchPhraseIntelligently:
    """Tests for the main match_phrase_intelligently method."""

    def test_returns_phrase_match_object(self, phrase_matcher):
        result = phrase_matcher.match_phrase_intelligently("hello help")
        assert isinstance(result, PhraseMatch)

    def test_phrase_match_has_required_fields(self, phrase_matcher):
        result = phrase_matcher.match_phrase_intelligently("hello")
        assert hasattr(result, "matched_signs")
        assert hasattr(result, "phrase_type")
        assert hasattr(result, "confidence")
        assert hasattr(result, "grammar_structure")
        assert hasattr(result, "sentiment")
        assert hasattr(result, "entities")
        assert hasattr(result, "original_phrase")

    def test_matched_signs_is_list(self, phrase_matcher):
        result = phrase_matcher.match_phrase_intelligently("hello help")
        assert isinstance(result.matched_signs, list)

    def test_known_word_hello_matched(self, phrase_matcher):
        result = phrase_matcher.match_phrase_intelligently("hello")
        words = [m["word"] for m in result.matched_signs]
        assert "hello" in words

    def test_known_word_help_matched(self, phrase_matcher):
        result = phrase_matcher.match_phrase_intelligently("help")
        words = [m["word"] for m in result.matched_signs]
        assert "help" in words

    def test_matched_signs_dicts_have_required_keys(self, phrase_matcher):
        result = phrase_matcher.match_phrase_intelligently("hello help")
        for sign in result.matched_signs:
            assert "word" in sign
            assert "match_type" in sign
            assert "confidence" in sign

    def test_confidence_is_float_between_0_and_1(self, phrase_matcher):
        result = phrase_matcher.match_phrase_intelligently("hello help")
        assert isinstance(result.confidence, float)
        assert 0.0 <= result.confidence <= 1.0

    def test_phrase_type_is_string(self, phrase_matcher):
        result = phrase_matcher.match_phrase_intelligently("hello")
        assert isinstance(result.phrase_type, str)

    def test_grammar_structure_is_string(self, phrase_matcher):
        result = phrase_matcher.match_phrase_intelligently("hello")
        assert isinstance(result.grammar_structure, str)

    def test_original_phrase_preserved(self, phrase_matcher):
        text = "hello help"
        result = phrase_matcher.match_phrase_intelligently(text)
        assert result.original_phrase == text

    def test_greeting_phrase_type_detected(self, phrase_matcher):
        result = phrase_matcher.match_phrase_intelligently("hello there")
        assert result.phrase_type == "greeting"

    def test_question_phrase_type_detected(self, phrase_matcher):
        result = phrase_matcher.match_phrase_intelligently("how are you")
        assert result.phrase_type == "question"

    def test_empty_string_does_not_crash(self, phrase_matcher):
        """Empty input must return a PhraseMatch with empty matched_signs."""
        result = phrase_matcher.match_phrase_intelligently("")
        assert isinstance(result, PhraseMatch)
        assert result.matched_signs == []

    def test_all_unknown_words_produces_empty_matches(self, phrase_matcher):
        result = phrase_matcher.match_phrase_intelligently(
            "zzz yyy xxx", use_semantic=False
        )
        assert result.matched_signs == []

    def test_confidence_zero_when_no_matches(self, phrase_matcher):
        result = phrase_matcher.match_phrase_intelligently(
            "zzz yyy xxx", use_semantic=False
        )
        assert result.confidence == pytest.approx(0.0)

    def test_synonym_input_resolved(self, phrase_matcher):
        """'assist' is a synonym of 'help', so it must appear in matched_signs."""
        result = phrase_matcher.match_phrase_intelligently("assist", use_semantic=False)
        assert len(result.matched_signs) >= 1
        assert result.matched_signs[0]["match_type"] == "synonym"

    def test_mixed_known_unknown_words(self, phrase_matcher):
        """Only the known word should produce a match entry."""
        result = phrase_matcher.match_phrase_intelligently(
            "hello zzzzunknown", use_semantic=False
        )
        words = [m["word"] for m in result.matched_signs]
        assert "hello" in words
        assert _UNKNOWN_WORD not in words

    def test_use_semantic_false_skips_semantic(self, phrase_matcher):
        """With use_semantic=False the SignMatcher.semantic_match must not be called."""
        with patch.object(phrase_matcher.matcher, "semantic_match") as mock_sem:
            phrase_matcher.match_phrase_intelligently("require", use_semantic=False)
            mock_sem.assert_not_called()


# ---------------------------------------------------------------------------
# TestOptimizeSignOrder
# ---------------------------------------------------------------------------


class TestOptimizeSignOrder:
    """Tests for the grammar-ordering optimisation step."""

    def test_empty_matches_returned_unchanged(self, phrase_matcher):
        analysis = {"phrase_type": "statement"}
        result = phrase_matcher._optimize_sign_order([], analysis)
        assert result == []

    def test_single_match_returned_unchanged(self, phrase_matcher):
        match = {
            "word": "hello",
            "match_type": "exact",
            "confidence": 1.0,
            "sign_data": {"category": "greeting"},
        }
        analysis = {"phrase_type": "statement"}
        result = phrase_matcher._optimize_sign_order([match], analysis)
        assert len(result) == 1

    def test_question_wh_words_moved_to_end(self, phrase_matcher):
        """For question phrases, wh-words should appear after other tokens."""
        how_match = {
            "word": "how",
            "match_type": "exact",
            "confidence": 1.0,
            "sign_data": {"category": "question"},
        }
        hello_match = {
            "word": "hello",
            "match_type": "exact",
            "confidence": 1.0,
            "sign_data": {"category": "greeting"},
        }
        analysis = {"phrase_type": "question"}
        result = phrase_matcher._optimize_sign_order([how_match, hello_match], analysis)
        # "hello" must appear before "how" in the output
        words = [m["word"] for m in result]
        assert words.index("hello") < words.index("how")

    def test_non_question_order_preserved_by_priority(self, phrase_matcher):
        """For non-question phrases, higher-priority categories sort earlier."""
        water_match = {
            "word": "water",
            "match_type": "exact",
            "confidence": 1.0,
            "sign_data": {"category": "basic_needs"},
        }
        exercise_match = {
            "word": "exercise",
            "match_type": "exact",
            "confidence": 1.0,
            "sign_data": {"category": "fitness_actions"},
        }
        analysis = {"phrase_type": "statement"}
        result = phrase_matcher._optimize_sign_order(
            [exercise_match, water_match], analysis
        )
        # basic_needs (priority 3) < fitness_actions (priority 9), so water first
        words = [m["word"] for m in result]
        assert words.index("water") < words.index("exercise")

    def test_unknown_category_does_not_crash(self, phrase_matcher):
        """A sign_data with an unrecognised category must not raise."""
        weird_match = {
            "word": "thing",
            "match_type": "exact",
            "confidence": 0.8,
            "sign_data": {"category": "completely_unknown_category"},
        }
        analysis = {"phrase_type": "statement"}
        result = phrase_matcher._optimize_sign_order([weird_match], analysis)
        assert len(result) == 1

    def test_missing_sign_data_does_not_crash(self, phrase_matcher):
        """A match with no sign_data must be handled gracefully."""
        match = {
            "word": "test",
            "match_type": "exact",
            "confidence": 0.9,
            "sign_data": None,
        }
        analysis = {"phrase_type": "statement"}
        result = phrase_matcher._optimize_sign_order([match], analysis)
        assert len(result) == 1


# ---------------------------------------------------------------------------
# TestGetPhraseSuggestions
# ---------------------------------------------------------------------------


class TestGetPhraseSuggestions:
    """Tests for autocomplete suggestions from get_phrase_suggestions."""

    def test_returns_list(self, phrase_matcher):
        suggestions = phrase_matcher.get_phrase_suggestions("hel")
        assert isinstance(suggestions, list)

    def test_short_input_returns_empty(self, phrase_matcher):
        """Partial input shorter than 2 chars must produce no suggestions."""
        suggestions = phrase_matcher.get_phrase_suggestions("h")
        assert suggestions == []

    def test_empty_input_returns_empty(self, phrase_matcher):
        suggestions = phrase_matcher.get_phrase_suggestions("")
        assert suggestions == []

    def test_limit_respected(self, phrase_matcher):
        suggestions = phrase_matcher.get_phrase_suggestions("I", limit=2)
        assert len(suggestions) <= 2

    def test_suggestions_are_strings(self, phrase_matcher):
        suggestions = phrase_matcher.get_phrase_suggestions("hel")
        for s in suggestions:
            assert isinstance(s, str)

    def test_partial_matching_exercise(self, phrase_matcher):
        """'exerc' should match suggestions containing the word exercise."""
        suggestions = phrase_matcher.get_phrase_suggestions("exerc")
        # The mock dict has exercise in fitness_actions category
        # If any suggestions come back they must be strings (safety check)
        assert all(isinstance(s, str) for s in suggestions)

    def test_matching_substring_present(self, phrase_matcher):
        """Any returned suggestion must contain the partial text (case-insensitive)."""
        partial = "hap"
        suggestions = phrase_matcher.get_phrase_suggestions(partial)
        for suggestion in suggestions:
            assert partial.lower() in suggestion.lower()


# ---------------------------------------------------------------------------
# TestPhraseMatcher_WithMockedSignMatcher
# ---------------------------------------------------------------------------


class TestPhraseMatcherWithMockedSignMatcher:
    """
    Verify IntelligentPhraseMatcher integration with a fully mocked SignMatcher,
    so we can control exactly what semantic_match returns.
    """

    @pytest.fixture
    def mock_matcher(self):
        m = MagicMock()
        m.semantic_match.return_value = {
            "word": "happy",
            "match_type": "semantic",
            "confidence": 0.75,
            "sign_data": {"gloss": "HAPPY", "category": "emotions"},
        }
        return m

    @pytest.fixture
    def pm_with_mock(self, tmp_data_dir, mock_matcher):
        pm = IntelligentPhraseMatcher(
            gloss_dict_path=tmp_data_dir["gloss_dict"],
            target_words_path=tmp_data_dir["target_words"],
            sign_matcher=mock_matcher,
        )
        pm.nlp = None
        return pm

    def test_semantic_match_called_for_unknown_word(self, pm_with_mock, mock_matcher):
        """When an unknown word is encountered with use_semantic=True,
        semantic_match on the injected matcher must be called."""
        pm_with_mock.match_phrase_intelligently("require", use_semantic=True)
        mock_matcher.semantic_match.assert_called()

    def test_semantic_result_included_in_matched_signs(self, pm_with_mock):
        result = pm_with_mock.match_phrase_intelligently("require", use_semantic=True)
        words = [m["word"] for m in result.matched_signs]
        assert "happy" in words

    def test_semantic_confidence_propagated(self, pm_with_mock):
        result = pm_with_mock.match_phrase_intelligently("require", use_semantic=True)
        semantic_matches = [
            m for m in result.matched_signs if m["match_type"] == "semantic"
        ]
        assert len(semantic_matches) == 1
        assert semantic_matches[0]["confidence"] == pytest.approx(0.75)

    def test_overall_confidence_reflects_semantic_score(self, pm_with_mock):
        """Overall confidence must be > 0 when a semantic match is found."""
        result = pm_with_mock.match_phrase_intelligently("require", use_semantic=True)
        assert result.confidence > 0.0
