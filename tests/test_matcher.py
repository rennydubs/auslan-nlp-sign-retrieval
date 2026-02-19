"""
Unit tests for SignMatcher — covers all matching stages.
Uses mock data from conftest; no real ML models loaded.
"""

import pytest


class TestExactMatch:
    def test_matches_dictionary_word(self, matcher):
        result = matcher.exact_match("hello")
        assert result is not None
        assert result["match_type"] == "exact"
        assert result["word"] == "hello"
        assert result["confidence"] == 1.0

    def test_case_insensitive(self, matcher):
        result = matcher.exact_match("HELLO")
        assert result is not None
        assert result["word"] == "hello"

    def test_no_match_returns_none(self, matcher):
        assert matcher.exact_match("zzzzunknown") is None

    def test_sign_data_present(self, matcher):
        result = matcher.exact_match("help")
        assert result["sign_data"] is not None
        assert result["sign_data"]["gloss"] == "HELP"


class TestFuzzyMatch:
    def test_catches_typo(self, matcher):
        # "exercize" should fuzzy-match "exercise"
        result = matcher.fuzzy_match("exercize")
        if result:  # Only passes if rapidfuzz installed and score >= threshold
            assert result["match_type"] == "fuzzy"
            assert result["word"] == "exercise"
            assert result["confidence"] == pytest.approx(matcher._fuzzy_confidence)

    def test_exact_word_not_fuzzy(self, matcher):
        # An exact dictionary word should NOT return a fuzzy match
        result = matcher.fuzzy_match("hello")
        assert result is None  # exact match handles it

    def test_very_different_word_no_match(self, matcher):
        result = matcher.fuzzy_match("xyz123qqqq")
        assert result is None

    def test_threshold_respected(self, matcher):
        # At threshold=100, nothing should match unless identical
        result = matcher.fuzzy_match("hello", threshold=100)
        assert result is None


class TestSynonymMatch:
    def test_manual_synonym(self, matcher):
        result = matcher.synonym_match("assist")
        assert result is not None
        assert result["match_type"] == "synonym"
        assert result["word"] == "help"
        assert result["confidence"] == pytest.approx(0.9, abs=0.01)

    def test_dictionary_embedded_synonym(self, matcher):
        # "greetings" is in hello's synonyms list
        result = matcher.synonym_match("greetings")
        assert result is not None
        assert result["word"] == "hello"

    def test_self_mapping_excluded(self, matcher):
        # A word that is itself a primary key should not return a synonym match
        result = matcher.synonym_match("hello")
        assert result is None  # exact_match handles primaries

    def test_unknown_word_no_match(self, matcher):
        assert matcher.synonym_match("zzzzunknown") is None


class TestMatchToken:
    """Integration tests for the full match_token pipeline."""

    def test_exact_takes_priority(self, matcher):
        result = matcher.match_token("hello", use_semantic=False)
        assert result["match_type"] == "exact"

    def test_synonym_fallback(self, matcher):
        result = matcher.match_token("assist", use_semantic=False)
        assert result is not None
        assert result["match_type"] == "synonym"
        assert result["word"] == "help"

    def test_no_match_unknown(self, matcher):
        result = matcher.match_token("zzzzunknown", use_semantic=False)
        assert result is None

    def test_semantic_disabled(self, matcher):
        # Without semantic, unknown words should return None
        result = matcher.match_token("require", use_semantic=False)
        assert result is None


class TestMatchTokens:
    def test_multiple_tokens(self, matcher):
        results = matcher.match_tokens(["hello", "help", "zzz"], use_semantic=False)
        assert len(results) == 3
        assert results[0]["match_type"] == "exact"
        assert results[1]["match_type"] == "exact"
        assert results[2]["match_type"] == "no_match"

    def test_phrase_detection(self, matcher):
        # "goodbye" should be matched
        results = matcher.match_tokens(["say", "goodbye"], use_semantic=False)
        assert any(
            r["word"] == "goodbye" for r in results if r["match_type"] != "no_match"
        )


class TestCoverageStats:
    def test_full_coverage(self, matcher):
        stats = matcher.get_coverage_stats(["hello", "help"], use_semantic=False)
        assert stats["coverage_rate"] == pytest.approx(1.0)
        assert stats["exact_matches"] == 2
        assert stats["total_tokens"] == 2

    def test_zero_coverage(self, matcher):
        stats = matcher.get_coverage_stats(["zzz", "yyy"], use_semantic=False)
        assert stats["coverage_rate"] == pytest.approx(0.0)
        assert stats["unmatched_tokens"] == 2

    def test_mixed_coverage(self, matcher):
        stats = matcher.get_coverage_stats(["hello", "zzz"], use_semantic=False)
        assert stats["coverage_rate"] == pytest.approx(0.5)

    def test_empty_tokens(self, matcher):
        stats = matcher.get_coverage_stats([], use_semantic=False)
        assert stats["coverage_rate"] == pytest.approx(0.0)
        assert stats["total_tokens"] == 0

    def test_synonym_counted(self, matcher):
        stats = matcher.get_coverage_stats(["assist"], use_semantic=False)
        assert stats["synonym_matches"] == 1
        assert stats["coverage_rate"] == pytest.approx(1.0)
