"""
Unit tests for TextPreprocessor.
All tests run without spaCy (shared_spacy_model=None).
"""

import pytest

from src.preprocessing import TextPreprocessor


@pytest.fixture
def prep():
    return TextPreprocessor(shared_spacy_model=None)


class TestCleanText:
    def test_lowercases(self, prep):
        assert prep.clean_text("HELLO World") == "hello world"

    def test_expands_contractions(self, prep):
        assert "do not" in prep.clean_text("I don't want to go")
        assert "will not" in prep.clean_text("won't")
        assert "cannot" in prep.clean_text("can't")

    def test_removes_punctuation(self, prep):
        result = prep.clean_text("Hello, world! How are you?")
        assert "," not in result
        assert "!" not in result
        assert "?" not in result

    def test_collapses_whitespace(self, prep):
        assert prep.clean_text("hello   world") == "hello world"

    def test_empty_string(self, prep):
        assert prep.clean_text("") == ""

    def test_non_string(self, prep):
        assert prep.clean_text(None) == ""
        assert prep.clean_text(123) == ""


class TestTokenize:
    def test_basic_tokenization(self, prep):
        tokens = prep.tokenize("hello world")
        assert tokens == ["hello", "world"]

    def test_removes_single_chars(self, prep):
        tokens = prep.tokenize("I a b go")
        # 'a' and 'i' are allowed, 'b' is not
        assert "b" not in tokens
        assert "a" in tokens

    def test_handles_punctuation(self, prep):
        tokens = prep.tokenize("Hello, how are you?")
        assert "," not in tokens
        assert "?" not in tokens

    def test_empty_input(self, prep):
        assert prep.tokenize("") == []

    def test_contractions_expanded(self, prep):
        tokens = prep.tokenize("I don't know")
        assert "do" in tokens
        assert "not" in tokens


class TestRemoveStopWords:
    def test_removes_common_stops(self, prep):
        tokens = ["the", "big", "house", "is", "on", "the", "hill"]
        filtered = prep.remove_stop_words(tokens)
        assert "the" not in filtered
        assert "is" not in filtered
        assert "on" not in filtered
        assert "big" in filtered
        assert "house" in filtered

    def test_empty_list(self, prep):
        assert prep.remove_stop_words([]) == []


class TestStemTokens:
    def test_stems_words(self, prep):
        result = prep.stem_tokens(["running", "exercises", "helping"])
        assert result[0] in ("run", "runn")  # Porter varies
        assert "exercis" in result[1]
        assert "help" in result[2]


class TestPreprocess:
    def test_full_pipeline(self, prep):
        tokens = prep.preprocess("Hello, how are you today?")
        assert isinstance(tokens, list)
        assert len(tokens) > 0
        assert "hello" in tokens

    def test_remove_stops_flag(self, prep):
        tokens_with = prep.preprocess("I am going to the store", remove_stops=False)
        tokens_without = prep.preprocess("I am going to the store", remove_stops=True)
        assert len(tokens_without) <= len(tokens_with)
        assert "the" not in tokens_without

    def test_stemming_flag(self, prep):
        tokens = prep.preprocess("She was running exercises", use_stemming=True)
        # After stemming, "running" -> "run" family, "exercises" -> "exercis" family
        joined = " ".join(tokens)
        assert "run" in joined or "exercis" in joined

    def test_lemmatization_flag(self, prep):
        """Lemmatization flag should either apply lemmatization or fall back gracefully."""
        tokens_normal = prep.preprocess("running exercises", use_lemmatization=False)
        tokens_lemma = prep.preprocess("running exercises", use_lemmatization=True)
        # Both must be non-empty lists of strings
        assert isinstance(tokens_normal, list) and len(tokens_normal) > 0
        assert isinstance(tokens_lemma, list) and len(tokens_lemma) > 0
        # If spaCy is available, lemmatization may change tokens (run/exercise);
        # if not, they should be identical. Either outcome is valid.
        if prep._nlp is None:
            assert tokens_normal == tokens_lemma
        else:
            # lemmatized forms should still contain the root concepts
            joined = " ".join(tokens_lemma)
            assert "run" in joined or "exercise" in joined

    def test_batch(self, prep):
        texts = ["hello world", "goodbye friend"]
        result = prep.preprocess_batch(texts)
        assert len(result) == 2
        assert "hello" in result[0]
        assert "goodbye" in result[1]
