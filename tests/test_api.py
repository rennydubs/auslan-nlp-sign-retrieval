"""
Integration tests for the FastAPI endpoints defined in api.py.

Strategy
--------
- The global ``_sign_system`` singleton in api.py is replaced with a
  MagicMock before every test so that no real ML models or file I/O occur.
- httpx.AsyncClient with ASGITransport drives requests directly against the
  ASGI app without a real server.
- pytest-asyncio handles the async test functions.

Run with:
    pytest tests/test_api.py -v
"""

from unittest.mock import MagicMock, patch

import pytest
from httpx import ASGITransport, AsyncClient

# ---------------------------------------------------------------------------
# Mock data fixtures
# ---------------------------------------------------------------------------

MOCK_GLOSS_DICT = {
    "hello": {
        "gloss": "HELLO",
        "category": "greetings",
        "video_url": "/media/videos/hello.mp4",
        "description": "A greeting sign",
        "synonyms": ["hi", "greetings"],
    },
    "help": {
        "gloss": "HELP",
        "category": "basic_needs",
        "video_url": "/media/videos/help.mp4",
        "description": "Asking for assistance",
        "synonyms": ["assist", "aid"],
    },
    "happy": {
        "gloss": "HAPPY",
        "category": "emotions",
        "video_url": "/media/videos/happy.mp4",
        "description": "Feeling joy",
        "synonyms": ["glad", "joyful"],
    },
}

MOCK_PROCESS_RESPONSE = {
    "original_text": "hello help",
    "processed_tokens": ["hello", "help"],
    "total_tokens": 2,
    "successful_matches": [
        {
            "word": "hello",
            "match_type": "exact",
            "confidence": 1.0,
            "sign_data": {
                "gloss": "HELLO",
                "video_url": "/media/videos/hello.mp4",
                "description": "greeting",
                "category": "greetings",
                "synonyms": [],
            },
        }
    ],
    "failed_matches": [],
    "coverage_stats": {
        "coverage_rate": 1.0,
        "exact_matches": 2,
        "fuzzy_matches": 0,
        "synonym_matches": 0,
        "semantic_matches": 0,
        "llm_matches": 0,
        "unmatched_tokens": 0,
        "matched_tokens": 2,
        "total_tokens": 2,
        "match_breakdown": {
            "exact": 1.0,
            "fuzzy": 0,
            "synonym": 0,
            "semantic": 0,
            "llm": 0,
        },
    },
    "signs_found": 2,
    "nlp_analysis": {
        "sentiment": "neutral",
        "sentiment_score": 0.0,
        "emotion": "neutral",
        "intent": "statement",
        "entities": [],
        "key_phrases": [],
        "formality": "neutral",
        "complexity": 0.5,
        "readability": 0.5,
    },
}


# ---------------------------------------------------------------------------
# NLPAnalysis object mock (returned by nlp_processor.analyze_text)
# ---------------------------------------------------------------------------


def _make_nlp_analysis_mock():
    """Return a mock that looks like an NLPAnalysis dataclass."""
    nlp = MagicMock()
    nlp.sentiment_label = "neutral"
    nlp.sentiment_score = 0.0
    nlp.confidence = 0.9
    nlp.emotion = "neutral"
    nlp.intent = "statement"
    nlp.entities = []
    nlp.key_phrases = []
    nlp.formality_level = "neutral"
    nlp.complexity_score = 0.5
    nlp.readability_score = 0.5
    return nlp


# ---------------------------------------------------------------------------
# Core mock fixture: _sign_system
# ---------------------------------------------------------------------------


@pytest.fixture
def mock_sign_system():
    """
    A MagicMock that masquerades as an AuslanSignSystem instance.

    Attributes configured:
    - matcher.gloss_dict        — the mock dictionary
    - matcher.semantic_model    — None (semantic not available)
    - nlp_processor             — set up so analyze_text returns a mock analysis
    - nlp_processor.nlp         — None (spaCy not available)
    - nlp_processor.sentiment_model  — None
    - nlp_processor.emotion_model    — None
    - phrase_matcher            — a simple MagicMock (intelligent matching present)
    - process_input             — returns MOCK_PROCESS_RESPONSE
    - batch_evaluation          — returns a minimal evaluation dict
    """
    system = MagicMock()

    # Matcher sub-mock
    system.matcher = MagicMock()
    system.matcher.gloss_dict = MOCK_GLOSS_DICT
    system.matcher.semantic_model = None  # no semantic model

    # NLP processor sub-mock
    nlp_proc = MagicMock()
    nlp_proc.nlp = None
    nlp_proc.sentiment_model = None
    nlp_proc.emotion_model = None
    nlp_proc.analyze_text.return_value = _make_nlp_analysis_mock()
    system.nlp_processor = nlp_proc

    # Phrase matcher sub-mock
    system.phrase_matcher = MagicMock()
    system.phrase_matcher.get_phrase_suggestions.return_value = [
        "I feel happy",
        "hello help",
    ]

    # LLM processor
    system.llm_processor = None

    # Main process_input call
    system.process_input.return_value = MOCK_PROCESS_RESPONSE

    # Batch evaluation
    system.batch_evaluation.return_value = {
        "test_texts": ["hello"],
        "individual_results": [MOCK_PROCESS_RESPONSE],
        "average_coverage": 1.0,
        "total_tests": 1,
    }

    return system


# ---------------------------------------------------------------------------
# Async client fixture
# ---------------------------------------------------------------------------


@pytest.fixture
async def async_client(mock_sign_system):
    """
    Async httpx client wired to the FastAPI ASGI app.

    The global ``_sign_system`` module-level variable in api.py is patched
    with ``mock_sign_system`` for the duration of each test, so the lifespan
    startup code is bypassed.
    """
    import api

    with patch.object(api, "_sign_system", mock_sign_system):
        async with AsyncClient(
            transport=ASGITransport(app=api.app),
            base_url="http://test",
        ) as client:
            yield client


# ===========================================================================
# Tests: GET /api/health
# ===========================================================================


class TestHealthEndpoint:
    """GET /api/health must return 200 with status and version information."""

    @pytest.mark.asyncio
    async def test_returns_200(self, async_client):
        response = await async_client.get("/api/health")
        assert response.status_code == 200

    @pytest.mark.asyncio
    async def test_has_status_field(self, async_client):
        response = await async_client.get("/api/health")
        body = response.json()
        assert "status" in body

    @pytest.mark.asyncio
    async def test_status_is_ok(self, async_client):
        response = await async_client.get("/api/health")
        body = response.json()
        assert body["status"] == "ok"

    @pytest.mark.asyncio
    async def test_has_uptime_seconds(self, async_client):
        response = await async_client.get("/api/health")
        body = response.json()
        assert "uptime_seconds" in body
        assert isinstance(body["uptime_seconds"], (int, float))

    @pytest.mark.asyncio
    async def test_has_total_signs(self, async_client):
        response = await async_client.get("/api/health")
        body = response.json()
        assert "total_signs" in body
        assert body["total_signs"] == len(MOCK_GLOSS_DICT)

    @pytest.mark.asyncio
    async def test_content_type_is_json(self, async_client):
        response = await async_client.get("/api/health")
        assert "application/json" in response.headers["content-type"]


# ===========================================================================
# Tests: GET /api/models/status
# ===========================================================================


class TestModelsStatusEndpoint:
    """GET /api/models/status must return 200 with component availability flags."""

    @pytest.mark.asyncio
    async def test_returns_200(self, async_client):
        response = await async_client.get("/api/models/status")
        assert response.status_code == 200

    @pytest.mark.asyncio
    async def test_has_semantic_available_field(self, async_client):
        response = await async_client.get("/api/models/status")
        body = response.json()
        assert "semantic_model_available" in body

    @pytest.mark.asyncio
    async def test_semantic_available_is_false_without_model(self, async_client):
        """semantic_model is None on the mock, so this must be False."""
        response = await async_client.get("/api/models/status")
        body = response.json()
        assert body["semantic_model_available"] is False

    @pytest.mark.asyncio
    async def test_has_spacy_available_field(self, async_client):
        response = await async_client.get("/api/models/status")
        body = response.json()
        assert "spacy_available" in body

    @pytest.mark.asyncio
    async def test_has_intelligent_matching_available(self, async_client):
        response = await async_client.get("/api/models/status")
        body = response.json()
        assert "intelligent_matching_available" in body
        assert body["intelligent_matching_available"] is True

    @pytest.mark.asyncio
    async def test_has_total_signs(self, async_client):
        response = await async_client.get("/api/models/status")
        body = response.json()
        assert "total_signs" in body

    @pytest.mark.asyncio
    async def test_has_system_version(self, async_client):
        response = await async_client.get("/api/models/status")
        body = response.json()
        assert "system_version" in body

    @pytest.mark.asyncio
    async def test_has_semantic_model_name(self, async_client):
        response = await async_client.get("/api/models/status")
        body = response.json()
        assert "semantic_model_name" in body
        assert isinstance(body["semantic_model_name"], str)


# ===========================================================================
# Tests: POST /api/process
# ===========================================================================


class TestProcessEndpoint:
    """POST /api/process — main sign matching endpoint."""

    @pytest.mark.asyncio
    async def test_valid_request_returns_200(self, async_client):
        response = await async_client.post("/api/process", json={"text": "hello help"})
        assert response.status_code == 200

    @pytest.mark.asyncio
    async def test_response_has_successful_matches(self, async_client):
        response = await async_client.post("/api/process", json={"text": "hello help"})
        body = response.json()
        assert "successful_matches" in body

    @pytest.mark.asyncio
    async def test_response_has_coverage_stats(self, async_client):
        response = await async_client.post("/api/process", json={"text": "hello help"})
        body = response.json()
        assert "coverage_stats" in body

    @pytest.mark.asyncio
    async def test_response_has_nlp_analysis(self, async_client):
        response = await async_client.post("/api/process", json={"text": "hello help"})
        body = response.json()
        assert "nlp_analysis" in body

    @pytest.mark.asyncio
    async def test_response_has_signs_found(self, async_client):
        response = await async_client.post("/api/process", json={"text": "hello help"})
        body = response.json()
        assert "signs_found" in body
        assert body["signs_found"] == 2

    @pytest.mark.asyncio
    async def test_successful_matches_is_list(self, async_client):
        response = await async_client.post("/api/process", json={"text": "hello help"})
        body = response.json()
        assert isinstance(body["successful_matches"], list)

    @pytest.mark.asyncio
    async def test_process_input_called_with_text(self, async_client, mock_sign_system):
        """process_input on the mock system must be called with the submitted text."""
        await async_client.post("/api/process", json={"text": "hello help"})
        mock_sign_system.process_input.assert_called_once()
        call_args = mock_sign_system.process_input.call_args
        assert (
            call_args[0][0] == "hello help"
            or call_args[1].get("text") == "hello help"
            or call_args.args[0] == "hello help"
        )

    @pytest.mark.asyncio
    async def test_empty_text_returns_422(self, async_client):
        """An empty string violates the min_length=1 constraint and must 422."""
        response = await async_client.post("/api/process", json={"text": ""})
        assert response.status_code == 422

    @pytest.mark.asyncio
    async def test_missing_text_field_returns_422(self, async_client):
        """A request body with no 'text' key must fail validation."""
        response = await async_client.post("/api/process", json={})
        assert response.status_code == 422

    @pytest.mark.asyncio
    async def test_options_remove_stops_passed_through(
        self, async_client, mock_sign_system
    ):
        """ProcessOptions fields must be forwarded to process_input."""
        await async_client.post(
            "/api/process",
            json={"text": "hello", "options": {"remove_stops": True}},
        )
        call_kwargs = mock_sign_system.process_input.call_args[1]
        assert call_kwargs.get("remove_stops") is True

    @pytest.mark.asyncio
    async def test_response_body_matches_mock(self, async_client):
        """The JSON body must mirror the value returned by process_input."""
        response = await async_client.post("/api/process", json={"text": "hello help"})
        body = response.json()
        assert body["original_text"] == "hello help"
        assert body["total_tokens"] == 2

    @pytest.mark.asyncio
    async def test_processed_at_timestamp_present(self, async_client):
        """api.py appends processed_at after calling process_input."""
        response = await async_client.post("/api/process", json={"text": "hello help"})
        body = response.json()
        assert "processed_at" in body


# ===========================================================================
# Tests: POST /api/analyze
# ===========================================================================


class TestAnalyzeEndpoint:
    """POST /api/analyze — detailed NLP analysis only."""

    @pytest.mark.asyncio
    async def test_valid_request_returns_200(self, async_client):
        response = await async_client.post(
            "/api/analyze", json={"text": "I feel happy"}
        )
        assert response.status_code == 200

    @pytest.mark.asyncio
    async def test_response_has_sentiment(self, async_client):
        response = await async_client.post(
            "/api/analyze", json={"text": "I feel happy"}
        )
        body = response.json()
        assert "sentiment" in body

    @pytest.mark.asyncio
    async def test_response_has_emotion(self, async_client):
        response = await async_client.post(
            "/api/analyze", json={"text": "I feel happy"}
        )
        body = response.json()
        assert "emotion" in body

    @pytest.mark.asyncio
    async def test_response_has_intent(self, async_client):
        response = await async_client.post(
            "/api/analyze", json={"text": "I feel happy"}
        )
        body = response.json()
        assert "intent" in body

    @pytest.mark.asyncio
    async def test_response_has_entities(self, async_client):
        response = await async_client.post(
            "/api/analyze", json={"text": "I feel happy"}
        )
        body = response.json()
        assert "entities" in body
        assert isinstance(body["entities"], list)

    @pytest.mark.asyncio
    async def test_response_has_key_phrases(self, async_client):
        response = await async_client.post(
            "/api/analyze", json={"text": "I feel happy"}
        )
        body = response.json()
        assert "key_phrases" in body

    @pytest.mark.asyncio
    async def test_response_has_formality(self, async_client):
        response = await async_client.post(
            "/api/analyze", json={"text": "I feel happy"}
        )
        body = response.json()
        assert "formality" in body

    @pytest.mark.asyncio
    async def test_analyze_text_called(self, async_client, mock_sign_system):
        await async_client.post("/api/analyze", json={"text": "I feel happy"})
        mock_sign_system.nlp_processor.analyze_text.assert_called_once_with(
            "I feel happy"
        )

    @pytest.mark.asyncio
    async def test_empty_text_returns_422(self, async_client):
        response = await async_client.post("/api/analyze", json={"text": ""})
        assert response.status_code == 422

    @pytest.mark.asyncio
    async def test_nlp_unavailable_returns_503(self, mock_sign_system):
        """When nlp_processor is None the endpoint must return 503."""
        import api

        mock_sign_system.nlp_processor = None
        with patch.object(api, "_sign_system", mock_sign_system):
            async with AsyncClient(
                transport=ASGITransport(app=api.app), base_url="http://test"
            ) as client:
                response = await client.post("/api/analyze", json={"text": "hello"})
        assert response.status_code == 503


# ===========================================================================
# Tests: GET /api/dictionary
# ===========================================================================


class TestDictionaryEndpoint:
    """GET /api/dictionary — full sign dictionary listing."""

    @pytest.mark.asyncio
    async def test_returns_200(self, async_client):
        response = await async_client.get("/api/dictionary")
        assert response.status_code == 200

    @pytest.mark.asyncio
    async def test_has_total_entries(self, async_client):
        response = await async_client.get("/api/dictionary")
        body = response.json()
        assert "total_entries" in body

    @pytest.mark.asyncio
    async def test_total_entries_matches_mock(self, async_client):
        response = await async_client.get("/api/dictionary")
        body = response.json()
        assert body["total_entries"] == len(MOCK_GLOSS_DICT)

    @pytest.mark.asyncio
    async def test_has_categories(self, async_client):
        response = await async_client.get("/api/dictionary")
        body = response.json()
        assert "categories" in body
        assert isinstance(body["categories"], dict)

    @pytest.mark.asyncio
    async def test_categories_match_mock_data(self, async_client):
        response = await async_client.get("/api/dictionary")
        body = response.json()
        expected_cats = {"greetings", "basic_needs", "emotions"}
        assert expected_cats == set(body["categories"].keys())

    @pytest.mark.asyncio
    async def test_has_entries(self, async_client):
        response = await async_client.get("/api/dictionary")
        body = response.json()
        assert "entries" in body
        assert isinstance(body["entries"], dict)

    @pytest.mark.asyncio
    async def test_entries_contain_mock_words(self, async_client):
        response = await async_client.get("/api/dictionary")
        body = response.json()
        for word in MOCK_GLOSS_DICT:
            assert word in body["entries"]

    @pytest.mark.asyncio
    async def test_entry_has_gloss_field(self, async_client):
        response = await async_client.get("/api/dictionary")
        body = response.json()
        entry = body["entries"]["hello"]
        assert "gloss" in entry
        assert entry["gloss"] == "HELLO"

    @pytest.mark.asyncio
    async def test_entry_has_category_field(self, async_client):
        response = await async_client.get("/api/dictionary")
        body = response.json()
        entry = body["entries"]["hello"]
        assert "category" in entry

    @pytest.mark.asyncio
    async def test_entry_has_synonyms_field(self, async_client):
        response = await async_client.get("/api/dictionary")
        body = response.json()
        entry = body["entries"]["hello"]
        assert "synonyms" in entry
        assert isinstance(entry["synonyms"], list)


# ===========================================================================
# Tests: POST /api/suggestions
# ===========================================================================


class TestSuggestionsEndpoint:
    """POST /api/suggestions — phrase autocomplete."""

    @pytest.mark.asyncio
    async def test_returns_200(self, async_client):
        response = await async_client.post("/api/suggestions", json={"text": "hel"})
        assert response.status_code == 200

    @pytest.mark.asyncio
    async def test_has_suggestions_field(self, async_client):
        response = await async_client.post("/api/suggestions", json={"text": "hel"})
        body = response.json()
        assert "suggestions" in body

    @pytest.mark.asyncio
    async def test_suggestions_is_list(self, async_client):
        response = await async_client.post("/api/suggestions", json={"text": "hel"})
        body = response.json()
        assert isinstance(body["suggestions"], list)

    @pytest.mark.asyncio
    async def test_suggestions_contain_mock_values(self, async_client):
        response = await async_client.post("/api/suggestions", json={"text": "hel"})
        body = response.json()
        assert "I feel happy" in body["suggestions"]

    @pytest.mark.asyncio
    async def test_partial_text_echoed_back(self, async_client):
        response = await async_client.post("/api/suggestions", json={"text": "hel"})
        body = response.json()
        assert "partial_text" in body
        assert body["partial_text"] == "hel"

    @pytest.mark.asyncio
    async def test_get_phrase_suggestions_called(self, async_client, mock_sign_system):
        await async_client.post("/api/suggestions", json={"text": "hel"})
        mock_sign_system.phrase_matcher.get_phrase_suggestions.assert_called_once_with(
            "hel", limit=8
        )

    @pytest.mark.asyncio
    async def test_too_short_text_returns_422(self, async_client):
        """SuggestionsRequest has min_length=2, so a single-char body must 422."""
        response = await async_client.post("/api/suggestions", json={"text": "h"})
        assert response.status_code == 422

    @pytest.mark.asyncio
    async def test_phrase_matcher_none_returns_empty_list(self, mock_sign_system):
        """When phrase_matcher is None the endpoint must return an empty list."""
        import api

        mock_sign_system.phrase_matcher = None
        with patch.object(api, "_sign_system", mock_sign_system):
            async with AsyncClient(
                transport=ASGITransport(app=api.app), base_url="http://test"
            ) as client:
                response = await client.post("/api/suggestions", json={"text": "hel"})
        assert response.status_code == 200
        assert response.json()["suggestions"] == []


# ===========================================================================
# Tests: GET /media/videos/<filename>
# ===========================================================================


class TestVideoEndpoint:
    """GET /media/videos/<filename> — video file serving."""

    @pytest.mark.asyncio
    async def test_nonexistent_video_returns_404(self, async_client):
        response = await async_client.get("/media/videos/nonexistent.mp4")
        assert response.status_code == 404

    @pytest.mark.asyncio
    async def test_path_traversal_blocked(self, async_client):
        """Filenames with '..' must be rejected with 400."""
        response = await async_client.get("/media/videos/../secret.txt")
        # FastAPI path parsing strips '..' segments; the sanitise check in the
        # handler may return 400 or 404 depending on routing — either is safe.
        assert response.status_code in (400, 404)

    @pytest.mark.asyncio
    async def test_forward_slash_in_filename_blocked(self, async_client):
        """Filenames with embedded '/' must be rejected — the router treats
        them as sub-paths and will return 404 (route not found)."""
        response = await async_client.get("/media/videos/sub/evil.mp4")
        assert response.status_code == 404


# ===========================================================================
# Tests: system not initialised (503 responses)
# ===========================================================================


class TestSystemNotInitialized:
    """Endpoints must return 503 when _sign_system is None."""

    @pytest.mark.asyncio
    async def test_health_returns_503_when_system_none(self):
        import api

        with patch.object(api, "_sign_system", None):
            async with AsyncClient(
                transport=ASGITransport(app=api.app), base_url="http://test"
            ) as client:
                response = await client.get("/api/health")
        assert response.status_code == 503

    @pytest.mark.asyncio
    async def test_process_returns_503_when_system_none(self):
        import api

        with patch.object(api, "_sign_system", None):
            async with AsyncClient(
                transport=ASGITransport(app=api.app), base_url="http://test"
            ) as client:
                response = await client.post("/api/process", json={"text": "hello"})
        assert response.status_code == 503

    @pytest.mark.asyncio
    async def test_dictionary_returns_503_when_system_none(self):
        import api

        with patch.object(api, "_sign_system", None):
            async with AsyncClient(
                transport=ASGITransport(app=api.app), base_url="http://test"
            ) as client:
                response = await client.get("/api/dictionary")
        assert response.status_code == 503
