"""
FastAPI application for the Auslan Sign Retrieval System.

Replaces the Flask app.py with async-capable, auto-documented endpoints.
Run with:  uvicorn api:app --reload --host 0.0.0.0 --port 8000
Docs at:   http://localhost:8000/docs
"""

from __future__ import annotations

import logging
import os
import time
from contextlib import asynccontextmanager
from datetime import datetime
from typing import Any, Dict

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

import config
from main import AuslanSignSystem
from models import (
    AnalyzeRequest,
    AnalyzeResponse,
    DictionaryEntryBrief,
    DictionaryResponse,
    EvaluateRequest,
    HealthResponse,
    ModelsStatusResponse,
    ProcessRequest,
    ProcessResponse,
    SuggestionsRequest,
    SuggestionsResponse,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Application startup / singleton
# ---------------------------------------------------------------------------

_sign_system: AuslanSignSystem | None = None
_start_time: float = time.time()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialize the sign system once on startup."""
    global _sign_system
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    logger.info("Starting Auslan Sign Retrieval API...")
    _sign_system = AuslanSignSystem()
    logger.info("System ready. %d signs loaded.", len(_sign_system.matcher.gloss_dict) if _sign_system.matcher else 0)
    yield
    logger.info("Shutting down.")


app = FastAPI(
    title="Auslan Sign Retrieval API",
    description="NLP-powered system that matches natural language text to Auslan sign videos.",
    version="2.0.0",
    lifespan=lifespan,
)

# ---------------------------------------------------------------------------
# CORS — allow the Next.js dev server (port 3000) and any origin in dev
# ---------------------------------------------------------------------------

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://localhost:3001"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def get_system() -> AuslanSignSystem:
    if _sign_system is None:
        raise HTTPException(status_code=503, detail="System not yet initialized")
    return _sign_system


# ---------------------------------------------------------------------------
# Health / utility endpoints
# ---------------------------------------------------------------------------

@app.get("/api/health", response_model=HealthResponse, tags=["System"])
async def health():
    """Health check endpoint."""
    system = get_system()
    return HealthResponse(
        status="ok",
        uptime_seconds=round(time.time() - _start_time, 1),
        total_signs=len(system.matcher.gloss_dict) if system.matcher else 0,
    )


@app.get("/api/models/status", response_model=ModelsStatusResponse, tags=["System"])
async def model_status():
    """Check which AI models and components are loaded."""
    system = get_system()
    return ModelsStatusResponse(
        spacy_available=system.nlp_processor is not None and system.nlp_processor.nlp is not None,
        semantic_model_available=system.matcher is not None and getattr(system.matcher, "semantic_model", None) is not None,
        sentiment_model_available=system.nlp_processor is not None and system.nlp_processor.sentiment_model is not None,
        emotion_model_available=system.nlp_processor is not None and system.nlp_processor.emotion_model is not None,
        llm_available=hasattr(system, "llm_processor") and getattr(system, "llm_processor", None) is not None and system.llm_processor.available,
        intelligent_matching_available=system.phrase_matcher is not None,
        total_signs=len(system.matcher.gloss_dict) if system.matcher else 0,
        semantic_model_name=config.SEMANTIC_MODEL_NAME,
        system_version="2.0.0",
    )


# ---------------------------------------------------------------------------
# Core processing endpoint
# ---------------------------------------------------------------------------

@app.post("/api/process", tags=["Matching"])
async def process_text(req: ProcessRequest) -> Dict[str, Any]:
    """
    Process natural language text and return matched Auslan signs.

    The pipeline runs: exact → fuzzy → synonym → semantic → (optional LLM fallback).
    """
    system = get_system()
    opts = req.options

    results = system.process_input(
        req.text,
        remove_stops=opts.remove_stops,
        use_semantic=opts.use_semantic,
        semantic_threshold=opts.semantic_threshold,
        use_stemming=opts.use_stemming,
        use_intelligent_matching=opts.use_intelligent_matching,
    )
    results["processed_at"] = datetime.now().isoformat()
    return results


# ---------------------------------------------------------------------------
# Batch evaluation endpoint
# ---------------------------------------------------------------------------

@app.post("/api/evaluate", tags=["Evaluation"])
async def evaluate_system(req: EvaluateRequest) -> Dict[str, Any]:
    """Run batch evaluation across a list of test sentences."""
    system = get_system()
    opts = req.options

    test_texts = req.test_texts or [
        "Hello, how are you today?",
        "I need help finding the toilet",
        "Let's exercise and build muscle strength",
        "Warm up before lifting weights",
        "Cool down with stretching exercises",
    ]

    evaluation = system.batch_evaluation(
        test_texts,
        remove_stops=opts.remove_stops,
        use_semantic=opts.use_semantic,
        semantic_threshold=opts.semantic_threshold,
        use_stemming=opts.use_stemming,
    )
    return evaluation


# ---------------------------------------------------------------------------
# NLP analysis endpoint
# ---------------------------------------------------------------------------

@app.post("/api/analyze", response_model=AnalyzeResponse, tags=["NLP"])
async def analyze_text(req: AnalyzeRequest) -> AnalyzeResponse:
    """Perform detailed NLP analysis (sentiment, emotion, entities, intent)."""
    system = get_system()
    if not system.nlp_processor:
        raise HTTPException(status_code=503, detail="NLP processor not available")

    analysis = system.nlp_processor.analyze_text(req.text)

    return AnalyzeResponse(
        text=req.text,
        sentiment={
            "label": analysis.sentiment_label,
            "score": analysis.sentiment_score,
            "confidence": analysis.confidence,
        },
        emotion=analysis.emotion,
        intent=analysis.intent,
        entities=analysis.entities,
        key_phrases=analysis.key_phrases,
        formality=analysis.formality_level,
        complexity=analysis.complexity_score,
        readability=analysis.readability_score,
    )


# ---------------------------------------------------------------------------
# Autocomplete suggestions endpoint
# ---------------------------------------------------------------------------

@app.post("/api/suggestions", response_model=SuggestionsResponse, tags=["NLP"])
async def get_suggestions(req: SuggestionsRequest) -> SuggestionsResponse:
    """Return phrase autocomplete suggestions for partial input."""
    system = get_system()
    if not system.phrase_matcher:
        return SuggestionsResponse(partial_text=req.text, suggestions=[])

    suggestions = system.phrase_matcher.get_phrase_suggestions(req.text, limit=8)
    return SuggestionsResponse(partial_text=req.text, suggestions=suggestions)


# ---------------------------------------------------------------------------
# Dictionary endpoint
# ---------------------------------------------------------------------------

@app.get("/api/dictionary", response_model=DictionaryResponse, tags=["Dictionary"])
async def get_dictionary() -> DictionaryResponse:
    """Return the full dictionary with category counts and all entries."""
    system = get_system()
    if not system.matcher:
        raise HTTPException(status_code=503, detail="Matcher not available")

    categories: Dict[str, int] = {}
    entries: Dict[str, DictionaryEntryBrief] = {}

    for word, data in system.matcher.gloss_dict.items():
        category = data.get("category", "unknown")
        categories[category] = categories.get(category, 0) + 1

        description = data.get("description", "")
        entries[word] = DictionaryEntryBrief(
            gloss=data.get("gloss"),
            category=category,
            synonyms=data.get("synonyms", [])[:3],
            description=description[:100] + "..." if len(description) > 100 else description,
        )

    return DictionaryResponse(
        total_entries=len(system.matcher.gloss_dict),
        categories=categories,
        entries=entries,
    )


# ---------------------------------------------------------------------------
# Video serving
# ---------------------------------------------------------------------------

@app.get("/media/videos/{filename}", tags=["Media"])
async def serve_video(filename: str):
    """Serve a sign video file, checking both the repo and scraped video dirs."""
    # Sanitise: only allow safe filenames
    if "/" in filename or "\\" in filename or ".." in filename:
        raise HTTPException(status_code=400, detail="Invalid filename")

    for video_dir in config.VIDEO_DIRS:
        video_path = os.path.join(video_dir, filename)
        if os.path.isfile(video_path):
            return FileResponse(video_path, media_type="video/mp4")

    raise HTTPException(status_code=404, detail="Video not found")


# ---------------------------------------------------------------------------
# Entrypoint
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "api:app",
        host=config.SERVER_HOST,
        port=config.SERVER_PORT,
        reload=config.DEBUG,
        log_level="info",
    )
