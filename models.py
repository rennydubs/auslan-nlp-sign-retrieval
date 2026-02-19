"""
Pydantic request/response schemas for the Auslan Sign Retrieval FastAPI app.
All request bodies are validated automatically by FastAPI.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field

# ---------------------------------------------------------------------------
# Shared / nested models
# ---------------------------------------------------------------------------


class MatchBreakdown(BaseModel):
    exact: float = 0.0
    fuzzy: float = 0.0
    synonym: float = 0.0
    semantic: float = 0.0


class CoverageStats(BaseModel):
    coverage_rate: float
    exact_matches: int
    fuzzy_matches: int = 0
    synonym_matches: int
    semantic_matches: int
    unmatched_tokens: int
    match_breakdown: MatchBreakdown


class SignData(BaseModel):
    gloss: Optional[str] = None
    category: Optional[str] = None
    video_url: Optional[str] = None
    video_local: Optional[str] = None
    description: Optional[str] = None
    synonyms: Optional[List[str]] = None
    keywords: Optional[List[str]] = None
    handshape: Optional[str] = None
    location: Optional[str] = None
    difficulty: Optional[str] = None
    source: Optional[str] = None


class SignMatch(BaseModel):
    word: str
    match_type: str  # exact | fuzzy | synonym | semantic | llm | no_match
    confidence: float
    sign_data: Optional[SignData] = None
    matched_synonym: Optional[str] = None
    matched_term: Optional[str] = None
    fuzzy_score: Optional[float] = None


class NLPAnalysis(BaseModel):
    sentiment: str
    sentiment_score: float
    emotion: str
    intent: str
    entities: List[Dict[str, Any]] = []
    key_phrases: List[str] = []
    formality: str
    complexity: float
    readability: float


class PhraseAnalysis(BaseModel):
    phrase_type: str
    overall_confidence: float
    grammar_structure: str


# ---------------------------------------------------------------------------
# Request models
# ---------------------------------------------------------------------------


class ProcessOptions(BaseModel):
    remove_stops: bool = False
    use_semantic: bool = True
    use_stemming: bool = False
    use_lemmatization: bool = False
    semantic_threshold: float = Field(0.6, ge=0.0, le=1.0)
    use_intelligent_matching: bool = True
    use_llm: bool = False


class ProcessRequest(BaseModel):
    text: str = Field(..., min_length=1, max_length=5000)
    options: ProcessOptions = Field(default_factory=ProcessOptions)


class EvaluateRequest(BaseModel):
    test_texts: List[str] = Field(default_factory=list)
    options: ProcessOptions = Field(default_factory=ProcessOptions)


class AnalyzeRequest(BaseModel):
    text: str = Field(..., min_length=1, max_length=5000)


class SuggestionsRequest(BaseModel):
    text: str = Field(..., min_length=2, max_length=500)


# ---------------------------------------------------------------------------
# Response models
# ---------------------------------------------------------------------------


class ProcessResponse(BaseModel):
    original_text: str
    processed_tokens: List[str]
    total_tokens: int
    signs_found: int
    successful_matches: List[Dict[str, Any]]
    failed_matches: List[Dict[str, Any]]
    coverage_stats: Dict[str, Any]
    nlp_analysis: Dict[str, Any]
    phrase_analysis: Optional[Dict[str, Any]] = None
    processed_at: datetime = Field(default_factory=datetime.now)


class AnalyzeResponse(BaseModel):
    text: str
    sentiment: Dict[str, Any]
    emotion: str
    intent: str
    entities: List[Dict[str, Any]]
    key_phrases: List[str]
    formality: str
    complexity: float
    readability: float
    analyzed_at: datetime = Field(default_factory=datetime.now)


class DictionaryEntryBrief(BaseModel):
    gloss: Optional[str]
    category: str
    synonyms: List[str]
    description: str


class DictionaryResponse(BaseModel):
    total_entries: int
    categories: Dict[str, int]
    entries: Dict[str, DictionaryEntryBrief]


class SuggestionsResponse(BaseModel):
    partial_text: str
    suggestions: List[str]


class ModelsStatusResponse(BaseModel):
    spacy_available: bool
    semantic_model_available: bool
    sentiment_model_available: bool
    emotion_model_available: bool
    llm_available: bool
    intelligent_matching_available: bool
    total_signs: int
    semantic_model_name: str
    system_version: str


class HealthResponse(BaseModel):
    status: str
    uptime_seconds: float
    total_signs: int
