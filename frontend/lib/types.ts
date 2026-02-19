// ---------------------------------------------------------------------------
// Core data models — mirrors the Pydantic schemas in models.py
// ---------------------------------------------------------------------------

export interface SignData {
  gloss: string
  id_gloss?: string
  category: string
  video_url: string
  video_local?: string
  description: string
  keywords?: string[]
  synonyms?: string[]
  handshape?: string
  location?: string
  difficulty?: string
  frequency?: string
  source?: string
}

export type MatchType = 'exact' | 'fuzzy' | 'synonym' | 'semantic' | 'llm' | 'no_match'

export interface SignMatch {
  word: string
  match_type: MatchType
  confidence: number
  sign_data: SignData | null
  matched_synonym?: string
  matched_term?: string
  fuzzy_score?: number
  llm_candidate?: string
}

export interface MatchBreakdown {
  exact: number
  fuzzy: number
  synonym: number
  semantic: number
  [key: string]: number
}

export interface CoverageStats {
  total_tokens: number
  matched_tokens: number
  unmatched_tokens: number
  coverage_rate: number
  exact_matches: number
  fuzzy_matches: number
  synonym_matches: number
  semantic_matches: number
  llm_matches: number
  match_breakdown: MatchBreakdown | Record<string, number>
}

export interface Entity {
  text: string
  label: string
}

export interface NLPAnalysis {
  sentiment: string
  sentiment_score: number
  emotion: string
  intent: string
  entities: Entity[]
  key_phrases: string[]
  formality: string
  complexity: number
  readability: number
}

export interface PhraseAnalysis {
  phrase_type: string
  overall_confidence: number
  grammar_structure: string
}

export interface ProcessResponse {
  original_text: string
  processed_tokens: string[]
  total_tokens: number
  successful_matches: SignMatch[]
  failed_matches: SignMatch[]
  coverage_stats: CoverageStats
  signs_found: number
  nlp_analysis: NLPAnalysis
  phrase_analysis?: PhraseAnalysis
  processing_time_ms?: number
  processed_at?: string
}

// ---------------------------------------------------------------------------
// Request types
// ---------------------------------------------------------------------------

export interface ProcessOptions {
  remove_stops?: boolean
  use_semantic?: boolean
  use_stemming?: boolean
  use_lemmatization?: boolean
  semantic_threshold?: number
  use_intelligent_matching?: boolean
  use_llm?: boolean
}

export interface ProcessRequest {
  text: string
  options?: ProcessOptions
}

// ---------------------------------------------------------------------------
// Health / model status
// ---------------------------------------------------------------------------

export interface HealthResponse {
  status: string
  uptime_seconds: number
  total_signs: number
}

export interface ModelsStatus {
  spacy_available: boolean
  semantic_model_available: boolean
  sentiment_model_available: boolean
  emotion_model_available: boolean
  llm_available: boolean
  intelligent_matching_available: boolean
  total_signs: number
  semantic_model_name: string
  system_version: string
}

// ---------------------------------------------------------------------------
// Dictionary
// ---------------------------------------------------------------------------

export interface DictionaryEntryBrief {
  gloss: string | null
  category: string
  synonyms: string[]
  description: string
}

export interface DictionaryResponse {
  total_entries: number
  categories: Record<string, number>
  entries: Record<string, DictionaryEntryBrief>
}

// ---------------------------------------------------------------------------
// UI state helpers
// ---------------------------------------------------------------------------

export type AppStatus = 'idle' | 'loading' | 'success' | 'error'
