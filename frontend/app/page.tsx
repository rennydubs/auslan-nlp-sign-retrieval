'use client'

import { useState, useCallback } from 'react'
import { motion, AnimatePresence } from 'framer-motion'
import {
  Settings2,
  ChevronDown,
  Sparkles,
  Zap,
  Brain,
  Bot,
  Timer,
  Hand,
} from 'lucide-react'
import { processText } from '@/lib/api'
import type { ProcessResponse, ProcessOptions, AppStatus } from '@/lib/types'
import { SearchBar } from '@/components/SearchBar'
import { SignResults } from '@/components/SignResults'
import { NLPAnalysis } from '@/components/NLPAnalysis'
import { StatsPanel } from '@/components/StatsPanel'
import { LoadingSpinner } from '@/components/LoadingSpinner'
import { ErrorMessage } from '@/components/ErrorMessage'
import { cn } from '@/lib/utils'

// ---------------------------------------------------------------------------
// Default options
// ---------------------------------------------------------------------------

const DEFAULT_OPTIONS: ProcessOptions = {
  use_semantic: true,
  remove_stops: false,
  use_stemming: false,
  use_llm: false,
  semantic_threshold: 0.6,
  use_intelligent_matching: true,
}

// ---------------------------------------------------------------------------
// Page
// ---------------------------------------------------------------------------

export default function HomePage() {
  const [status, setStatus] = useState<AppStatus>('idle')
  const [results, setResults] = useState<ProcessResponse | null>(null)
  const [error, setError] = useState<string | null>(null)
  const [options, setOptions] = useState<ProcessOptions>(DEFAULT_OPTIONS)
  const [optionsOpen, setOptionsOpen] = useState(false)

  const handleSubmit = useCallback(
    async (text: string) => {
      if (!text.trim()) return
      setStatus('loading')
      setError(null)
      setResults(null)

      try {
        const data = await processText({ text: text.trim(), options })
        setResults(data)
        setStatus('success')
      } catch (err) {
        setError(err instanceof Error ? err.message : 'An unexpected error occurred.')
        setStatus('error')
      }
    },
    [options],
  )

  const toggleOption = (key: keyof ProcessOptions) => {
    setOptions((prev) => ({ ...prev, [key]: !prev[key] }))
  }

  const setThreshold = (value: number) => {
    setOptions((prev) => ({ ...prev, semantic_threshold: value }))
  }

  return (
    <div className="mx-auto max-w-7xl px-4 py-10 sm:px-6 lg:px-8">
      {/* Hero section */}
      <motion.section
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.5, ease: 'easeOut' }}
        className="mb-10 text-center"
      >
        <div className="mx-auto mb-4 flex h-14 w-14 items-center justify-center rounded-2xl bg-indigo-500/15 ring-1 ring-indigo-500/25">
          <Hand className="h-7 w-7 text-indigo-400" />
        </div>
        <h1 className="mb-3 text-4xl font-bold tracking-tight text-zinc-100 sm:text-5xl">
          Translate English to{' '}
          <span className="text-gradient">Auslan</span>
        </h1>
        <p className="mx-auto max-w-2xl text-lg text-zinc-400">
          Type any phrase and the system will find matching Australian Sign Language
          signs using NLP.
        </p>
      </motion.section>

      {/* Search area */}
      <motion.section
        initial={{ opacity: 0, y: 16 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.5, delay: 0.1, ease: 'easeOut' }}
        className="mb-6"
      >
        <SearchBar onSubmit={handleSubmit} isLoading={status === 'loading'} />
      </motion.section>

      {/* Options panel */}
      <motion.section
        initial={{ opacity: 0, y: 12 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.5, delay: 0.15, ease: 'easeOut' }}
        className="mb-8"
      >
        <OptionsPanel
          options={options}
          onToggle={toggleOption}
          onThreshold={setThreshold}
          open={optionsOpen}
          onOpenChange={setOptionsOpen}
        />
      </motion.section>

      {/* Suggestion pills — shown in idle state */}
      {status === 'idle' && (
        <SuggestionPills onSelect={handleSubmit} />
      )}

      {/* Loading state */}
      {status === 'loading' && (
        <div className="flex justify-center py-20">
          <LoadingSpinner label="Processing your text..." size="lg" />
        </div>
      )}

      {/* Error state */}
      {status === 'error' && error && (
        <motion.div
          initial={{ opacity: 0, y: 8 }}
          animate={{ opacity: 1, y: 0 }}
          className="mb-8"
        >
          <ErrorMessage
            message={error}
            detail="Make sure the FastAPI backend is running on http://localhost:8000."
          />
        </motion.div>
      )}

      {/* Results */}
      <AnimatePresence>
        {status === 'success' && results && (
          <motion.div
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            transition={{ duration: 0.4 }}
            className="space-y-8"
          >
            {/* Stats row */}
            <StatsPanel stats={results.coverage_stats} signsFound={results.signs_found} />

            {/* Two-column: NLP analysis + unmatched */}
            <div className="grid gap-6 lg:grid-cols-3">
              <div className="lg:col-span-2">
                <NLPAnalysis analysis={results.nlp_analysis} />
              </div>

              {results.failed_matches.length > 0 && (
                <div className="rounded-2xl border border-zinc-800/60 bg-zinc-900/50 p-5 backdrop-blur-sm">
                  <h3 className="mb-3 text-sm font-semibold uppercase tracking-wider text-zinc-500">
                    Unmatched words
                  </h3>
                  <div className="flex flex-wrap gap-2">
                    {results.failed_matches.map((m, i) => (
                      <span
                        key={i}
                        className="rounded-full bg-zinc-800 px-3 py-1 text-sm text-zinc-400 ring-1 ring-zinc-700/50"
                      >
                        {m.word}
                      </span>
                    ))}
                  </div>
                  <p className="mt-3 text-xs text-zinc-500">
                    These words had no match in the sign dictionary.
                  </p>
                </div>
              )}
            </div>

            {/* Phrase analysis banner */}
            {results.phrase_analysis && (
              <PhraseAnalysisBanner analysis={results.phrase_analysis} />
            )}

            {/* Sign result cards */}
            {results.successful_matches.length > 0 ? (
              <SignResults matches={results.successful_matches} />
            ) : (
              <div className="rounded-2xl border border-dashed border-zinc-700 py-16 text-center">
                <p className="text-zinc-500">
                  No signs matched for this input. Try different words or enable
                  more matching options.
                </p>
              </div>
            )}

            {/* Timing */}
            {results.processing_time_ms !== undefined && (
              <p className="flex items-center justify-end gap-1.5 text-xs text-zinc-500">
                <Timer className="h-3 w-3" />
                Processed in {results.processing_time_ms.toFixed(0)} ms
              </p>
            )}
          </motion.div>
        )}
      </AnimatePresence>
    </div>
  )
}

// ---------------------------------------------------------------------------
// Suggestion pills
// ---------------------------------------------------------------------------

function SuggestionPills({ onSelect }: { onSelect: (text: string) => void }) {
  const examples = [
    'Hello, how are you?',
    'I feel happy today',
    'Please help me find the toilet',
    'Warm up before exercise',
    'Thank you very much',
    'What time is it?',
  ]

  return (
    <motion.div
      initial={{ opacity: 0, y: 12 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.5, delay: 0.2 }}
      className="py-12 text-center"
    >
      <p className="mb-4 text-sm text-zinc-500">Try one of these phrases:</p>
      <div className="flex flex-wrap justify-center gap-2">
        {examples.map((ex) => (
          <motion.button
            key={ex}
            whileHover={{ scale: 1.05 }}
            whileTap={{ scale: 0.97 }}
            onClick={() => onSelect(ex)}
            className={cn(
              'rounded-full border border-zinc-700/60 bg-zinc-800/50 px-4 py-2 text-sm text-zinc-300',
              'transition-colors duration-200 hover:border-indigo-500/40 hover:bg-indigo-500/10 hover:text-indigo-300'
            )}
          >
            {ex}
          </motion.button>
        ))}
      </div>
    </motion.div>
  )
}

// ---------------------------------------------------------------------------
// Options panel
// ---------------------------------------------------------------------------

interface OptionsPanelProps {
  options: ProcessOptions
  onToggle: (key: keyof ProcessOptions) => void
  onThreshold: (value: number) => void
  open: boolean
  onOpenChange: (open: boolean) => void
}

function OptionsPanel({ options, onToggle, onThreshold, open, onOpenChange }: OptionsPanelProps) {
  return (
    <div className="mx-auto max-w-3xl">
      <button
        onClick={() => onOpenChange(!open)}
        className={cn(
          'flex w-full items-center justify-between rounded-2xl border px-5 py-3 text-sm font-medium transition-all duration-200',
          open
            ? 'border-zinc-700/60 bg-zinc-900/70 text-zinc-200'
            : 'border-zinc-800/60 bg-zinc-900/40 text-zinc-400 hover:border-zinc-700/60 hover:text-zinc-300'
        )}
      >
        <span className="flex items-center gap-2">
          <Settings2 className="h-4 w-4" />
          Processing options
        </span>
        <ChevronDown className={cn('h-4 w-4 transition-transform duration-200', open && 'rotate-180')} />
      </button>

      <AnimatePresence>
        {open && (
          <motion.div
            initial={{ height: 0, opacity: 0 }}
            animate={{ height: 'auto', opacity: 1 }}
            exit={{ height: 0, opacity: 0 }}
            transition={{ duration: 0.2, ease: 'easeInOut' }}
            className="overflow-hidden"
          >
            <div className="mt-2 rounded-2xl border border-zinc-800/60 bg-zinc-900/50 px-5 py-4 backdrop-blur-sm">
              <div className="grid gap-3 sm:grid-cols-2 lg:grid-cols-4">
                <ToggleOption
                  label="Semantic matching"
                  description="Sentence-transformer embeddings"
                  checked={options.use_semantic ?? true}
                  onChange={() => onToggle('use_semantic')}
                  icon={<Brain className="h-3.5 w-3.5" />}
                />
                <ToggleOption
                  label="Remove stop words"
                  description="Filter common words"
                  checked={options.remove_stops ?? false}
                  onChange={() => onToggle('remove_stops')}
                  icon={<Zap className="h-3.5 w-3.5" />}
                />
                <ToggleOption
                  label="Stemming"
                  description="Root-form matching"
                  checked={options.use_stemming ?? false}
                  onChange={() => onToggle('use_stemming')}
                  icon={<Sparkles className="h-3.5 w-3.5" />}
                />
                <ToggleOption
                  label="LLM fallback"
                  description="Language model for unknowns"
                  checked={options.use_llm ?? false}
                  onChange={() => onToggle('use_llm')}
                  icon={<Bot className="h-3.5 w-3.5" />}
                />
              </div>

              {/* Threshold slider */}
              {options.use_semantic && (
                <div className="mt-4 flex items-center gap-4">
                  <label className="w-48 text-sm text-zinc-400">
                    Semantic threshold:{' '}
                    <span className="font-semibold text-zinc-200">
                      {(options.semantic_threshold ?? 0.6).toFixed(2)}
                    </span>
                  </label>
                  <input
                    type="range"
                    min="0"
                    max="1"
                    step="0.05"
                    value={options.semantic_threshold ?? 0.6}
                    onChange={(e) => onThreshold(parseFloat(e.target.value))}
                    className="h-1.5 w-full cursor-pointer appearance-none rounded-full bg-zinc-700 accent-indigo-500 [&::-webkit-slider-thumb]:h-3.5 [&::-webkit-slider-thumb]:w-3.5 [&::-webkit-slider-thumb]:rounded-full [&::-webkit-slider-thumb]:bg-indigo-400 [&::-webkit-slider-thumb]:appearance-none"
                    aria-label="Semantic similarity threshold"
                  />
                  <span className="w-8 text-right text-xs text-zinc-500">1.0</span>
                </div>
              )}
            </div>
          </motion.div>
        )}
      </AnimatePresence>
    </div>
  )
}

// ---------------------------------------------------------------------------
// Toggle option
// ---------------------------------------------------------------------------

interface ToggleOptionProps {
  label: string
  description: string
  checked: boolean
  onChange: () => void
  icon?: React.ReactNode
}

function ToggleOption({ label, description, checked, onChange, icon }: ToggleOptionProps) {
  return (
    <label className="flex cursor-pointer items-start gap-3 rounded-xl p-2 transition-colors hover:bg-white/[0.03]">
      <div className="relative mt-0.5 flex-shrink-0">
        <input type="checkbox" checked={checked} onChange={onChange} className="peer sr-only" />
        <div
          className={cn(
            'h-5 w-9 rounded-full transition-colors duration-200',
            checked ? 'bg-indigo-600' : 'bg-zinc-700'
          )}
        />
        <div
          className={cn(
            'absolute top-0.5 h-4 w-4 rounded-full bg-white shadow transition-transform duration-200',
            checked ? 'translate-x-4' : 'translate-x-0.5'
          )}
        />
      </div>
      <div>
        <p className="flex items-center gap-1.5 text-sm font-medium text-zinc-200">
          {icon && <span className="text-zinc-500">{icon}</span>}
          {label}
        </p>
        <p className="text-xs text-zinc-500">{description}</p>
      </div>
    </label>
  )
}

// ---------------------------------------------------------------------------
// Phrase analysis banner
// ---------------------------------------------------------------------------

interface PhraseAnalysisBannerProps {
  analysis: {
    phrase_type: string
    overall_confidence: number
    grammar_structure: string
  }
}

function PhraseAnalysisBanner({ analysis }: PhraseAnalysisBannerProps) {
  return (
    <div className="flex flex-wrap items-center gap-3 rounded-2xl border border-indigo-500/20 bg-indigo-500/[0.06] px-5 py-3 text-sm">
      <span className="font-medium text-indigo-300">Phrase structure:</span>
      <span className="rounded-full bg-indigo-500/15 px-2.5 py-0.5 text-xs font-semibold text-indigo-400 ring-1 ring-indigo-500/25">
        {analysis.phrase_type}
      </span>
      <span className="text-indigo-400/80">{analysis.grammar_structure}</span>
      <span className="ml-auto text-indigo-400">
        Confidence: {(analysis.overall_confidence * 100).toFixed(0)}%
      </span>
    </div>
  )
}
