'use client'

import { motion } from 'framer-motion'
import { Brain, Heart, MessageCircle, Shield, Gauge, Users, Key } from 'lucide-react'
import type { NLPAnalysis as NLPAnalysisType, Entity } from '@/lib/types'
import { cn } from '@/lib/utils'

// ---------------------------------------------------------------------------
// Sentiment styles
// ---------------------------------------------------------------------------

const SENTIMENT_STYLE: Record<string, string> = {
  positive: 'text-emerald-400',
  negative: 'text-red-400',
  neutral: 'text-zinc-400',
}

const SENTIMENT_BG: Record<string, string> = {
  positive: 'border-emerald-500/20 bg-emerald-500/[0.06]',
  negative: 'border-red-500/20 bg-red-500/[0.06]',
  neutral: 'border-zinc-700/60 bg-zinc-800/40',
}

// Entity type colours
const ENTITY_COLORS: Record<string, string> = {
  PERSON:   'bg-violet-500/15 text-violet-400 ring-1 ring-violet-500/25',
  ORG:      'bg-sky-500/15 text-sky-400 ring-1 ring-sky-500/25',
  GPE:      'bg-emerald-500/15 text-emerald-400 ring-1 ring-emerald-500/25',
  LOC:      'bg-teal-500/15 text-teal-400 ring-1 ring-teal-500/25',
  DATE:     'bg-amber-500/15 text-amber-400 ring-1 ring-amber-500/25',
  TIME:     'bg-orange-500/15 text-orange-400 ring-1 ring-orange-500/25',
  CARDINAL: 'bg-zinc-500/15 text-zinc-400 ring-1 ring-zinc-500/25',
  default:  'bg-indigo-500/15 text-indigo-400 ring-1 ring-indigo-500/25',
}

function entityColor(label: string): string {
  return ENTITY_COLORS[label] ?? ENTITY_COLORS.default
}

// ---------------------------------------------------------------------------
// Sub-components
// ---------------------------------------------------------------------------

function SectionLabel({ children, icon }: { children: React.ReactNode; icon?: React.ReactNode }) {
  return (
    <h4 className="mb-2 flex items-center gap-1.5 text-xs font-semibold uppercase tracking-wider text-zinc-500">
      {icon}
      {children}
    </h4>
  )
}

function ScoreBar({ value, max = 1 }: { value: number; max?: number }) {
  const pct = Math.min(100, Math.round((value / max) * 100))
  return (
    <div className="confidence-bar mt-1">
      <motion.div
        initial={{ width: 0 }}
        animate={{ width: `${pct}%` }}
        transition={{ duration: 0.5, ease: 'easeOut' }}
        className="confidence-bar-fill bg-indigo-500"
        role="progressbar"
        aria-valuenow={pct}
        aria-valuemin={0}
        aria-valuemax={100}
      />
    </div>
  )
}

// ---------------------------------------------------------------------------
// Main component
// ---------------------------------------------------------------------------

interface NLPAnalysisProps {
  analysis: NLPAnalysisType
}

export function NLPAnalysis({ analysis }: NLPAnalysisProps) {
  const sentimentKey = analysis.sentiment?.toLowerCase() ?? 'neutral'

  return (
    <div className="rounded-2xl border border-zinc-800/60 bg-zinc-900/50 backdrop-blur-sm">
      {/* Header */}
      <div className="flex items-center gap-2 border-b border-zinc-800/60 px-5 py-3">
        <Brain className="h-4 w-4 text-indigo-400" />
        <h3 className="text-sm font-semibold text-zinc-200">NLP Analysis</h3>
      </div>

      <div className="grid gap-4 p-5 sm:grid-cols-2 lg:grid-cols-3">
        {/* Sentiment */}
        <div className={cn('rounded-xl border p-3', SENTIMENT_BG[sentimentKey] ?? SENTIMENT_BG.neutral)}>
          <SectionLabel icon={<Heart className="h-3 w-3" />}>Sentiment</SectionLabel>
          <span className={cn('text-lg font-bold capitalize', SENTIMENT_STYLE[sentimentKey] ?? 'text-zinc-300')}>
            {analysis.sentiment}
          </span>
          <div className="mt-2 flex items-center justify-between text-xs text-zinc-500">
            <span>Score</span>
            <span className="text-zinc-300">{analysis.sentiment_score.toFixed(2)}</span>
          </div>
          <ScoreBar value={Math.abs(analysis.sentiment_score)} />
        </div>

        {/* Emotion */}
        <div className="rounded-xl border border-zinc-800/60 bg-zinc-800/30 p-3">
          <SectionLabel icon={<Heart className="h-3 w-3" />}>Emotion</SectionLabel>
          <p className="font-semibold capitalize text-zinc-200">
            {analysis.emotion}
          </p>
        </div>

        {/* Intent */}
        <div className="rounded-xl border border-zinc-800/60 bg-zinc-800/30 p-3">
          <SectionLabel icon={<MessageCircle className="h-3 w-3" />}>Intent</SectionLabel>
          <span className="inline-block rounded-full bg-indigo-500/15 px-3 py-0.5 text-sm font-semibold capitalize text-indigo-400 ring-1 ring-indigo-500/25">
            {analysis.intent}
          </span>
        </div>

        {/* Formality */}
        <div className="rounded-xl border border-zinc-800/60 bg-zinc-800/30 p-3">
          <SectionLabel icon={<Shield className="h-3 w-3" />}>Formality</SectionLabel>
          <p className="font-semibold capitalize text-zinc-200">{analysis.formality}</p>
        </div>

        {/* Complexity + Readability */}
        <div className="rounded-xl border border-zinc-800/60 bg-zinc-800/30 p-3">
          <SectionLabel icon={<Gauge className="h-3 w-3" />}>Complexity</SectionLabel>
          <div className="flex items-center justify-between text-xs text-zinc-500">
            <span>Complexity</span>
            <span className="text-zinc-300">{analysis.complexity.toFixed(2)}</span>
          </div>
          <ScoreBar value={analysis.complexity} />
          <div className="mt-2 flex items-center justify-between text-xs text-zinc-500">
            <span>Readability</span>
            <span className="text-zinc-300">{analysis.readability.toFixed(0)}</span>
          </div>
          <ScoreBar value={analysis.readability} max={100} />
        </div>

        {/* Named entities */}
        {analysis.entities && analysis.entities.length > 0 && (
          <div className="rounded-xl border border-zinc-800/60 bg-zinc-800/30 p-3">
            <SectionLabel icon={<Users className="h-3 w-3" />}>Entities</SectionLabel>
            <div className="flex flex-wrap gap-1.5">
              {analysis.entities.map((ent: Entity, i: number) => (
                <span
                  key={i}
                  title={ent.label}
                  className={cn('rounded-full px-2.5 py-0.5 text-xs font-medium', entityColor(ent.label))}
                >
                  {ent.text}
                  <span className="ml-1 opacity-50">{ent.label}</span>
                </span>
              ))}
            </div>
          </div>
        )}

        {/* Key phrases */}
        {analysis.key_phrases && analysis.key_phrases.length > 0 && (
          <div className="rounded-xl border border-zinc-800/60 bg-zinc-800/30 p-3 sm:col-span-2 lg:col-span-3">
            <SectionLabel icon={<Key className="h-3 w-3" />}>Key phrases</SectionLabel>
            <div className="flex flex-wrap gap-2">
              {analysis.key_phrases.map((phrase, i) => (
                <span
                  key={i}
                  className="rounded-lg bg-zinc-800 px-2.5 py-1 text-sm text-zinc-300 ring-1 ring-zinc-700/50"
                >
                  {phrase}
                </span>
              ))}
            </div>
          </div>
        )}
      </div>
    </div>
  )
}
