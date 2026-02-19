'use client'

import { motion } from 'framer-motion'
import { Hash, CheckCircle2, XCircle, Percent } from 'lucide-react'
import type { CoverageStats } from '@/lib/types'
import { cn } from '@/lib/utils'

// ---------------------------------------------------------------------------
// Stat tile
// ---------------------------------------------------------------------------

interface StatTileProps {
  label: string
  value: string | number
  subtitle?: string
  icon: React.ReactNode
  accentClass?: string
}

function StatTile({ label, value, subtitle, icon, accentClass }: StatTileProps) {
  return (
    <div
      className={cn(
        'flex flex-col rounded-2xl border border-zinc-800/60 bg-zinc-900/50 px-4 py-3 backdrop-blur-sm',
        accentClass
      )}
    >
      <div className="mb-1 flex items-center gap-2">
        <span className="text-zinc-500">{icon}</span>
        <span className="text-xs font-medium uppercase tracking-wider text-zinc-500">
          {label}
        </span>
      </div>
      <span className="text-2xl font-bold text-zinc-100">{value}</span>
      {subtitle && (
        <span className="mt-0.5 text-xs text-zinc-500">{subtitle}</span>
      )}
    </div>
  )
}

// ---------------------------------------------------------------------------
// Breakdown row
// ---------------------------------------------------------------------------

interface BreakdownRowProps {
  label: string
  count: number
  total: number
  colorClass: string
}

function BreakdownRow({ label, count, total, colorClass }: BreakdownRowProps) {
  const pct = total > 0 ? Math.round((count / total) * 100) : 0
  if (count === 0) return null

  return (
    <div className="flex items-center gap-3">
      <span className="w-20 flex-shrink-0 text-right text-xs font-medium text-zinc-400">
        {label}
      </span>
      <div className="h-2 flex-1 overflow-hidden rounded-full bg-zinc-800">
        <motion.div
          initial={{ width: 0 }}
          animate={{ width: `${pct}%` }}
          transition={{ duration: 0.6, ease: 'easeOut' }}
          className={cn('h-2 rounded-full', colorClass)}
          role="progressbar"
          aria-valuenow={pct}
          aria-valuemin={0}
          aria-valuemax={100}
          aria-label={`${label}: ${count}`}
        />
      </div>
      <span className="w-8 flex-shrink-0 text-right text-xs text-zinc-500">
        {count}
      </span>
    </div>
  )
}

// ---------------------------------------------------------------------------
// StatsPanel
// ---------------------------------------------------------------------------

interface StatsPanelProps {
  stats: CoverageStats
  signsFound: number
}

export function StatsPanel({ stats, signsFound }: StatsPanelProps) {
  const coveragePct = Math.round((stats.coverage_rate ?? 0) * 100)
  const total = stats.total_tokens ?? 0
  const matchedCount = signsFound

  return (
    <section aria-label="Coverage statistics">
      {/* Tiles */}
      <div className="mb-4 grid grid-cols-2 gap-3 sm:grid-cols-4">
        <StatTile
          label="Total tokens"
          value={total}
          subtitle="words processed"
          icon={<Hash className="h-3.5 w-3.5" />}
        />
        <StatTile
          label="Signs found"
          value={signsFound}
          subtitle="matched"
          icon={<CheckCircle2 className="h-3.5 w-3.5" />}
          accentClass="border-indigo-500/20"
        />
        <StatTile
          label="Unmatched"
          value={stats.unmatched_tokens ?? 0}
          subtitle="no sign found"
          icon={<XCircle className="h-3.5 w-3.5" />}
        />
        <StatTile
          label="Coverage"
          value={`${coveragePct}%`}
          subtitle="of tokens matched"
          icon={<Percent className="h-3.5 w-3.5" />}
          accentClass={
            coveragePct >= 80
              ? 'border-emerald-500/20'
              : coveragePct >= 50
              ? 'border-amber-500/20'
              : 'border-red-500/20'
          }
        />
      </div>

      {/* Breakdown */}
      {matchedCount > 0 && (
        <div className="rounded-2xl border border-zinc-800/60 bg-zinc-900/50 px-5 py-4 backdrop-blur-sm">
          <h3 className="mb-3 text-xs font-semibold uppercase tracking-wider text-zinc-500">
            Match type breakdown
          </h3>
          <div className="space-y-2">
            <BreakdownRow label="Exact" count={stats.exact_matches ?? 0} total={total} colorClass="bg-emerald-500" />
            <BreakdownRow label="Fuzzy" count={stats.fuzzy_matches ?? 0} total={total} colorClass="bg-amber-500" />
            <BreakdownRow label="Synonym" count={stats.synonym_matches ?? 0} total={total} colorClass="bg-sky-500" />
            <BreakdownRow label="Semantic" count={stats.semantic_matches ?? 0} total={total} colorClass="bg-violet-500" />
            <BreakdownRow label="LLM" count={stats.llm_matches ?? 0} total={total} colorClass="bg-orange-500" />
          </div>
        </div>
      )}
    </section>
  )
}
