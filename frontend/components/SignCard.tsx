'use client'

import { useState, useRef } from 'react'
import { motion } from 'framer-motion'
import { Play, Pause, VideoOff, Tag, MapPin, Fingerprint } from 'lucide-react'
import type { SignMatch, MatchType } from '@/lib/types'
import { videoUrl } from '@/lib/api'
import { cn } from '@/lib/utils'

// ---------------------------------------------------------------------------
// Match type badge config — dark glassmorphism style
// ---------------------------------------------------------------------------

const MATCH_CONFIG: Record<
  MatchType,
  { label: string; badgeClass: string; barClass: string }
> = {
  exact: {
    label: 'Exact',
    badgeClass: 'bg-emerald-500/15 text-emerald-400 ring-1 ring-emerald-500/25',
    barClass: 'bg-emerald-500',
  },
  fuzzy: {
    label: 'Fuzzy',
    badgeClass: 'bg-amber-500/15 text-amber-400 ring-1 ring-amber-500/25',
    barClass: 'bg-amber-500',
  },
  synonym: {
    label: 'Synonym',
    badgeClass: 'bg-sky-500/15 text-sky-400 ring-1 ring-sky-500/25',
    barClass: 'bg-sky-500',
  },
  semantic: {
    label: 'Semantic',
    badgeClass: 'bg-violet-500/15 text-violet-400 ring-1 ring-violet-500/25',
    barClass: 'bg-violet-500',
  },
  llm: {
    label: 'LLM',
    badgeClass: 'bg-orange-500/15 text-orange-400 ring-1 ring-orange-500/25',
    barClass: 'bg-orange-500',
  },
  no_match: {
    label: 'No match',
    badgeClass: 'bg-zinc-500/15 text-zinc-400 ring-1 ring-zinc-500/25',
    barClass: 'bg-zinc-600',
  },
}

// ---------------------------------------------------------------------------
// Video player
// ---------------------------------------------------------------------------

function VideoPlayer({ src, gloss }: { src: string; gloss: string }) {
  const videoRef = useRef<HTMLVideoElement>(null)
  const [failed, setFailed] = useState(false)
  const [playing, setPlaying] = useState(false)

  const togglePlay = () => {
    const vid = videoRef.current
    if (!vid) return
    if (vid.paused) {
      vid.play()
      setPlaying(true)
    } else {
      vid.pause()
      setPlaying(false)
    }
  }

  if (failed) {
    return (
      <div className="flex h-40 w-full items-center justify-center rounded-xl bg-zinc-800/50 text-sm text-zinc-500">
        <VideoOff className="mr-2 h-4 w-4" />
        Video unavailable
      </div>
    )
  }

  return (
    <div className="relative overflow-hidden rounded-xl bg-black/40">
      <video
        ref={videoRef}
        src={src}
        loop
        playsInline
        preload="metadata"
        onError={() => setFailed(true)}
        onEnded={() => setPlaying(false)}
        aria-label={`Sign video for ${gloss}`}
        className="w-full object-contain"
        style={{ maxHeight: '200px' }}
      />
      <button
        onClick={togglePlay}
        aria-label={playing ? `Pause ${gloss}` : `Play ${gloss}`}
        className="absolute inset-0 flex items-center justify-center transition-colors hover:bg-black/20"
      >
        {!playing && (
          <span className="flex h-11 w-11 items-center justify-center rounded-full bg-white/10 text-white backdrop-blur-sm ring-1 ring-white/20 transition-transform hover:scale-110">
            <Play className="h-5 w-5 pl-0.5" />
          </span>
        )}
        {playing && (
          <span className="flex h-11 w-11 items-center justify-center rounded-full bg-white/10 text-white opacity-0 backdrop-blur-sm ring-1 ring-white/20 transition-opacity hover:opacity-100">
            <Pause className="h-5 w-5" />
          </span>
        )}
      </button>
    </div>
  )
}

// ---------------------------------------------------------------------------
// SignCard
// ---------------------------------------------------------------------------

interface SignCardProps {
  match: SignMatch
  index?: number
}

export function SignCard({ match, index = 0 }: SignCardProps) {
  const config = MATCH_CONFIG[match.match_type] ?? MATCH_CONFIG.no_match
  const confidence = Math.round(match.confidence * 100)
  const sign = match.sign_data

  return (
    <motion.article
      initial={{ opacity: 0, y: 16 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.35, delay: index * 0.05, ease: 'easeOut' }}
      className={cn(
        'group flex flex-col overflow-hidden rounded-2xl',
        'border border-zinc-800/60 bg-zinc-900/50 backdrop-blur-sm',
        'transition-all duration-300 hover:border-zinc-700/60 hover:bg-zinc-900/70',
        'hover:shadow-xl hover:shadow-black/20'
      )}
    >
      {/* Header: gloss + badge */}
      <div className="flex items-start justify-between gap-2 border-b border-zinc-800/60 px-4 py-3">
        <div>
          <h3 className="text-lg font-bold uppercase tracking-wide text-zinc-100">
            {sign?.gloss ?? match.word}
          </h3>
          {match.matched_synonym && (
            <p className="text-xs text-zinc-500">
              via synonym: <span className="italic text-zinc-400">{match.matched_synonym}</span>
            </p>
          )}
          {match.matched_term && match.matched_term !== match.word && (
            <p className="text-xs text-zinc-500">
              matched: <span className="italic text-zinc-400">{match.matched_term}</span>
            </p>
          )}
        </div>
        <span className={cn('badge flex-shrink-0 rounded-full px-2.5 py-0.5 text-xs font-medium', config.badgeClass)}>
          {config.label}
        </span>
      </div>

      {/* Video — only rendered when a URL exists */}
      {sign?.video_url && (
        <div className="px-3 pt-3">
          <VideoPlayer src={videoUrl(sign.video_url)} gloss={sign.gloss ?? match.word} />
        </div>
      )}

      {/* Body */}
      <div className="flex-1 space-y-3 px-4 py-3">
        {/* Confidence bar */}
        <div>
          <div className="mb-1.5 flex items-center justify-between text-xs text-zinc-400">
            <span>Confidence</span>
            <span className="font-semibold text-zinc-200">{confidence}%</span>
          </div>
          <div className="confidence-bar">
            <div
              className={cn('confidence-bar-fill', config.barClass)}
              style={{ width: `${confidence}%` }}
              role="progressbar"
              aria-valuenow={confidence}
              aria-valuemin={0}
              aria-valuemax={100}
              aria-label={`${confidence}% confidence`}
            />
          </div>
        </div>

        {/* Category + metadata */}
        {sign && (
          <>
            {sign.category && (
              <div className="flex flex-wrap gap-1.5">
                <span className="inline-flex items-center gap-1 rounded-full bg-zinc-800 px-2 py-0.5 text-xs text-zinc-400 ring-1 ring-zinc-700/50">
                  <Tag className="h-3 w-3" />
                  {sign.category}
                </span>
                {sign.difficulty && (
                  <span className="rounded-full bg-zinc-800 px-2 py-0.5 text-xs text-zinc-400 ring-1 ring-zinc-700/50">
                    {sign.difficulty}
                  </span>
                )}
              </div>
            )}

            {sign.description && (
              <p className="text-xs leading-relaxed text-zinc-400 line-clamp-3">
                {sign.description}
              </p>
            )}

            {(sign.handshape || sign.location) && (
              <div className="space-y-0.5 text-xs text-zinc-500">
                {sign.handshape && (
                  <p className="flex items-center gap-1.5">
                    <Fingerprint className="h-3 w-3 text-zinc-600" />
                    <span className="font-medium text-zinc-400">{sign.handshape}</span>
                  </p>
                )}
                {sign.location && (
                  <p className="flex items-center gap-1.5">
                    <MapPin className="h-3 w-3 text-zinc-600" />
                    <span className="font-medium text-zinc-400">{sign.location}</span>
                  </p>
                )}
              </div>
            )}

            {sign.synonyms && sign.synonyms.length > 0 && (
              <div>
                <p className="mb-1 text-xs font-medium text-zinc-500">Also known as:</p>
                <div className="flex flex-wrap gap-1">
                  {sign.synonyms.slice(0, 5).map((syn) => (
                    <span
                      key={syn}
                      className="rounded bg-zinc-800/80 px-1.5 py-0.5 text-xs text-zinc-400"
                    >
                      {syn}
                    </span>
                  ))}
                </div>
              </div>
            )}
          </>
        )}
      </div>
    </motion.article>
  )
}
