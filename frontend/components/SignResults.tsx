'use client'

import { motion } from 'framer-motion'
import type { SignMatch } from '@/lib/types'
import { SignCard } from './SignCard'

interface SignResultsProps {
  matches: SignMatch[]
}

export function SignResults({ matches }: SignResultsProps) {
  if (matches.length === 0) return null

  return (
    <section aria-label="Matched signs">
      <div className="mb-4 flex items-center justify-between">
        <h2 className="text-lg font-semibold text-zinc-100">
          Matched signs
        </h2>
        <span className="text-sm text-zinc-500">
          {matches.length} sign{matches.length !== 1 ? 's' : ''} found
        </span>
      </div>

      <motion.div
        initial="hidden"
        animate="visible"
        variants={{
          hidden: {},
          visible: { transition: { staggerChildren: 0.05 } },
        }}
        className="grid grid-cols-1 gap-4 sm:grid-cols-2 lg:grid-cols-3 xl:grid-cols-4"
      >
        {matches.map((match, index) => (
          <SignCard key={`${match.word}-${index}`} match={match} index={index} />
        ))}
      </motion.div>
    </section>
  )
}
