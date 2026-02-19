'use client'

import {
  useState,
  useRef,
  useEffect,
  useCallback,
  type KeyboardEvent,
  type FormEvent,
} from 'react'
import { motion, AnimatePresence } from 'framer-motion'
import { Search, Loader2, Command } from 'lucide-react'
import { getSuggestions } from '@/lib/api'
import { cn } from '@/lib/utils'

interface SearchBarProps {
  onSubmit: (text: string) => void
  isLoading?: boolean
}

export function SearchBar({ onSubmit, isLoading = false }: SearchBarProps) {
  const [text, setText] = useState('')
  const [suggestions, setSuggestions] = useState<string[]>([])
  const [showSuggestions, setShowSuggestions] = useState(false)
  const [activeSuggestion, setActiveSuggestion] = useState(-1)
  const [isFocused, setIsFocused] = useState(false)
  const textareaRef = useRef<HTMLTextAreaElement>(null)
  const debounceRef = useRef<ReturnType<typeof setTimeout> | null>(null)

  // Auto-resize textarea
  useEffect(() => {
    const ta = textareaRef.current
    if (!ta) return
    ta.style.height = 'auto'
    ta.style.height = `${ta.scrollHeight}px`
  }, [text])

  const fetchSuggestions = useCallback(async (value: string) => {
    if (value.trim().length < 3) {
      setSuggestions([])
      return
    }
    const results = await getSuggestions(value)
    setSuggestions(results)
    setShowSuggestions(results.length > 0)
  }, [])

  const handleChange = (value: string) => {
    setText(value)
    setActiveSuggestion(-1)
    if (debounceRef.current) clearTimeout(debounceRef.current)
    debounceRef.current = setTimeout(() => fetchSuggestions(value), 350)
  }

  const handleSubmit = (e?: FormEvent) => {
    e?.preventDefault()
    if (!text.trim() || isLoading) return
    setShowSuggestions(false)
    setSuggestions([])
    onSubmit(text)
  }

  const selectSuggestion = (s: string) => {
    setText(s)
    setSuggestions([])
    setShowSuggestions(false)
    textareaRef.current?.focus()
  }

  const handleKeyDown = (e: KeyboardEvent<HTMLTextAreaElement>) => {
    if (!showSuggestions) {
      if (e.key === 'Enter' && (e.metaKey || e.ctrlKey)) {
        e.preventDefault()
        handleSubmit()
      }
      return
    }

    if (e.key === 'ArrowDown') {
      e.preventDefault()
      setActiveSuggestion((prev) => Math.min(prev + 1, suggestions.length - 1))
    } else if (e.key === 'ArrowUp') {
      e.preventDefault()
      setActiveSuggestion((prev) => Math.max(prev - 1, -1))
    } else if (e.key === 'Enter' && activeSuggestion >= 0) {
      e.preventDefault()
      selectSuggestion(suggestions[activeSuggestion])
    } else if (e.key === 'Escape') {
      setShowSuggestions(false)
      setActiveSuggestion(-1)
    }
  }

  return (
    <form onSubmit={handleSubmit} className="relative mx-auto max-w-3xl">
      <div
        className={cn(
          'relative flex items-start rounded-2xl border bg-zinc-900/80 backdrop-blur-sm transition-all duration-300',
          isFocused
            ? 'border-indigo-500/50 shadow-[0_0_0_1px_hsl(224_76%_48%/0.5),0_0_20px_hsl(224_76%_48%/0.15),0_0_40px_hsl(224_76%_48%/0.05)]'
            : 'border-zinc-700/60 shadow-lg shadow-black/20'
        )}
      >
        {/* Search icon */}
        <div className="flex items-center pl-4 pt-3.5">
          <Search className="h-5 w-5 text-zinc-500" aria-hidden="true" />
        </div>

        {/* Textarea */}
        <textarea
          ref={textareaRef}
          value={text}
          onChange={(e) => handleChange(e.target.value)}
          onKeyDown={handleKeyDown}
          onBlur={() => {
            setIsFocused(false)
            setTimeout(() => setShowSuggestions(false), 150)
          }}
          onFocus={() => {
            setIsFocused(true)
            suggestions.length > 0 && setShowSuggestions(true)
          }}
          placeholder="Type an English phrase... e.g. Hello, how are you?"
          rows={2}
          maxLength={500}
          disabled={isLoading}
          aria-label="Enter text to translate to Auslan"
          aria-autocomplete="list"
          aria-controls="suggestions-list"
          aria-activedescendant={
            activeSuggestion >= 0 ? `suggestion-${activeSuggestion}` : undefined
          }
          className={cn(
            'min-h-[3rem] flex-1 resize-none bg-transparent px-3 py-3 text-base',
            'text-zinc-100 placeholder-zinc-500 focus:outline-none disabled:opacity-60'
          )}
        />

        {/* Submit button */}
        <button
          type="submit"
          disabled={!text.trim() || isLoading}
          aria-label="Translate to Auslan"
          className={cn(
            'm-2 flex items-center gap-2 rounded-xl px-5 py-2.5 text-sm font-semibold transition-all duration-200',
            'bg-indigo-600 text-white hover:bg-indigo-500',
            'disabled:cursor-not-allowed disabled:opacity-40',
            'shadow-lg shadow-indigo-500/20 hover:shadow-indigo-500/30'
          )}
        >
          {isLoading ? (
            <>
              <Loader2 className="h-4 w-4 animate-spin" aria-hidden="true" />
              Processing
            </>
          ) : (
            <>
              <Search className="h-4 w-4" aria-hidden="true" />
              Translate
            </>
          )}
        </button>
      </div>

      {/* Character counter */}
      {text.length > 400 && (
        <p className="mt-1 text-right text-xs text-zinc-500">
          {text.length} / 500
        </p>
      )}

      {/* Keyboard hint */}
      <div className="mt-2 flex items-center justify-center gap-1.5 text-xs text-zinc-500">
        <kbd className="inline-flex h-5 items-center gap-0.5 rounded border border-zinc-700 bg-zinc-800 px-1.5 font-mono text-[10px] text-zinc-400">
          <Command className="h-2.5 w-2.5" />
          Enter
        </kbd>
        <span>to translate</span>
      </div>

      {/* Autocomplete dropdown */}
      <AnimatePresence>
        {showSuggestions && suggestions.length > 0 && (
          <motion.ul
            initial={{ opacity: 0, y: -4 }}
            animate={{ opacity: 1, y: 0 }}
            exit={{ opacity: 0, y: -4 }}
            transition={{ duration: 0.15 }}
            id="suggestions-list"
            role="listbox"
            aria-label="Autocomplete suggestions"
            className="absolute left-0 right-0 top-full z-50 mt-2 overflow-hidden rounded-xl border border-zinc-700/60 bg-zinc-900/95 shadow-xl shadow-black/30 backdrop-blur-xl"
          >
            {suggestions.map((s, i) => (
              <li
                key={s}
                id={`suggestion-${i}`}
                role="option"
                aria-selected={i === activeSuggestion}
                onMouseDown={() => selectSuggestion(s)}
                className={cn(
                  'cursor-pointer px-4 py-2.5 text-sm transition-colors',
                  i === activeSuggestion
                    ? 'bg-indigo-500/20 text-indigo-200'
                    : 'text-zinc-300 hover:bg-white/[0.04]'
                )}
              >
                {s}
              </li>
            ))}
          </motion.ul>
        )}
      </AnimatePresence>
    </form>
  )
}
