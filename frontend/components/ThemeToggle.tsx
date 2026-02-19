'use client'

import { Sun, Moon } from 'lucide-react'
import { useTheme } from './ThemeProvider'
import { cn } from '@/lib/utils'

export function ThemeToggle() {
  const { theme, toggleTheme } = useTheme()

  return (
    <button
      onClick={toggleTheme}
      aria-label={theme === 'dark' ? 'Switch to light mode' : 'Switch to dark mode'}
      title={theme === 'dark' ? 'Switch to light mode' : 'Switch to dark mode'}
      className={cn(
        'flex h-8 w-8 items-center justify-center rounded-lg',
        'text-zinc-400 transition-colors duration-200',
        'hover:bg-white/[0.06] hover:text-zinc-100'
      )}
    >
      {theme === 'dark' ? <Sun className="h-4 w-4" /> : <Moon className="h-4 w-4" />}
    </button>
  )
}
