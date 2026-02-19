'use client'

import Link from 'next/link'
import { Hand, Github, BookOpen } from 'lucide-react'
import { ThemeToggle } from './ThemeToggle'
import { cn } from '@/lib/utils'

export function Navbar() {
  return (
    <header className="sticky top-0 z-40 glass border-b border-white/[0.06]">
      <nav
        className="mx-auto flex max-w-7xl items-center justify-between px-4 py-3 sm:px-6 lg:px-8"
        aria-label="Main navigation"
      >
        {/* Logo + brand */}
        <Link
          href="/"
          className="flex items-center gap-2.5 text-zinc-100 transition-opacity hover:opacity-80"
          aria-label="Auslan Sign Retrieval — home"
        >
          <div className="flex h-8 w-8 items-center justify-center rounded-lg bg-indigo-500/20">
            <Hand className="h-4 w-4 text-indigo-400" />
          </div>
          <span className="text-base font-semibold tracking-tight">
            Auslan<span className="text-gradient ml-1">Sign</span>
          </span>
        </Link>

        {/* Nav icons + toggle */}
        <div className="flex items-center gap-1">
          <NavIconLink href="/about" label="About" icon={<BookOpen className="h-4 w-4" />} />
          <NavIconLink
            href="https://github.com/rennydubs/auslan-nlp-sign-retrieval"
            label="GitHub"
            icon={<Github className="h-4 w-4" />}
            external
          />
          <div className="ml-1 h-4 w-px bg-zinc-700/60" />
          <ThemeToggle />
        </div>
      </nav>
    </header>
  )
}

interface NavIconLinkProps {
  href: string
  label: string
  icon: React.ReactNode
  external?: boolean
}

function NavIconLink({ href, label, icon, external }: NavIconLinkProps) {
  const classes = cn(
    'flex h-8 w-8 items-center justify-center rounded-lg',
    'text-zinc-400 transition-colors duration-200',
    'hover:bg-white/[0.06] hover:text-zinc-100'
  )

  if (external) {
    return (
      <a
        href={href}
        target="_blank"
        rel="noopener noreferrer"
        aria-label={label}
        title={label}
        className={classes}
      >
        {icon}
      </a>
    )
  }

  return (
    <Link href={href} aria-label={label} title={label} className={classes}>
      {icon}
    </Link>
  )
}
