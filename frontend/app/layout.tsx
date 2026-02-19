import type { Metadata } from 'next'
import { Inter } from 'next/font/google'
import './globals.css'
import { Navbar } from '@/components/Navbar'
import { ThemeProvider } from '@/components/ThemeProvider'

const inter = Inter({
  subsets: ['latin'],
  variable: '--font-inter',
  display: 'swap',
})

export const metadata: Metadata = {
  title: {
    default: 'Auslan Sign Retrieval',
    template: '%s | Auslan Sign Retrieval',
  },
  description:
    'NLP-powered system that translates natural language text into matched Auslan (Australian Sign Language) sign videos.',
  keywords: ['Auslan', 'Australian Sign Language', 'NLP', 'sign language', 'accessibility'],
  authors: [{ name: 'Auslan NLP Team' }],
  openGraph: {
    type: 'website',
    locale: 'en_AU',
    title: 'Auslan Sign Retrieval',
    description: 'Translate natural language into Auslan sign videos using NLP.',
    siteName: 'Auslan Sign Retrieval',
  },
  robots: { index: false, follow: false },
}

export default function RootLayout({
  children,
}: {
  children: React.ReactNode
}) {
  return (
    <html lang="en" suppressHydrationWarning className={inter.variable}>
      <head>
        <script
          dangerouslySetInnerHTML={{
            __html: `
              (function() {
                try {
                  var stored = localStorage.getItem('theme');
                  var prefersDark = window.matchMedia('(prefers-color-scheme: dark)').matches;
                  if (stored === 'dark' || (!stored && prefersDark)) {
                    document.documentElement.classList.add('dark');
                  }
                } catch (e) {}
              })();
            `,
          }}
        />
      </head>
      <body className="min-h-screen bg-zinc-950 font-sans text-zinc-100 antialiased">
        <ThemeProvider>
          {/* Ambient background glow */}
          <div className="pointer-events-none fixed inset-0 -z-10 overflow-hidden">
            <div className="absolute -top-40 left-1/2 h-[600px] w-[900px] -translate-x-1/2 rounded-full bg-indigo-500/[0.07] blur-3xl" />
            <div className="absolute -bottom-40 right-0 h-[400px] w-[600px] rounded-full bg-purple-500/[0.05] blur-3xl" />
          </div>

          <div className="flex min-h-screen flex-col">
            <Navbar />
            <main className="flex-1">{children}</main>
            <footer className="border-t border-zinc-800/60 py-6 text-center text-sm text-zinc-500">
              Auslan Sign Retrieval &mdash; NLP-powered translation to Australian Sign Language
            </footer>
          </div>
        </ThemeProvider>
      </body>
    </html>
  )
}
