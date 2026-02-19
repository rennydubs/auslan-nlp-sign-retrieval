import type { Metadata } from 'next'

export const metadata: Metadata = {
  title: 'About',
  description:
    'Learn how the Auslan Sign Retrieval system works, including the NLP matching pipeline and the Auslan Signbank dataset.',
}

// ---------------------------------------------------------------------------
// Static about page — Server Component (no 'use client')
// ---------------------------------------------------------------------------

export default function AboutPage() {
  return (
    <div className="mx-auto max-w-4xl px-4 py-12 sm:px-6 lg:px-8">
      <h1 className="mb-2 text-4xl font-bold tracking-tight text-slate-900 dark:text-white">
        About Auslan Sign Retrieval
      </h1>
      <p className="mb-10 text-lg text-slate-600 dark:text-slate-400">
        An NLP-powered system that bridges written English and Australian Sign Language.
      </p>

      {/* What is Auslan? */}
      <Section title="What is Auslan?">
        <p>
          Auslan (Australian Sign Language) is the primary language of the Australian Deaf
          community. It is a natural, complete language with its own grammar, vocabulary and
          syntax&mdash;distinct from spoken English and from other sign languages such as ASL
          or BSL.
        </p>
        <p className="mt-3">
          This system helps learners, interpreters, and Deaf-accessible application developers
          find Auslan sign videos for common English phrases.
        </p>
      </Section>

      {/* What the system does */}
      <Section title="What the system does">
        <p>
          You type a phrase in English. The system tokenises and preprocesses the input, then
          runs each token through a five-stage matching pipeline to find the best Auslan sign
          available in the dictionary.
        </p>
        <ul className="mt-3 list-inside list-disc space-y-1 text-slate-700 dark:text-slate-300">
          <li>Matching is word-by-word, respecting Auslan gloss conventions.</li>
          <li>Each matched sign links to an MP4 video served from the local media store.</li>
          <li>An NLP layer provides sentiment, emotion, intent and entity analysis.</li>
          <li>Coverage statistics show how many tokens were successfully matched.</li>
        </ul>
      </Section>

      {/* Matching pipeline */}
      <Section title="The matching pipeline">
        <p className="mb-4">
          Matches are attempted in priority order. Once a match is found at any stage the
          pipeline short-circuits for that token.
        </p>
        <ol className="space-y-4">
          <PipelineStep
            step={1}
            label="Exact match"
            badge="green"
            badgeText="100% confidence"
            description="The lowercased token is looked up directly in the gloss dictionary. If the key exists, this is the highest-quality result."
          />
          <PipelineStep
            step={2}
            label="Fuzzy match"
            badge="yellow"
            badgeText="variable confidence"
            description="Edit-distance (Levenshtein) comparison is used to catch common spelling errors or slight inflectional variations."
          />
          <PipelineStep
            step={3}
            label="Synonym match"
            badge="blue"
            badgeText="90% confidence"
            description="The token is checked against an external synonym mapping (synonym_mapping.json) and the synonyms listed in each dictionary entry. The canonical gloss is returned."
          />
          <PipelineStep
            step={4}
            label="Semantic match"
            badge="purple"
            badgeText="threshold-filtered"
            description="When spaCy and the all-MiniLM-L6-v2 sentence-transformer model are available, cosine similarity is computed between the token embedding and pre-computed dictionary embeddings. Matches above the configurable threshold are returned."
          />
          <PipelineStep
            step={5}
            label="LLM fallback"
            badge="orange"
            badgeText="optional"
            description="If enabled, an LLM is consulted as a last resort to suggest the most appropriate sign gloss for words not found by earlier stages."
          />
        </ol>
      </Section>

      {/* Dataset */}
      <Section title="Dataset — Auslan Signbank">
        <p>
          The sign dictionary is sourced from{' '}
          <a
            href="https://www.auslan.org.au"
            target="_blank"
            rel="noopener noreferrer"
            className="text-blue-600 underline hover:text-blue-700 dark:text-blue-400 dark:hover:text-blue-300"
          >
            Auslan Signbank
          </a>
          , the authoritative online dictionary for Australian Sign Language maintained by
          Macquarie University. The current dictionary contains approximately 46 signs spanning
          categories such as greetings, emotions, fitness, daily activities, and body parts.
        </p>
        <p className="mt-3">
          Video files are stored locally using Git LFS and served directly from the FastAPI
          backend at{' '}
          <code className="rounded bg-slate-100 px-1.5 py-0.5 text-sm dark:bg-slate-800">
            /media/videos/
          </code>
          .
        </p>
      </Section>

      {/* Tech stack */}
      <Section title="Tech stack">
        <div className="grid gap-4 sm:grid-cols-2">
          <TechCard title="Backend" items={[
            'Python 3.8+',
            'FastAPI + Uvicorn',
            'spaCy (en_core_web_sm)',
            'sentence-transformers (all-MiniLM-L6-v2)',
            'TextBlob (fallback sentiment)',
          ]} />
          <TechCard title="Frontend" items={[
            'Next.js 14 App Router',
            'React 18',
            'TypeScript',
            'Tailwind CSS v3',
          ]} />
          <TechCard title="NLP features" items={[
            'Sentiment analysis',
            'Emotion detection',
            'Named entity recognition',
            'Intent classification',
            'Formality scoring',
          ]} />
          <TechCard title="Matching" items={[
            'Dictionary exact lookup',
            'Fuzzy (edit-distance)',
            'Synonym expansion',
            'Semantic cosine similarity',
            'LLM-assisted fallback (optional)',
          ]} />
        </div>
      </Section>

      {/* Graceful degradation note */}
      <Section title="Graceful degradation">
        <p>
          The system is designed to run even when heavy ML dependencies are not installed. If
          spaCy or the sentence-transformer model is unavailable, it falls back to TextBlob for
          sentiment and regex-based entity detection. The exact and synonym matching tiers always
          function regardless of optional dependencies.
        </p>
      </Section>
    </div>
  )
}

// ---------------------------------------------------------------------------
// Local sub-components (no separate files needed — static page only)
// ---------------------------------------------------------------------------

function Section({ title, children }: { title: string; children: React.ReactNode }) {
  return (
    <section className="mb-10">
      <h2 className="mb-4 text-2xl font-semibold text-slate-900 dark:text-white">
        {title}
      </h2>
      <div className="text-slate-700 dark:text-slate-300 leading-relaxed">{children}</div>
    </section>
  )
}

const BADGE_COLORS: Record<string, string> = {
  green:  'bg-green-100 text-green-700 dark:bg-green-900 dark:text-green-300',
  yellow: 'bg-yellow-100 text-yellow-700 dark:bg-yellow-900 dark:text-yellow-300',
  blue:   'bg-blue-100 text-blue-700 dark:bg-blue-900 dark:text-blue-300',
  purple: 'bg-purple-100 text-purple-700 dark:bg-purple-900 dark:text-purple-300',
  orange: 'bg-orange-100 text-orange-700 dark:bg-orange-900 dark:text-orange-300',
}

interface PipelineStepProps {
  step: number
  label: string
  badge: string
  badgeText: string
  description: string
}

function PipelineStep({ step, label, badge, badgeText, description }: PipelineStepProps) {
  return (
    <li className="flex gap-4">
      <div className="flex h-8 w-8 flex-shrink-0 items-center justify-center rounded-full bg-slate-200 text-sm font-bold text-slate-700 dark:bg-slate-700 dark:text-slate-300">
        {step}
      </div>
      <div>
        <div className="mb-1 flex flex-wrap items-center gap-2">
          <span className="font-semibold text-slate-900 dark:text-white">{label}</span>
          <span
            className={`rounded-full px-2 py-0.5 text-xs font-semibold ${BADGE_COLORS[badge] ?? BADGE_COLORS.blue}`}
          >
            {badgeText}
          </span>
        </div>
        <p className="text-sm text-slate-600 dark:text-slate-400">{description}</p>
      </div>
    </li>
  )
}

function TechCard({ title, items }: { title: string; items: string[] }) {
  return (
    <div className="rounded-xl border border-slate-200 bg-white p-4 shadow-sm dark:border-slate-700 dark:bg-slate-900">
      <h3 className="mb-3 text-sm font-semibold uppercase tracking-wider text-slate-500 dark:text-slate-400">
        {title}
      </h3>
      <ul className="space-y-1">
        {items.map((item) => (
          <li key={item} className="flex items-center gap-2 text-sm text-slate-700 dark:text-slate-300">
            <span className="h-1.5 w-1.5 rounded-full bg-blue-500 flex-shrink-0" />
            {item}
          </li>
        ))}
      </ul>
    </div>
  )
}
