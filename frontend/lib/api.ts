import type {
  DictionaryResponse,
  HealthResponse,
  ModelsStatus,
  ProcessRequest,
  ProcessResponse,
} from './types'

// ---------------------------------------------------------------------------
// Base URL — override with NEXT_PUBLIC_API_URL env var if needed
// ---------------------------------------------------------------------------

export const API_BASE =
  process.env.NEXT_PUBLIC_API_URL ?? 'http://localhost:8000'

// ---------------------------------------------------------------------------
// Generic fetch helper
// ---------------------------------------------------------------------------

async function apiFetch<T>(
  path: string,
  init?: RequestInit,
): Promise<T> {
  const url = `${API_BASE}${path}`
  const res = await fetch(url, {
    ...init,
    headers: {
      'Content-Type': 'application/json',
      ...(init?.headers ?? {}),
    },
  })

  if (!res.ok) {
    let detail = `HTTP ${res.status}`
    try {
      const body = await res.json()
      if (body?.detail) detail = String(body.detail)
    } catch {
      // ignore JSON parse errors — keep the status text
    }
    throw new Error(detail)
  }

  return res.json() as Promise<T>
}

// ---------------------------------------------------------------------------
// API functions
// ---------------------------------------------------------------------------

/**
 * Process natural language text through the sign-matching pipeline.
 */
export async function processText(
  request: ProcessRequest,
): Promise<ProcessResponse> {
  return apiFetch<ProcessResponse>('/api/process', {
    method: 'POST',
    body: JSON.stringify(request),
  })
}

/**
 * Get autocomplete phrase suggestions for partial text input.
 */
export async function getSuggestions(text: string): Promise<string[]> {
  try {
    const data = await apiFetch<{ partial_text: string; suggestions: string[] }>(
      '/api/suggestions',
      {
        method: 'POST',
        body: JSON.stringify({ text }),
      },
    )
    return data.suggestions ?? []
  } catch {
    return []
  }
}

/**
 * Health check — returns uptime and total sign count.
 */
export async function getHealth(): Promise<HealthResponse> {
  return apiFetch<HealthResponse>('/api/health')
}

/**
 * Model / component availability status.
 */
export async function getModelsStatus(): Promise<ModelsStatus> {
  return apiFetch<ModelsStatus>('/api/models/status')
}

/**
 * Full sign dictionary with category breakdown.
 */
export async function getDictionary(): Promise<DictionaryResponse> {
  return apiFetch<DictionaryResponse>('/api/dictionary')
}

/**
 * Build the full URL for a sign video.
 * video_url from the API is a relative path like /media/videos/hello.mp4
 */
export function videoUrl(path: string): string {
  if (!path) return ''
  if (path.startsWith('http://') || path.startsWith('https://')) return path
  return `${API_BASE}${path.startsWith('/') ? path : `/${path}`}`
}
