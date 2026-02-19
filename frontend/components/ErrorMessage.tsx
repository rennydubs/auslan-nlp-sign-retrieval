import { AlertTriangle } from 'lucide-react'

interface ErrorMessageProps {
  message: string
  detail?: string
}

export function ErrorMessage({ message, detail }: ErrorMessageProps) {
  return (
    <div
      role="alert"
      className="flex items-start gap-3 rounded-2xl border border-red-500/20 bg-red-500/[0.08] px-5 py-4 text-sm backdrop-blur-sm"
    >
      <AlertTriangle className="h-5 w-5 flex-shrink-0 text-red-400" aria-hidden="true" />
      <div>
        <p className="font-semibold text-red-300">{message}</p>
        {detail && (
          <p className="mt-1 text-red-400/80">{detail}</p>
        )}
      </div>
    </div>
  )
}
