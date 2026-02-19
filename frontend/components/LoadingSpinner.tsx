import { Loader2 } from 'lucide-react'

interface LoadingSpinnerProps {
  label?: string
  size?: 'sm' | 'md' | 'lg'
}

const SIZE_CLASSES = {
  sm: 'h-5 w-5',
  md: 'h-8 w-8',
  lg: 'h-12 w-12',
}

export function LoadingSpinner({
  label = 'Loading...',
  size = 'md',
}: LoadingSpinnerProps) {
  return (
    <div className="flex flex-col items-center gap-3" role="status" aria-live="polite">
      <Loader2
        className={`animate-spin text-indigo-400 ${SIZE_CLASSES[size]}`}
        aria-hidden="true"
      />
      <span className="text-sm text-zinc-400">{label}</span>
    </div>
  )
}
