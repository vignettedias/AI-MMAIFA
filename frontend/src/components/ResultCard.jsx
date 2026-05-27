import { AlertTriangle, CheckCircle2, RotateCcw, ShieldAlert } from 'lucide-react'
import EvidencePanel from './EvidencePanel'
import ScoreBreakdown from './ScoreBreakdown'

export default function ResultCard({
  result,
  onReset,
  onFeedback,
  feedbackStatus,
  adminKey,
  onAdminKeyChange,
}) {
  if (!result) return null

  const lowConfidence = result.confidence < 60
  const isReal = result.label === 'REAL'
  const isReview = result.label === 'HUMAN REVIEW' || result.label === 'UNKNOWN'
  const tone = isReview ? 'yellow' : isReal ? (lowConfidence ? 'yellow' : 'green') : 'red'
  const Icon = isReview ? AlertTriangle : isReal ? (lowConfidence ? AlertTriangle : CheckCircle2) : ShieldAlert
  const badgeText = result.decisionBand
    ? result.decisionBand.replaceAll('_', ' ')
    : lowConfidence
      ? 'Low confidence'
      : 'Meta-classifier'

  const toneClasses = {
    green: {
      badge: 'bg-emerald-50 text-emerald-700 ring-emerald-200',
      bar: 'bg-emerald-600',
      icon: 'bg-emerald-50 text-emerald-700',
    },
    red: {
      badge: 'bg-rose-50 text-rose-700 ring-rose-200',
      bar: 'bg-rose-600',
      icon: 'bg-rose-50 text-rose-700',
    },
    yellow: {
      badge: 'bg-amber-50 text-amber-800 ring-amber-200',
      bar: 'bg-amber-500',
      icon: 'bg-amber-50 text-amber-700',
    },
  }[tone]

  return (
    <div className="space-y-5 rounded-md border border-slate-200 bg-white p-5 shadow-sm">
      <div className="flex flex-col gap-4 sm:flex-row sm:items-start sm:justify-between">
        <div className="flex gap-3">
          <div className={`grid h-11 w-11 place-items-center rounded-md ${toneClasses.icon}`}>
            <Icon className="h-6 w-6" aria-hidden="true" />
          </div>
          <div>
            <p className="text-xs font-semibold uppercase tracking-normal text-slate-500">Final Decision</p>
            <div className="mt-1 flex flex-wrap items-center gap-2">
              <h2 className="text-2xl font-bold text-slate-950">{result.label}</h2>
              <span className={`rounded-full px-2.5 py-1 text-xs font-semibold ring-1 ${toneClasses.badge}`}>
                {badgeText}
              </span>
            </div>
            {result.decisionMessage ? (
              <p className="mt-1 max-w-xl text-xs leading-5 text-slate-500">{result.decisionMessage}</p>
            ) : null}
          </div>
        </div>
        <button
          type="button"
          onClick={onReset}
          className="inline-flex items-center justify-center gap-2 rounded-md border border-slate-300 bg-white px-3 py-2 text-sm font-semibold text-slate-700 shadow-sm transition hover:bg-slate-50"
        >
          <RotateCcw className="h-4 w-4" aria-hidden="true" />
          New image
        </button>
      </div>

      <div>
        <div className="mb-2 flex items-center justify-between">
          <span className="text-sm font-medium text-slate-600">Confidence</span>
          <span className="text-sm font-bold text-slate-950">{result.confidence.toFixed(1)}%</span>
        </div>
        <div className="h-3 overflow-hidden rounded-full bg-slate-200">
          <div
            className={`h-3 rounded-full transition-all duration-700 ${toneClasses.bar}`}
            style={{ width: `${result.confidence}%` }}
          />
        </div>
        {result.metaDecision?.strategy === 'weighted_profile_guard' ? (
          <p className="mt-2 text-xs leading-5 text-slate-500">
            Demo XGBoost calibration was guarded by the weighted multi-stream evidence profile.
          </p>
        ) : null}
        {result.dynamicMemory?.strategy && result.dynamicMemory.strategy !== 'no_memory' ? (
          <p className="mt-2 text-xs leading-5 text-slate-500">
            Dynamic memory: {result.dynamicMemory.strategy.replaceAll('_', ' ')}.
          </p>
        ) : null}
        {result.autoMemory?.status ? (
          <p className="mt-2 text-xs leading-5 text-slate-500">
            Auto-learning: {result.autoMemory.status.replaceAll('_', ' ')}.
          </p>
        ) : null}
        {result.reviewDecision?.reasons?.length ? (
          <p className="mt-2 text-xs leading-5 text-slate-500">
            Review policy: {result.reviewDecision.reasons.join(' ')}
          </p>
        ) : null}
        {Number.isFinite(Number(result.uncertainty?.ood_score)) ? (
          <p className="mt-2 text-xs leading-5 text-slate-500">
            OOD {(Number(result.uncertainty.ood_score) * 100).toFixed(0)}% / Reliability{' '}
            {(Number(result.uncertainty.reliability || 0) * 100).toFixed(0)}%
          </p>
        ) : null}
      </div>

      <ScoreBreakdown breakdown={result.breakdown} />
      <EvidencePanel result={result} />

      <div className="rounded-md border border-slate-200 bg-slate-50 p-4">
        <p className="text-sm font-semibold text-slate-950">Teach the detector</p>
        <p className="mt-1 text-xs leading-5 text-slate-500">
          Admin/creator only. Enter the admin key before saving a correction.
        </p>
        <input
          type="password"
          value={adminKey}
          onChange={(event) => onAdminKeyChange?.(event.target.value)}
          placeholder="Admin key"
          className="mt-3 w-full rounded-md border border-slate-300 bg-white px-3 py-2 text-sm text-slate-950 outline-none transition focus:border-sky-500 focus:ring-2 focus:ring-sky-100"
        />
        <div className="mt-3 grid gap-2 sm:grid-cols-2">
          <button
            type="button"
            onClick={() => onFeedback?.('REAL')}
            disabled={!adminKey}
            className="rounded-md border border-emerald-200 bg-white px-3 py-2 text-sm font-semibold text-emerald-700 transition hover:bg-emerald-50 disabled:cursor-not-allowed disabled:opacity-50"
          >
            Correct: REAL
          </button>
          <button
            type="button"
            onClick={() => onFeedback?.('FAKE')}
            disabled={!adminKey}
            className="rounded-md border border-rose-200 bg-white px-3 py-2 text-sm font-semibold text-rose-700 transition hover:bg-rose-50 disabled:cursor-not-allowed disabled:opacity-50"
          >
            Correct: AI-GENERATED
          </button>
        </div>
        {feedbackStatus ? (
          <p className="mt-3 text-xs font-medium text-slate-600">{feedbackStatus}</p>
        ) : null}
      </div>
    </div>
  )
}
