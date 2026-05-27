import { Activity, Brain, Layers3, Microscope } from 'lucide-react'

const icons = {
  'Pixel Ensemble': Layers3,
  'Forensic Signals': Microscope,
  'Semantic Reasoning': Brain,
}

export default function ScoreBreakdown({ breakdown }) {
  const streams = breakdown?.streams || []
  const modelScores = breakdown?.modelScores || []

  if (!streams.length && !modelScores.length) {
    return null
  }

  return (
    <section className="space-y-4">
      {streams.length ? (
        <div className="grid gap-3 sm:grid-cols-3">
          {streams.map((stream) => {
            const Icon = icons[stream.name] || Activity
            return (
              <div key={stream.name} className="rounded-md border border-slate-200 bg-slate-50 p-3">
                <div className="flex items-center gap-2">
                  <Icon className="h-4 w-4 text-slate-600" aria-hidden="true" />
                  <p className="text-xs font-semibold uppercase tracking-normal text-slate-600">{stream.name}</p>
                </div>
                <div className="mt-3 h-2 rounded-full bg-slate-200">
                  <div
                    className="h-2 rounded-full bg-sky-600 transition-all duration-700"
                    style={{ width: `${stream.value}%` }}
                  />
                </div>
                <div className="mt-2 flex items-baseline justify-between gap-2">
                  <p className="text-xl font-bold text-slate-950">{stream.value.toFixed(1)}%</p>
                  <p className="text-right text-[11px] leading-4 text-slate-500">{stream.description}</p>
                </div>
              </div>
            )
          })}
        </div>
      ) : null}

      {modelScores.length ? (
        <div className="rounded-md border border-slate-200 bg-white p-4">
          <div className="mb-3 flex items-center gap-2">
            <Activity className="h-4 w-4 text-slate-600" aria-hidden="true" />
            <h3 className="text-sm font-semibold text-slate-950">Model Evidence</h3>
          </div>
          <div className="grid gap-2 sm:grid-cols-2">
            {modelScores.map((score) => (
              <div key={score.name} className="grid grid-cols-[88px_1fr_48px] items-center gap-2">
                <span className="text-xs font-medium text-slate-600">{score.name}</span>
                <div className="h-2 rounded-full bg-slate-200">
                  <div
                    className="h-2 rounded-full bg-slate-700 transition-all duration-700"
                    style={{ width: `${score.value}%` }}
                  />
                </div>
                <span className="text-right text-xs font-semibold text-slate-800">{score.value.toFixed(0)}%</span>
              </div>
            ))}
          </div>
        </div>
      ) : null}
    </section>
  )
}
