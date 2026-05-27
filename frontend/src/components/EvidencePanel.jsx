import { AlertTriangle, GitBranch, Radar, ScanSearch } from 'lucide-react'

export default function EvidencePanel({ result }) {
  if (!result) return null

  const fusion = result.evidenceFusion || {}
  const uncertainty = result.uncertainty || {}
  const agents = result.semanticAgents || []
  const topEvidence = result.topEvidence || []
  const routed = fusion?.mixture_of_experts?.primary || []

  return (
    <section className="space-y-3">
      <div className="grid gap-3 sm:grid-cols-3">
        <MetricCard
          icon={Radar}
          label="OOD"
          value={toPercent(uncertainty.ood_score)}
          tone={Number(uncertainty.ood_score) >= 0.72 ? 'amber' : 'slate'}
        />
        <MetricCard
          icon={AlertTriangle}
          label="Uncertainty"
          value={toPercent(uncertainty.overall_uncertainty)}
          tone={Number(uncertainty.overall_uncertainty) >= 0.66 ? 'amber' : 'slate'}
        />
        <MetricCard
          icon={GitBranch}
          label="Disagreement"
          value={toPercent(fusion.disagreement ?? uncertainty.disagreement)}
          tone={Number(fusion.disagreement ?? uncertainty.disagreement) >= 0.55 ? 'amber' : 'slate'}
        />
      </div>

      {routed.length ? (
        <div className="rounded-md border border-slate-200 bg-white p-4">
          <div className="mb-3 flex items-center gap-2">
            <ScanSearch className="h-4 w-4 text-slate-600" aria-hidden="true" />
            <h3 className="text-sm font-semibold text-slate-950">Routed Evidence</h3>
          </div>
          <div className="grid gap-2">
            {routed.map((item) => (
              <div key={item.stream} className="grid grid-cols-[96px_1fr_52px] items-center gap-2">
                <span className="text-xs font-medium capitalize text-slate-600">{item.stream}</span>
                <div className="h-2 rounded-full bg-slate-200">
                  <div
                    className="h-2 rounded-full bg-indigo-600"
                    style={{ width: `${toPercent(item.route_score)}%` }}
                  />
                </div>
                <span className="text-right text-xs font-semibold text-slate-800">
                  {toPercent(item.evidence).toFixed(0)}%
                </span>
              </div>
            ))}
          </div>
        </div>
      ) : null}

      {topEvidence.length ? (
        <div className="rounded-md border border-slate-200 bg-white p-4">
          <h3 className="text-sm font-semibold text-slate-950">Top Evidence</h3>
          <div className="mt-3 grid gap-2 sm:grid-cols-2">
            {topEvidence.slice(0, 8).map((item) => (
              <div key={item.name} className="flex items-center justify-between gap-3 rounded-md bg-slate-50 px-3 py-2">
                <span className="min-w-0 truncate text-xs font-medium text-slate-600">
                  {formatName(item.name)}
                </span>
                <span className="text-xs font-bold text-slate-900">{toPercent(item.value).toFixed(0)}%</span>
              </div>
            ))}
          </div>
        </div>
      ) : null}

      {agents.length ? (
        <div className="rounded-md border border-slate-200 bg-white p-4">
          <h3 className="text-sm font-semibold text-slate-950">Semantic Agents</h3>
          <div className="mt-3 space-y-2">
            {agents.slice(0, 5).map((agent) => (
              <div key={agent.agent} className="rounded-md bg-slate-50 px-3 py-2">
                <div className="flex items-center justify-between gap-2">
                  <p className="text-xs font-semibold capitalize text-slate-800">{formatName(agent.agent)}</p>
                  <p className="text-xs font-bold text-slate-900">{toPercent(agent.value).toFixed(0)}%</p>
                </div>
                {agent.reasoning ? (
                  <p className="mt-1 text-xs leading-5 text-slate-500">{agent.reasoning}</p>
                ) : null}
              </div>
            ))}
          </div>
        </div>
      ) : null}
    </section>
  )
}

function MetricCard({ icon: Icon, label, value, tone }) {
  const classes =
    tone === 'amber'
      ? 'border-amber-200 bg-amber-50 text-amber-800'
      : 'border-slate-200 bg-slate-50 text-slate-700'
  return (
    <div className={`rounded-md border p-3 ${classes}`}>
      <div className="flex items-center gap-2">
        <Icon className="h-4 w-4" aria-hidden="true" />
        <span className="text-xs font-semibold uppercase tracking-normal">{label}</span>
      </div>
      <p className="mt-2 text-xl font-bold">{Number.isFinite(value) ? `${value.toFixed(0)}%` : '0%'}</p>
    </div>
  )
}

function toPercent(value) {
  const number = Number(value)
  if (!Number.isFinite(number)) return 0
  return Math.min(100, Math.max(0, number <= 1 ? number * 100 : number))
}

function formatName(value) {
  return String(value || '').replaceAll('_', ' ')
}
