import { ScanSearch } from 'lucide-react'

export default function Loader() {
  return (
    <div className="flex items-center gap-3 rounded-md border border-slate-200 bg-white px-4 py-3 shadow-sm">
      <div className="grid h-9 w-9 place-items-center rounded-md bg-sky-50 text-sky-700">
        <ScanSearch className="h-5 w-5 animate-pulse" aria-hidden="true" />
      </div>
      <div>
        <p className="text-sm font-semibold text-slate-950">Analyzing image...</p>
        <p className="text-xs text-slate-500">Running pixel, forensic, and semantic streams.</p>
      </div>
    </div>
  )
}
