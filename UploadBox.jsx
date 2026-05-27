import { ImageUp, SearchCheck, Upload, X } from 'lucide-react'
import { useRef, useState } from 'react'

const ACCEPTED_TYPES = ['image/jpeg', 'image/png', 'image/webp']

export default function UploadBox({ file, previewUrl, onFileSelect, onAnalyze, onClear, isLoading }) {
  const inputRef = useRef(null)
  const [isDragging, setIsDragging] = useState(false)

  function handleFiles(fileList) {
    const selected = fileList?.[0]
    if (selected) onFileSelect(selected)
  }

  return (
    <section className="rounded-md border border-slate-200 bg-white p-5 shadow-sm">
      <div
        onDragOver={(event) => {
          event.preventDefault()
          setIsDragging(true)
        }}
        onDragLeave={() => setIsDragging(false)}
        onDrop={(event) => {
          event.preventDefault()
          setIsDragging(false)
          handleFiles(event.dataTransfer.files)
        }}
        className={`grid min-h-72 place-items-center rounded-md border-2 border-dashed p-4 text-center transition ${
          isDragging ? 'border-sky-500 bg-sky-50' : 'border-slate-300 bg-slate-50'
        }`}
      >
        {previewUrl ? (
          <div className="w-full space-y-4">
            <div className="relative mx-auto aspect-[4/3] w-full max-w-xl overflow-hidden rounded-md border border-slate-200 bg-slate-100">
              <img src={previewUrl} alt="Selected upload preview" className="h-full w-full object-contain" />
              <button
                type="button"
                onClick={onClear}
                className="absolute right-3 top-3 grid h-9 w-9 place-items-center rounded-md bg-white text-slate-700 shadow transition hover:bg-slate-100"
                aria-label="Remove selected image"
              >
                <X className="h-4 w-4" aria-hidden="true" />
              </button>
            </div>
            <div className="flex flex-col items-center justify-between gap-3 sm:flex-row">
              <div className="min-w-0 text-left">
                <p className="truncate text-sm font-semibold text-slate-950">{file?.name}</p>
                <p className="text-xs text-slate-500">{formatFileSize(file?.size)} selected for multi-stream analysis</p>
              </div>
              <button
                type="button"
                onClick={onAnalyze}
                disabled={isLoading}
                className="inline-flex w-full items-center justify-center gap-2 rounded-md bg-slate-950 px-4 py-2.5 text-sm font-semibold text-white shadow-sm transition hover:bg-slate-800 disabled:cursor-not-allowed disabled:bg-slate-400 sm:w-auto"
              >
                <SearchCheck className="h-4 w-4" aria-hidden="true" />
                {isLoading ? 'Analyzing...' : 'Analyze image'}
              </button>
            </div>
          </div>
        ) : (
          <div className="mx-auto max-w-md">
            <div className="mx-auto grid h-14 w-14 place-items-center rounded-md bg-sky-50 text-sky-700">
              <ImageUp className="h-7 w-7" aria-hidden="true" />
            </div>
            <h2 className="mt-4 text-xl font-bold text-slate-950">Upload an image for authenticity analysis</h2>
            <p className="mt-2 text-sm leading-6 text-slate-600">
              Drop a JPEG, PNG, JPG, or WEBP image here. The backend will combine pixel-level, forensic, and semantic evidence.
            </p>
            <button
              type="button"
              onClick={() => inputRef.current?.click()}
              className="mt-5 inline-flex items-center justify-center gap-2 rounded-md bg-slate-950 px-4 py-2.5 text-sm font-semibold text-white shadow-sm transition hover:bg-slate-800"
            >
              <Upload className="h-4 w-4" aria-hidden="true" />
              Choose image
            </button>
          </div>
        )}
      </div>

      <input
        ref={inputRef}
        type="file"
        accept={ACCEPTED_TYPES.join(',')}
        className="hidden"
        onChange={(event) => handleFiles(event.target.files)}
      />
    </section>
  )
}

function formatFileSize(size = 0) {
  if (!size) return '0 KB'
  if (size < 1024 * 1024) return `${(size / 1024).toFixed(1)} KB`
  return `${(size / (1024 * 1024)).toFixed(2)} MB`
}
