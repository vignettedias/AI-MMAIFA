import { useEffect, useMemo, useState } from 'react'
import { AlertCircle, Cpu, ShieldCheck, Sparkles } from 'lucide-react'
import { predictImage, submitFeedback } from './api/predict'
import Loader from './components/Loader'
import ResultCard from './components/ResultCard'
import UploadBox from './components/UploadBox'

const ACCEPTED_TYPES = ['image/jpeg', 'image/png', 'image/webp']

function App() {
  const [file, setFile] = useState(null)
  const [result, setResult] = useState(null)
  const [error, setError] = useState('')
  const [isLoading, setIsLoading] = useState(false)
  const [feedbackStatus, setFeedbackStatus] = useState('')
  const [adminKey, setAdminKey] = useState(import.meta.env.VITE_ADMIN_KEY || '')

  const previewUrl = useMemo(() => (file ? URL.createObjectURL(file) : ''), [file])

  useEffect(() => {
    return () => {
      if (previewUrl) URL.revokeObjectURL(previewUrl)
    }
  }, [previewUrl])


  const systemStats = useMemo(
    () => [
      { label: 'Pixel Ensemble', value: 'EfficientNet + AlexNet + GoogLeNet', icon: Cpu },
      { label: 'Forensics', value: 'ELA, FFT, noise consistency', icon: ShieldCheck },
      { label: 'Semantic Reasoning', value: 'Prompt-based anomaly signals', icon: Sparkles },
    ],
    [],
  )

  function handleFileSelect(selectedFile) {
    setError('')
    setResult(null)
    setFeedbackStatus('')

    if (!ACCEPTED_TYPES.includes(selectedFile.type)) {
      setFile(null)
      setError('Unsupported file type. Please upload JPEG, PNG, JPG, or WEBP.')
      return
    }

    setFile(selectedFile)
  }

  async function handleAnalyze() {
    if (!file || isLoading) return

    setIsLoading(true)
    setError('')
    setResult(null)
    setFeedbackStatus('')

    try {
      const prediction = await predictImage(file)
      setResult(prediction)
    } catch (requestError) {
      const message = requestError?.response?.data?.detail || requestError?.message || 'Prediction request failed.'
      setError(`Could not analyze image. ${message}`)
    } finally {
      setIsLoading(false)
    }
  }

  function handleReset() {
    setFile(null)
    setResult(null)
    setError('')
    setFeedbackStatus('')
  }

  async function handleFeedback(label) {
    if (!result) return
    setError('')
    setFeedbackStatus('Saving correction...')

    try {
      const response = await submitFeedback({
        result,
        label,
        note: `User corrected prediction to ${label}`,
        adminKey,
      })
      setFeedbackStatus(
        `Learned ${label}. Memory now has ${response.examples} examples.`,
      )
    } catch (requestError) {
      const message = requestError?.response?.data?.detail || requestError?.message || 'Feedback failed.'
      setFeedbackStatus('')
      setError(`Could not save correction. ${message}`)
    }
  }

  return (
    <main className="min-h-screen bg-[#f5f7fb] px-4 py-6 text-slate-950 sm:px-6 lg:px-8">
      <div className="mx-auto flex w-full max-w-6xl flex-col gap-6">
        <header className="grid gap-5 rounded-md border border-slate-200 bg-white p-5 shadow-sm lg:grid-cols-[1fr_420px] lg:items-center">
          <div>
            <div className="inline-flex items-center gap-2 rounded-full bg-sky-50 px-3 py-1 text-xs font-semibold text-sky-800 ring-1 ring-sky-100">
              <ShieldCheck className="h-3.5 w-3.5" aria-hidden="true" />
              Multi-modal image authenticity verification
            </div>
            <h1 className="mt-4 max-w-3xl text-3xl font-bold text-slate-950 sm:text-4xl">
              Detect real photographs versus AI-generated images.
            </h1>
            <p className="mt-3 max-w-2xl text-sm leading-6 text-slate-600">
              Upload an image and send it through the backend's pixel ensemble, forensic analyzers, semantic reasoning, and meta-classifier fusion.
            </p>
          </div>
          <div className="grid gap-3">
            {systemStats.map((item) => {
              const Icon = item.icon
              return (
                <div key={item.label} className="flex items-center gap-3 rounded-md border border-slate-200 bg-slate-50 p-3">
                  <div className="grid h-9 w-9 place-items-center rounded-md bg-white text-slate-700 shadow-sm">
                    <Icon className="h-4 w-4" aria-hidden="true" />
                  </div>
                  <div>
                    <p className="text-sm font-semibold text-slate-950">{item.label}</p>
                    <p className="text-xs text-slate-500">{item.value}</p>
                  </div>
                </div>
              )
            })}
          </div>
        </header>

        {error ? (
          <div className="flex items-start gap-3 rounded-md border border-rose-200 bg-rose-50 px-4 py-3 text-rose-800">
            <AlertCircle className="mt-0.5 h-5 w-5 flex-none" aria-hidden="true" />
            <p className="text-sm font-medium">{error}</p>
          </div>
        ) : null}

        <div className="grid gap-6 lg:grid-cols-[minmax(0,1fr)_420px]">
          <UploadBox
            file={file}
            previewUrl={previewUrl}
            onFileSelect={handleFileSelect}
            onAnalyze={handleAnalyze}
            onClear={handleReset}
            isLoading={isLoading}
          />

          <aside className="space-y-4">
            {isLoading ? <Loader /> : null}
            {result ? (
              <ResultCard
                result={result}
                onReset={handleReset}
                onFeedback={handleFeedback}
                feedbackStatus={feedbackStatus}
                adminKey={adminKey}
                onAdminKeyChange={setAdminKey}
              />
            ) : null}
            {!isLoading && !result ? (
              <div className="rounded-md border border-slate-200 bg-white p-5 shadow-sm">
                <h2 className="text-sm font-bold uppercase tracking-normal text-slate-500">Decision Output</h2>
                <div className="mt-4 space-y-3">
                  <div className="h-3 w-2/3 rounded-full bg-slate-200" />
                  <div className="h-3 w-full rounded-full bg-slate-100" />
                  <div className="h-3 w-5/6 rounded-full bg-slate-100" />
                </div>
                <p className="mt-5 text-sm leading-6 text-slate-600">
                  Results will appear here with confidence, stream evidence, and model-level signal bars.
                </p>
              </div>
            ) : null}
          </aside>
        </div>
      </div>
    </main>
  )
}

export default App
