import axios from 'axios'

const API_BASE_URL = import.meta.env.VITE_API_BASE_URL || '/api'

const api = axios.create({
  baseURL: API_BASE_URL,
  timeout: 60000,
})

export async function predictImage(file) {
  const formData = new FormData()
  formData.append('file', file)

  const response = await api.post('/predict', formData, {
    headers: {
      'Content-Type': 'multipart/form-data',
    },
  })

  return normalizePrediction(response.data)
}

export async function submitFeedback({ result, label, note = '', adminKey }) {
  const response = await api.post(
    '/feedback',
    {
      label,
      features: result.features,
      image_hash: result.imageHash,
      note,
    },
    {
      headers: {
        'X-Admin-Key': adminKey,
      },
    },
  )

  return response.data
}

function normalizePrediction(payload) {
  const rawLabel = String(payload?.label || '').toUpperCase()
  const isFake = rawLabel === 'FAKE' || rawLabel.includes('AI')
  const label = isFake ? 'AI-GENERATED' : 'REAL'
  const confidence = clamp(Number(payload?.confidence ?? 0), 0, 100)
  const features = payload?.features || {}
  const details = payload?.details || {}
  const streamScores = payload?.stream_scores || {}

  return {
    label,
    rawLabel,
    confidence,
    features,
    imageHash: payload?.image_hash || details?.image_hash || null,
    fakeProbability: Number(payload?.fake_probability ?? (isFake ? confidence / 100 : 1 - confidence / 100)),
    breakdown: buildBreakdown(features, details, streamScores),
    featureOrder: details?.feature_order || [],
    metaBackend: details?.meta_backend || 'unknown',
    metaDecision: details?.meta_decision || {},
    dynamicMemory: details?.dynamic_memory || {},
    autoMemory: details?.auto_memory || {},
    pixelBackends: details?.pixel_backends || {},
  }
}

function buildBreakdown(features, details, streamScores) {
  const directDetails = {
    pixel: details?.pixel,
    forensic: details?.forensic,
    semantic: details?.semantic,
  }

  const pixel = firstNumber([
    directDetails.pixel,
    features.pixel_ensemble,
    streamScores.pixel_ensemble,
    average([
      features.pixel_efficientnet,
      features.pixel_alexnet,
      features.pixel_googlenet,
    ]),
  ])

  const forensic = firstNumber([
    directDetails.forensic,
    features.forensic_ensemble,
    streamScores.forensic_ensemble,
    average([
      features.forensic_ela,
      features.forensic_fft,
      features.forensic_noise,
    ]),
  ])

  const semantic = firstNumber([
    directDetails.semantic,
    features.semantic_anomaly,
    streamScores.semantic_anomaly,
  ])

  const modelScores = [
    ['EfficientNet', features.pixel_efficientnet ?? streamScores.pixel_efficientnet],
    ['AlexNet', features.pixel_alexnet ?? streamScores.pixel_alexnet],
    ['GoogLeNet', features.pixel_googlenet ?? streamScores.pixel_googlenet],
    ['ELA', features.forensic_ela ?? streamScores.forensic_ela],
    ['FFT', features.forensic_fft ?? streamScores.forensic_fft],
    ['Noise', features.forensic_noise ?? streamScores.forensic_noise],
  ]
    .filter(([, value]) => Number.isFinite(Number(value)))
    .map(([name, value]) => ({ name, value: toPercent(value) }))

  return {
    streams: [
      {
        name: 'Pixel Ensemble',
        value: toPercent(pixel),
        description: 'CNN evidence from EfficientNet, AlexNet, and GoogLeNet.',
      },
      {
        name: 'Forensic Signals',
        value: toPercent(forensic),
        description: 'ELA, frequency spectrum, and noise inconsistency checks.',
      },
      {
        name: 'Semantic Reasoning',
        value: toPercent(semantic),
        description: 'Vision-language consistency and synthetic-anomaly cues.',
      },
    ].filter((item) => Number.isFinite(item.value)),
    modelScores,
  }
}

function firstNumber(values) {
  return values.find((value) => Number.isFinite(Number(value)))
}

function average(values) {
  const numbers = values.map(Number).filter(Number.isFinite)
  if (!numbers.length) return undefined
  return numbers.reduce((sum, value) => sum + value, 0) / numbers.length
}

function toPercent(value) {
  const number = Number(value)
  if (!Number.isFinite(number)) return Number.NaN
  return clamp(number <= 1 ? number * 100 : number, 0, 100)
}

function clamp(value, min, max) {
  return Math.min(max, Math.max(min, value))
}
