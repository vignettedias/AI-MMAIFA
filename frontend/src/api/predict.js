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
  const isReview = rawLabel.includes('REVIEW')
  const isUnknown = rawLabel === 'UNKNOWN'
  const isManipulated = rawLabel === 'MANIPULATED'
  const label = isReview
    ? 'HUMAN REVIEW'
    : isUnknown
      ? 'UNKNOWN'
      : isManipulated
        ? 'MANIPULATED'
        : isFake
          ? 'AI-GENERATED'
          : 'REAL'
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
    uncertainty: details?.uncertainty || {},
    reviewDecision: details?.review_decision || {},
    evidenceFusion: details?.evidence_fusion || {},
    streamDetails: details?.stream_details || {},
    topEvidence: buildTopEvidence(features),
    semanticAgents: buildSemanticAgents(features, details?.stream_details || {}),
    decisionBand: details?.decision_band || null,
    decisionMessage: details?.decision_message || '',
    requiresHumanReview: Boolean(details?.review_decision?.requires_human_review),
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
      features.forensic_compression,
      features.forensic_spectral_intelligence,
      features.forensic_camera_provenance,
      features.forensic_manipulation,
      features.forensic_metadata_provenance,
    ]),
  ])

  const semantic = firstNumber([
    directDetails.semantic,
    features.semantic_anomaly,
    streamScores.semantic_anomaly,
    average([
      features.semantic_portrait_synthetic,
      features.semantic_stylized_composite,
      features.semantic_low_resolution_composite,
      features.semantic_nature_render,
      features.semantic_architecture_render,
    ]),
  ])

  const modelScores = [
    ['EfficientNet', features.pixel_efficientnet ?? streamScores.pixel_efficientnet],
    ['AlexNet', features.pixel_alexnet ?? streamScores.pixel_alexnet],
    ['GoogLeNet', features.pixel_googlenet ?? streamScores.pixel_googlenet],
    ['ConvNeXt-V2', features.pixel_convnext_v2 ?? streamScores.pixel_convnext_v2],
    ['Swin V2', features.pixel_swin_v2 ?? streamScores.pixel_swin_v2],
    ['DINOv2', features.pixel_dinov2 ?? streamScores.pixel_dinov2],
    ['CLIP L/14', features.pixel_clip_vit_l14 ?? streamScores.pixel_clip_vit_l14],
    ['SigLIP', features.pixel_siglip ?? streamScores.pixel_siglip],
    ['Diffusion', features.expert_diffusion_artifact ?? streamScores.expert_diffusion_artifact],
    ['Face swap', features.expert_faceswap ?? streamScores.expert_faceswap],
    ['ELA', features.forensic_ela ?? streamScores.forensic_ela],
    ['FFT', features.forensic_fft ?? streamScores.forensic_fft],
    ['Noise', features.forensic_noise ?? streamScores.forensic_noise],
    ['Compression', features.forensic_compression ?? streamScores.forensic_compression],
    ['PRNU', features.forensic_prnu ?? streamScores.forensic_prnu],
    ['Diff trace', features.forensic_diffusion_trace ?? streamScores.forensic_diffusion_trace],
    ['Adv freq', features.forensic_advanced_frequency ?? streamScores.forensic_advanced_frequency],
    ['Provenance', features.forensic_provenance_consistency ?? streamScores.forensic_provenance_consistency],
    ['Spectral AI', features.forensic_spectral_intelligence ?? streamScores.forensic_spectral_intelligence],
    ['Camera trace', features.forensic_camera_provenance ?? streamScores.forensic_camera_provenance],
    ['Manipulation', features.forensic_manipulation ?? streamScores.forensic_manipulation],
    ['Metadata', features.forensic_metadata_provenance ?? streamScores.forensic_metadata_provenance],
    ['Portrait', features.semantic_portrait_synthetic ?? streamScores.semantic_portrait_synthetic],
    ['Composite', features.semantic_stylized_composite ?? streamScores.semantic_stylized_composite],
    ['Low-res', features.semantic_low_resolution_composite ?? streamScores.semantic_low_resolution_composite],
    ['Landscape', features.semantic_nature_render ?? streamScores.semantic_nature_render],
    ['Architecture', features.semantic_architecture_render ?? streamScores.semantic_architecture_render],
    ['Agents', features.semantic_multi_agent ?? streamScores.semantic_multi_agent],
    ['Physics', features.semantic_physical_realism ?? streamScores.semantic_physical_realism],
    ['Typography', features.semantic_typography ?? streamScores.semantic_typography],
  ]
    .filter(([, value]) => Number.isFinite(Number(value)))
    .map(([name, value]) => ({ name, value: toPercent(value) }))

  return {
    streams: [
      {
        name: 'Pixel Ensemble',
        value: toPercent(pixel),
        description: 'Foundation and specialist pixel evidence.',
      },
      {
        name: 'Forensic Signals',
        value: toPercent(forensic),
        description: 'Noise, diffusion, provenance, and spectral traces.',
      },
      {
        name: 'Semantic Reasoning',
        value: toPercent(semantic),
        description: 'Agent reasoning over realism and coherence.',
      },
    ].filter((item) => Number.isFinite(item.value)),
    modelScores,
  }
}

function buildTopEvidence(features) {
  return Object.entries(features || {})
    .map(([name, value]) => ({ name, value: Number(value) }))
    .filter((item) => Number.isFinite(item.value))
    .sort((a, b) => b.value - a.value)
    .slice(0, 12)
}

function buildSemanticAgents(features, streamDetails) {
  return Object.entries(features || {})
    .filter(([name]) => name.startsWith('semantic_agent_'))
    .map(([name, value]) => ({
      agent: name.replace('semantic_agent_', ''),
      value: Number(value),
      reasoning: streamDetails?.[name]?.reasoning || '',
    }))
    .filter((item) => Number.isFinite(item.value))
    .sort((a, b) => b.value - a.value)
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
