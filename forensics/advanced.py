from __future__ import annotations

import math

import numpy as np
from PIL import Image

from utils.image_io import image_to_array
from utils.math_utils import clip01, entropy01, normalize_minmax, safe_mean
from utils.types import StreamScore


class AdvancedNoiseForensicsAnalyzer:
    """Camera-noise and residual-coherence forensic probes.

    These probes approximate PRNU, SRM, BayarConv, and denoiser residual
    behavior without requiring trained weights. The output contract mirrors the
    production model interface: every score is normalized fake evidence.
    """

    name = "forensic_noise_advanced"

    def __init__(self, size: int = 256):
        self.size = size

    def analyze(self, image: Image.Image) -> dict[str, StreamScore]:
        arr = image_to_array(image, size=(self.size, self.size), grayscale=False)
        gray = np.mean(arr, axis=2)
        residual = gray - _box_blur(gray, 5)
        srm = _srm_residual(gray)
        bayar = _bayar_like_residual(gray)
        dncnn = residual - _box_blur(residual, 3)

        prnu_strength = _block_std_cv(residual)
        srm_energy = float(np.mean(np.abs(srm)))
        bayar_energy = float(np.mean(np.abs(bayar)))
        dncnn_smoothness = clip01(1.0 - normalize_minmax(float(np.mean(np.abs(dncnn))), 0.004, 0.05))
        patch_consistency = clip01(1.0 - normalize_minmax(_block_std_cv(residual, 32), 0.14, 0.70))
        camera_noise_coherence = _noise_luma_coherence(gray, residual)

        scores = {
            "forensic_prnu": _fake_from_missing_camera_trace(prnu_strength, camera_noise_coherence),
            "forensic_srm_residual": normalize_minmax(srm_energy, 0.020, 0.105),
            "forensic_bayar_residual": normalize_minmax(bayar_energy, 0.018, 0.095),
            "forensic_dncnn_residual": dncnn_smoothness,
            "forensic_patch_noise_consistency": clip01(1.0 - patch_consistency),
            "forensic_camera_noise_coherence": clip01(1.0 - camera_noise_coherence),
        }
        details = {
            "prnu_strength": prnu_strength,
            "srm_energy": srm_energy,
            "bayar_energy": bayar_energy,
            "dncnn_smoothness": dncnn_smoothness,
            "patch_consistency": patch_consistency,
            "camera_noise_coherence": camera_noise_coherence,
        }
        stream_scores = {
            name: StreamScore(name, score, {**details, "probe": name})
            for name, score in scores.items()
        }
        stream_scores[self.name] = StreamScore(
            self.name,
            safe_mean(list(scores.values())),
            {**details, "members": list(scores)},
        )
        return stream_scores


class DiffusionTraceAnalyzer:
    """Dedicated diffusion and latent-upsampling trace detector."""

    name = "forensic_diffusion_trace"

    def __init__(self, size: int = 256):
        self.size = size

    def analyze(self, image: Image.Image) -> dict[str, StreamScore]:
        arr = image_to_array(image, size=(self.size, self.size), grayscale=False)
        gray = np.mean(arr, axis=2)
        grad = _gradient(gray)
        residual = gray - _box_blur(gray, 5)
        spectrum = _spectrum(gray)
        radius_norm = spectrum["radius_norm"]
        magnitude = spectrum["log_magnitude"]

        flat_regions = grad < np.percentile(grad, 36)
        edge_regions = grad > np.percentile(grad, 82)
        flat_noise = float(np.mean(np.abs(residual[flat_regions]))) if np.any(flat_regions) else 0.0
        edge_noise = float(np.mean(np.abs(residual[edge_regions]))) if np.any(edge_regions) else 0.0
        denoising_remnants = clip01(1.0 - normalize_minmax(flat_noise, 0.005, 0.050))
        latent_upsampling = clip01(0.55 * _grid_periodicity(gray, period=8) + 0.45 * _grid_periodicity(gray, period=16))
        timestep_artifacts = _radial_ripple_score(magnitude, radius_norm)
        microtexture_smoothness = clip01(0.70 * denoising_remnants + 0.30 * (1.0 - entropy01(gray)))
        edge_oversampling = normalize_minmax(edge_noise / (flat_noise + 1e-8), 2.2, 7.0)
        spectral_flattening = _spectral_flatness(np.abs(spectrum["complex_magnitude"]))
        hallucination_residue = clip01(
            0.42 * _block_anomaly(gray, grad)
            + 0.26 * spectral_flattening
            + 0.18 * latent_upsampling
            + 0.14 * timestep_artifacts
        )

        values = {
            "forensic_diffusion_denoising": denoising_remnants,
            "forensic_diffusion_latent_upsampling": latent_upsampling,
            "forensic_diffusion_timestep": timestep_artifacts,
            "forensic_diffusion_microtexture": microtexture_smoothness,
            "forensic_diffusion_edge_oversampling": edge_oversampling,
            "forensic_diffusion_spectral_flattening": spectral_flattening,
            "forensic_diffusion_hallucination": hallucination_residue,
        }
        details = {
            "flat_noise": flat_noise,
            "edge_noise": edge_noise,
            "trained_generator_domains": [
                "Stable Diffusion",
                "SDXL",
                "Midjourney",
                "Flux",
                "DALL-E",
                "Firefly",
                "Kandinsky",
                "Imagen",
            ],
            "generalization_policy": "score traces rather than memorizing generator labels",
        }
        scores = {
            name: StreamScore(name, score, {**details, "probe": name})
            for name, score in values.items()
        }
        scores[self.name] = StreamScore(self.name, safe_mean(list(values.values())), details)
        return scores


class AdvancedFrequencyAnalyzer:
    """Multi-resolution frequency forensics beyond raw FFT."""

    name = "forensic_advanced_frequency"

    def __init__(self, size: int = 256):
        self.size = size

    def analyze(self, image: Image.Image) -> dict[str, StreamScore]:
        gray = image_to_array(image, size=(self.size, self.size), grayscale=True)
        spectrum = _spectrum(gray)
        magnitude = spectrum["log_magnitude"]
        radius_norm = spectrum["radius_norm"]

        radial = _radial_profile(magnitude, radius_norm)
        radial_slope = _radial_slope(radial)
        wavelet_energy = _wavelet_detail_energy(gray)
        pyramid_inconsistency = _fourier_pyramid_inconsistency(gray)
        anisotropic = _anisotropy_score(magnitude)
        periodicity = _periodicity_score(magnitude, radius_norm)
        spectral_entropy = entropy01(magnitude / max(float(np.max(magnitude)), 1e-8))
        hf_attenuation = clip01(1.0 - normalize_minmax(radial[-1] / (radial[1] + 1e-8), 0.12, 0.58))
        band_energy = _band_energy_score(magnitude, radius_norm)

        values = {
            "forensic_radial_spectral": normalize_minmax(abs(radial_slope), 0.006, 0.030),
            "forensic_wavelet": wavelet_energy,
            "forensic_fourier_pyramid": pyramid_inconsistency,
            "forensic_anisotropic_spectral": anisotropic,
            "forensic_periodicity": periodicity,
            "forensic_spectral_entropy": clip01(1.0 - spectral_entropy),
            "forensic_high_frequency_attenuation": hf_attenuation,
            "forensic_band_energy": band_energy,
        }
        details = {
            "radial_slope": radial_slope,
            "wavelet_detail_energy": wavelet_energy,
            "pyramid_inconsistency": pyramid_inconsistency,
            "anisotropy": anisotropic,
            "periodicity": periodicity,
            "spectral_entropy": spectral_entropy,
            "high_frequency_attenuation": hf_attenuation,
            "band_energy": band_energy,
            "frequency_anomaly_map": "available via /heatmap",
            "localized_periodicity_map": "available via /heatmap",
        }
        scores = {
            name: StreamScore(name, score, {**details, "probe": name})
            for name, score in values.items()
        }
        scores[self.name] = StreamScore(self.name, safe_mean(list(values.values())), details)
        return scores


class ProvenanceConsistencyAnalyzer:
    """Physical camera provenance consistency probes."""

    name = "forensic_provenance_consistency"

    def __init__(self, size: int = 256):
        self.size = size

    def analyze(self, image: Image.Image) -> dict[str, StreamScore]:
        arr = image_to_array(image, size=(self.size, self.size), grayscale=False)
        gray = np.mean(arr, axis=2)
        grad = _gradient(gray)

        cfa_missing = _cfa_missing_score(gray)
        lens_distortion_missing = _lens_distortion_missing_score(grad)
        rolling_shutter_missing = _rolling_shutter_missing_score(gray)
        exif_inconsistency = _exif_inconsistency_score(image)
        optical_aberration_missing = _optical_aberration_missing_score(arr, grad)
        chromatic_aberration_missing = _chromatic_aberration_missing_score(arr, grad)

        values = {
            "forensic_provenance_cfa": cfa_missing,
            "forensic_provenance_lens": lens_distortion_missing,
            "forensic_provenance_rolling_shutter": rolling_shutter_missing,
            "forensic_provenance_exif": exif_inconsistency,
            "forensic_provenance_optical_aberration": optical_aberration_missing,
            "forensic_provenance_chromatic_aberration": chromatic_aberration_missing,
        }
        details = {
            "cfa_missing": cfa_missing,
            "lens_distortion_missing": lens_distortion_missing,
            "rolling_shutter_missing": rolling_shutter_missing,
            "exif_inconsistency": exif_inconsistency,
            "optical_aberration_missing": optical_aberration_missing,
            "chromatic_aberration_missing": chromatic_aberration_missing,
            "interpretation": "higher values mean weaker physical camera-origin evidence",
        }
        scores = {
            name: StreamScore(name, score, {**details, "probe": name})
            for name, score in values.items()
        }
        scores[self.name] = StreamScore(self.name, safe_mean(list(values.values())), details)
        return scores


def _box_blur(arr: np.ndarray, kernel_size: int) -> np.ndarray:
    pad = kernel_size // 2
    padded = np.pad(arr, pad_width=pad, mode="reflect")
    out = np.zeros_like(arr, dtype=np.float32)
    for y in range(kernel_size):
        for x in range(kernel_size):
            out += padded[y : y + arr.shape[0], x : x + arr.shape[1]]
    return out / float(kernel_size * kernel_size)


def _gradient(values: np.ndarray) -> np.ndarray:
    gy = np.abs(np.diff(values, axis=0, append=values[-1:, :]))
    gx = np.abs(np.diff(values, axis=1, append=values[:, -1:]))
    return gx + gy


def _srm_residual(gray: np.ndarray) -> np.ndarray:
    padded = np.pad(gray, 1, mode="reflect")
    prediction = (
        padded[:-2, 1:-1]
        + padded[2:, 1:-1]
        + padded[1:-1, :-2]
        + padded[1:-1, 2:]
    ) / 4.0
    return gray - prediction


def _bayar_like_residual(gray: np.ndarray) -> np.ndarray:
    padded = np.pad(gray, 2, mode="reflect")
    neighborhood = np.zeros_like(gray, dtype=np.float32)
    count = 0
    for y in range(5):
        for x in range(5):
            if y == 2 and x == 2:
                continue
            neighborhood += padded[y : y + gray.shape[0], x : x + gray.shape[1]]
            count += 1
    return gray - (neighborhood / float(count))


def _block_std_cv(values: np.ndarray, block_size: int = 32) -> float:
    blocks = []
    for y in range(0, values.shape[0] - block_size + 1, block_size):
        for x in range(0, values.shape[1] - block_size + 1, block_size):
            blocks.append(float(np.std(values[y : y + block_size, x : x + block_size])))
    if not blocks:
        return 0.0
    return float(np.std(blocks) / (np.mean(blocks) + 1e-8))


def _noise_luma_coherence(gray: np.ndarray, residual: np.ndarray) -> float:
    bins = np.linspace(0.0, 1.0, 10)
    curve = []
    for low, high in zip(bins[:-1], bins[1:], strict=True):
        mask = (gray >= low) & (gray < high)
        if np.mean(mask) > 0.015:
            curve.append(float(np.mean(np.abs(residual[mask]))))
    if len(curve) < 3:
        return 0.0
    return clip01((float(np.corrcoef(np.arange(len(curve)), curve)[0, 1]) + 1.0) / 2.0)


def _fake_from_missing_camera_trace(prnu_strength: float, camera_noise_coherence: float) -> float:
    weak_prnu = clip01(1.0 - normalize_minmax(prnu_strength, 0.12, 0.58))
    weak_luma = clip01(1.0 - camera_noise_coherence)
    return clip01(0.58 * weak_prnu + 0.42 * weak_luma)


def _spectrum(gray: np.ndarray) -> dict[str, np.ndarray]:
    centered = gray - float(np.mean(gray))
    complex_magnitude = np.abs(np.fft.fftshift(np.fft.fft2(centered)))
    log_magnitude = np.log1p(complex_magnitude)
    h, w = log_magnitude.shape
    yy, xx = np.ogrid[:h, :w]
    cy, cx = h / 2.0, w / 2.0
    radius = np.sqrt((yy - cy) ** 2 + (xx - cx) ** 2)
    return {
        "complex_magnitude": complex_magnitude,
        "log_magnitude": log_magnitude,
        "radius_norm": radius / max(float(radius.max()), 1e-8),
    }


def _grid_periodicity(gray: np.ndarray, period: int) -> float:
    gy = np.abs(np.diff(gray, axis=0))
    gx = np.abs(np.diff(gray, axis=1))
    vertical = float(np.mean(gx[:, period - 1 :: period])) if gx.shape[1] > period else 0.0
    horizontal = float(np.mean(gy[period - 1 :: period, :])) if gy.shape[0] > period else 0.0
    base = float((np.mean(gx) + np.mean(gy)) / 2.0 + 1e-8)
    return normalize_minmax(((vertical + horizontal) / 2.0) / base, 0.95, 1.45)


def _radial_ripple_score(magnitude: np.ndarray, radius_norm: np.ndarray) -> float:
    profile = _radial_profile(magnitude, radius_norm, bins=36)
    second = np.diff(profile, n=2)
    return normalize_minmax(float(np.std(second)), 0.002, 0.045)


def _spectral_flatness(magnitude: np.ndarray) -> float:
    values = magnitude.ravel().astype(np.float64) + 1e-8
    geo = float(np.exp(np.mean(np.log(values))))
    arith = float(np.mean(values))
    return normalize_minmax(geo / (arith + 1e-8), 0.10, 0.55)


def _block_anomaly(gray: np.ndarray, grad: np.ndarray, block_size: int = 32) -> float:
    values = []
    for y in range(0, gray.shape[0] - block_size + 1, block_size):
        for x in range(0, gray.shape[1] - block_size + 1, block_size):
            values.append(float(np.mean(grad[y : y + block_size, x : x + block_size])))
    if not values:
        return 0.0
    values_arr = np.asarray(values, dtype=np.float64)
    return normalize_minmax(float(np.std(values_arr) / (np.mean(values_arr) + 1e-8)), 0.12, 0.85)


def _radial_profile(magnitude: np.ndarray, radius_norm: np.ndarray, bins: int = 32) -> np.ndarray:
    profile = []
    for low, high in zip(np.linspace(0.0, 1.0, bins)[:-1], np.linspace(0.0, 1.0, bins)[1:], strict=True):
        mask = (radius_norm >= low) & (radius_norm < high)
        profile.append(float(np.mean(magnitude[mask])) if np.any(mask) else 0.0)
    profile_arr = np.asarray(profile, dtype=np.float64)
    return profile_arr / max(float(np.max(profile_arr)), 1e-8)


def _radial_slope(radial: np.ndarray) -> float:
    if radial.size < 2:
        return 0.0
    x = np.arange(radial.size, dtype=np.float64)
    return float(np.polyfit(x, radial, deg=1)[0])


def _wavelet_detail_energy(gray: np.ndarray) -> float:
    even_rows = gray[0::2, :]
    odd_rows = gray[1::2, :]
    even_cols = gray[:, 0::2]
    odd_cols = gray[:, 1::2]
    horizontal = float(np.mean(np.abs(even_rows[: odd_rows.shape[0], :] - odd_rows)))
    vertical = float(np.mean(np.abs(even_cols[:, : odd_cols.shape[1]] - odd_cols)))
    return normalize_minmax((horizontal + vertical) / 2.0, 0.010, 0.090)


def _fourier_pyramid_inconsistency(gray: np.ndarray) -> float:
    energies = []
    current = gray
    for _ in range(3):
        spectrum = _spectrum(current)
        mag = spectrum["log_magnitude"]
        radius = spectrum["radius_norm"]
        energies.append(float(np.mean(mag[radius > 0.55]) / (np.mean(mag) + 1e-8)))
        if min(current.shape) <= 64:
            break
        current = _box_blur(current, 3)[::2, ::2]
    return normalize_minmax(float(np.std(energies)), 0.02, 0.32) if energies else 0.0


def _anisotropy_score(log_magnitude: np.ndarray) -> float:
    h, w = log_magnitude.shape
    vertical = float(np.mean(log_magnitude[:, w // 2 - 2 : w // 2 + 3]))
    horizontal = float(np.mean(log_magnitude[h // 2 - 2 : h // 2 + 3, :]))
    diag = float(np.mean(np.diag(log_magnitude)) + np.mean(np.diag(np.fliplr(log_magnitude)))) / 2.0
    base = float(np.mean(log_magnitude) + 1e-8)
    return normalize_minmax(max(abs(vertical - horizontal), abs(vertical - diag)) / base, 0.03, 0.26)


def _periodicity_score(log_magnitude: np.ndarray, radius_norm: np.ndarray) -> float:
    outer = log_magnitude[radius_norm > 0.25]
    if outer.size < 8:
        return 0.0
    peaks = np.percentile(outer, 99.3)
    mean = float(np.mean(outer))
    std = float(np.std(outer) + 1e-8)
    return normalize_minmax((peaks - mean) / std, 2.0, 5.5)


def _band_energy_score(magnitude: np.ndarray, radius_norm: np.ndarray) -> float:
    low = float(np.mean(magnitude[(radius_norm > 0.05) & (radius_norm <= 0.22)]))
    mid = float(np.mean(magnitude[(radius_norm > 0.22) & (radius_norm <= 0.55)]))
    high = float(np.mean(magnitude[radius_norm > 0.55]))
    return normalize_minmax(abs(mid - (low + high) / 2.0) / (low + mid + high + 1e-8), 0.025, 0.24)


def _cfa_missing_score(gray: np.ndarray) -> float:
    residual = gray - _box_blur(gray, 3)
    cells = [residual[y::2, x::2] for y in range(2) for x in range(2)]
    energies = np.asarray([float(np.mean(np.abs(cell))) for cell in cells], dtype=np.float64)
    cfa_coherence = float(np.std(energies) / (np.mean(energies) + 1e-8))
    return clip01(1.0 - normalize_minmax(cfa_coherence, 0.035, 0.18))


def _lens_distortion_missing_score(grad: np.ndarray) -> float:
    h, w = grad.shape
    yy, xx = np.ogrid[:h, :w]
    center = np.sqrt((yy - h / 2.0) ** 2 + (xx - w / 2.0) ** 2)
    center /= max(float(center.max()), 1e-8)
    radial_bins = np.linspace(0.0, 1.0, 8)
    curve = []
    for low, high in zip(radial_bins[:-1], radial_bins[1:], strict=True):
        mask = (center >= low) & (center < high)
        curve.append(float(np.mean(grad[mask])) if np.any(mask) else 0.0)
    curvature = float(np.std(np.diff(curve, n=2))) if len(curve) > 3 else 0.0
    return clip01(1.0 - normalize_minmax(curvature, 0.002, 0.035))


def _rolling_shutter_missing_score(gray: np.ndarray) -> float:
    row_grad = np.abs(np.diff(gray, axis=1)).mean(axis=1)
    drift = float(np.std(_moving_average(row_grad, 9)))
    return clip01(1.0 - normalize_minmax(drift, 0.002, 0.030))


def _exif_inconsistency_score(image: Image.Image) -> float:
    exif = getattr(image, "getexif", lambda: {})()
    exif_count = len(exif) if exif else 0
    width, height = image.size
    screenshot_like = width / max(height, 1) in {16 / 9, 9 / 16}
    missing_metadata = 1.0 if exif_count == 0 else 0.0
    tiny_or_generated_canvas = 1.0 if min(width, height) < 384 else 0.0
    return clip01(0.62 * missing_metadata + 0.24 * tiny_or_generated_canvas + 0.14 * float(screenshot_like))


def _optical_aberration_missing_score(arr: np.ndarray, grad: np.ndarray) -> float:
    channel_edges = [_gradient(arr[:, :, idx]) for idx in range(3)]
    rg = _corr(channel_edges[0].ravel(), channel_edges[1].ravel())
    gb = _corr(channel_edges[1].ravel(), channel_edges[2].ravel())
    edge_mask = grad > np.percentile(grad, 80)
    edge_mass = float(np.mean(edge_mask))
    over_aligned = normalize_minmax((rg + gb) / 2.0, 0.86, 0.988)
    return clip01(0.82 * over_aligned + 0.18 * normalize_minmax(edge_mass, 0.05, 0.25))


def _chromatic_aberration_missing_score(arr: np.ndarray, grad: np.ndarray) -> float:
    edge_mask = grad > np.percentile(grad, 82)
    if not np.any(edge_mask):
        return 0.5
    rg_shift = float(np.mean(np.abs(arr[:, :, 0][edge_mask] - arr[:, :, 1][edge_mask])))
    gb_shift = float(np.mean(np.abs(arr[:, :, 1][edge_mask] - arr[:, :, 2][edge_mask])))
    aberration = (rg_shift + gb_shift) / 2.0
    return clip01(1.0 - normalize_minmax(aberration, 0.018, 0.085))


def _moving_average(values: np.ndarray, window: int) -> np.ndarray:
    if values.size < window:
        return values
    kernel = np.ones(window, dtype=np.float64) / float(window)
    return np.convolve(values, kernel, mode="same")


def _corr(a: np.ndarray, b: np.ndarray) -> float:
    if a.size < 2 or b.size < 2:
        return 0.0
    if math.isclose(float(np.std(a)), 0.0) or math.isclose(float(np.std(b)), 0.0):
        return 0.0
    return float(np.corrcoef(a, b)[0, 1])
