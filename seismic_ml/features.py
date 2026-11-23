"""Feature extraction for a waveform window around a candidate seismic event.

Every candidate (an STA/LTA trigger, a catalogue event or a random noise window)
is turned into one fixed-length feature vector.  The features follow the table
in report section 3.2.2-d and add a few spectral / envelope descriptors that help
separate genuine quakes from glitches and environmental noise.

All amplitude features are computed on the *normalised* trace (|x| <= 1), so
lunar (m/s) and Martian (counts/s) recordings live on the same scale.
"""

from __future__ import annotations

import numpy as np
from scipy import signal, stats

from .config import PlanetConfig
from .sta_lta import DetectionResult

FEATURE_NAMES: list[str] = [
    # --- report features -----------------------------------------------------
    "duration_s",              # time the smoothed envelope stays above 3x noise level
    "rms_amplitude",           # root-mean-square of the window
    "peak_amplitude",          # max |x| in the window
    "mean_abs_velocity",       # mean |x| (mean of x itself is ~0 after detrending)
    "sta_lta_peak",            # max STA/LTA ratio in the window
    "waveform_skewness",
    "waveform_kurtosis",
    "energy",                  # integral of x^2 dt
    # --- STA/LTA context --------------------------------------------------------
    "sta_lta_mean",
    "sta_lta_peak_over_threshold",
    "time_above_threshold_s",
    # --- amplitude / shape -----------------------------------------------------
    "snr_db",                  # window RMS vs. robust trace noise level
    "crest_factor",            # peak / RMS  (large for spikes and glitches)
    "envelope_skewness",
    "envelope_kurtosis",
    "rise_time_s",             # anchor -> envelope maximum
    "coda_energy_fraction",    # share of energy after the envelope maximum
    # --- spectral --------------------------------------------------------------
    "zero_crossing_rate",
    "dominant_freq_hz",
    "spectral_centroid_hz",
    "spectral_bandwidth_hz",
]


def robust_noise_level(x: np.ndarray) -> float:
    """Gaussian-equivalent standard deviation estimated with the MAD."""
    mad = np.median(np.abs(x - np.median(x)))
    sigma = 1.4826 * mad
    return float(sigma) if sigma > 0 else float(np.std(x) or 1e-12)


def _moving_average(x: np.ndarray, n: int) -> np.ndarray:
    if n <= 1 or len(x) < n:
        return x
    kernel = np.ones(n) / n
    return np.convolve(x, kernel, mode="same")


def window_bounds(anchor_idx: int, n_samples: int, fs: float, cfg: PlanetConfig) -> tuple[int, int]:
    start = int(anchor_idx - round(cfg.pre_event_sec * fs))
    stop = start + int(round(cfg.window_sec * fs))
    return max(0, start), min(n_samples, stop)


def extract_features(x: np.ndarray, det: DetectionResult, fs: float, anchor_idx: int,
                     cfg: PlanetConfig, noise_sigma: float | None = None) -> dict[str, float]:
    """Compute :data:`FEATURE_NAMES` for the window anchored at ``anchor_idx``."""
    if noise_sigma is None:
        noise_sigma = robust_noise_level(x)
    start, stop = window_bounds(anchor_idx, len(x), fs, cfg)
    seg = x[start:stop]
    r = det.ratio[start:stop]
    if len(seg) < 8:
        return {name: 0.0 for name in FEATURE_NAMES}

    n = len(seg)
    abs_seg = np.abs(seg)
    rms = float(np.sqrt(np.mean(seg ** 2)))
    peak = float(abs_seg.max())

    envelope = np.abs(signal.hilbert(seg))
    smooth_n = max(1, int(round(cfg.sta_sec * fs / 4)))
    env_s = _moving_average(envelope, smooth_n)
    env_peak_idx = int(np.argmax(env_s))
    anchor_in_seg = anchor_idx - start

    energy_cum = np.cumsum(seg ** 2)
    total_energy = float(energy_cum[-1])

    # spectral descriptors (Welch PSD)
    nperseg = min(n, 1024)
    freqs, psd = signal.welch(seg, fs=fs, nperseg=nperseg)
    psd_sum = float(psd.sum())
    if psd_sum > 0:
        centroid = float(np.sum(freqs * psd) / psd_sum)
        bandwidth = float(np.sqrt(np.sum(((freqs - centroid) ** 2) * psd) / psd_sum))
        dominant = float(freqs[int(np.argmax(psd))])
    else:
        centroid = bandwidth = dominant = 0.0

    zero_crossings = np.count_nonzero(np.diff(np.signbit(seg)))
    thr = det.threshold
    peak_over_thr = float(r.max() / thr) if len(r) and np.isfinite(thr) and thr > 0 else 0.0
    coda_fraction = (total_energy - energy_cum[env_peak_idx]) / total_energy if total_energy > 0 else 0.0

    feats = {
        "duration_s": float(np.count_nonzero(env_s > 3.0 * noise_sigma) / fs),
        "rms_amplitude": rms,
        "peak_amplitude": peak,
        "mean_abs_velocity": float(abs_seg.mean()),
        "sta_lta_peak": float(r.max()) if len(r) else 0.0,
        "waveform_skewness": float(stats.skew(seg)),
        "waveform_kurtosis": float(stats.kurtosis(seg)),
        "energy": float(total_energy / fs),
        "sta_lta_mean": float(r.mean()) if len(r) else 0.0,
        "sta_lta_peak_over_threshold": peak_over_thr,
        "time_above_threshold_s": float(np.count_nonzero(r > det.threshold) / fs),
        "snr_db": float(20.0 * np.log10(max(rms, 1e-12) / max(noise_sigma, 1e-12))),
        "crest_factor": float(peak / rms) if rms > 0 else 0.0,
        "envelope_skewness": float(stats.skew(env_s)),
        "envelope_kurtosis": float(stats.kurtosis(env_s)),
        "rise_time_s": float((env_peak_idx - anchor_in_seg) / fs),
        "coda_energy_fraction": float(coda_fraction),
        "zero_crossing_rate": float(zero_crossings / (n / fs)),
        "dominant_freq_hz": dominant,
        "spectral_centroid_hz": centroid,
        "spectral_bandwidth_hz": bandwidth,
    }
    return {k: float(np.nan_to_num(v, nan=0.0, posinf=0.0, neginf=0.0)) for k, v in feats.items()}
