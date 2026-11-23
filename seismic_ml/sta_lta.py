"""Classical STA/LTA detection with a dynamic (mu + k*sigma) threshold.

    R(t) = STA(t) / LTA(t)

    STA(t) = mean of the signal energy over the last N_s samples
    LTA(t) = mean of the signal energy over the last N_l samples

A trigger switches on when R(t) exceeds ``mu + k*sigma`` and switches off when
R(t) drops below ``mu + k_off*sigma``.  Triggers separated by less than
``min_gap`` are merged so that a single long event is not reported many times.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .config import PlanetConfig


@dataclass
class Trigger:
    onset: int        # sample index where R first exceeds the on-threshold
    offset: int       # sample index where R falls back below the off-threshold
    peak: int         # sample index of the maximum R inside the trigger
    peak_ratio: float


def sta_lta_ratio(x: np.ndarray, n_sta: int, n_lta: int) -> np.ndarray:
    """Trailing-window STA/LTA ratio of the signal energy (x**2).

    Uses cumulative sums so it runs in O(N) even for day-long lunar traces.
    The first ``n_lta`` samples (LTA warm-up) are set to 0.
    """
    n = len(x)
    n_sta = max(1, int(n_sta))
    n_lta = max(n_sta + 1, int(n_lta))
    ratio = np.zeros(n, dtype=float)
    if n <= n_lta:
        return ratio

    energy = np.asarray(x, dtype=float) ** 2
    csum = np.concatenate(([0.0], np.cumsum(energy)))
    end = np.arange(n_lta, n + 1)            # window end (exclusive) for samples n_lta-1 .. n-1
    sta = (csum[end] - csum[end - n_sta]) / n_sta
    lta = (csum[end] - csum[end - n_lta]) / n_lta
    with np.errstate(divide="ignore", invalid="ignore"):
        r = np.where(lta > 0, sta / lta, 0.0)
    ratio[n_lta - 1:] = r
    return ratio


def dynamic_threshold(ratio: np.ndarray, k: float, warmup: int = 0) -> tuple[float, float, float]:
    """Return (threshold, mean, std) of the ratio, ignoring the warm-up zeros."""
    valid = ratio[warmup:] if warmup < len(ratio) else ratio
    valid = valid[np.isfinite(valid)]
    if valid.size == 0:
        return float("inf"), 0.0, 0.0
    mu, sigma = float(np.mean(valid)), float(np.std(valid))
    return mu + k * sigma, mu, sigma


def trigger_onsets(ratio: np.ndarray, thr_on: float, thr_off: float, min_gap: int) -> list[Trigger]:
    """Find on/off trigger pairs and merge those closer than ``min_gap`` samples."""
    thr_off = min(thr_off, thr_on)
    above = ratio > thr_on
    if not above.any():
        return []

    n = len(ratio)
    on_idx = np.flatnonzero(above)
    off_idx = np.flatnonzero(ratio < thr_off)
    raw: list[list[int]] = []
    ptr = 0
    while ptr < len(on_idx):
        onset = int(on_idx[ptr])
        # first sample after the onset where the ratio drops below the off threshold
        j = int(np.searchsorted(off_idx, onset))
        offset = int(off_idx[j]) if j < len(off_idx) else n - 1
        raw.append([onset, offset])
        # skip all on-samples already covered by this trigger
        ptr = int(np.searchsorted(on_idx, offset + 1))

    merged: list[list[int]] = [raw[0]]
    for onset, offset in raw[1:]:
        if onset - merged[-1][1] <= min_gap:
            merged[-1][1] = max(merged[-1][1], offset)
        else:
            merged.append([onset, offset])

    triggers = []
    for onset, offset in merged:
        seg = ratio[onset:offset + 1]
        peak = onset + int(np.argmax(seg))
        triggers.append(Trigger(onset=onset, offset=offset, peak=peak, peak_ratio=float(ratio[peak])))
    return triggers


@dataclass
class DetectionResult:
    ratio: np.ndarray
    threshold: float
    threshold_off: float
    mean: float
    std: float
    triggers: list[Trigger]
    n_sta: int
    n_lta: int


def run_sta_lta(x: np.ndarray, fs: float, cfg: PlanetConfig) -> DetectionResult:
    """Full STA/LTA detection on a preprocessed trace."""
    n_sta = int(round(cfg.sta_sec * fs))
    n_lta = int(round(cfg.lta_sec * fs))
    ratio = sta_lta_ratio(x, n_sta, n_lta)
    thr_on, mu, sigma = dynamic_threshold(ratio, cfg.k, warmup=n_lta)
    thr_off = mu + cfg.k_off * sigma
    triggers = trigger_onsets(ratio, thr_on, thr_off, int(round(cfg.min_gap_sec * fs)))
    return DetectionResult(ratio, thr_on, thr_off, mu, sigma, triggers, n_sta, n_lta)
