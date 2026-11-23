"""Synthetic planetary seismograms for demos and automated tests.

The real Apollo / InSight data (NASA Space Apps 2024 package) are ~1 GB and are
not stored in this repository.  This module writes small CSV files in exactly
the same format, so the whole pipeline can be run and tested without them.

Each synthetic trace contains
* coloured background noise + slow baseline drift (+ a few missing samples),
* 1-2 catalogued quakes (emergent onset, exponentially decaying coda),
* un-catalogued disturbances that fool STA/LTA: short high-amplitude glitches
  and longer box-shaped noise bursts.

**Results obtained on synthetic data say nothing about real planetary data.**
"""

from __future__ import annotations

import os

import numpy as np
import pandas as pd

from .preprocessing import butterworth_filter

PLANET_SPECS = {
    "lunar": {
        "fs": 6.625, "duration_s": 4 * 3600, "sigma": 2e-10, "start": "1971-01-01",
        "file_step": pd.Timedelta(days=1),
        "cols": ("time_abs(%Y-%m-%dT%H:%M:%S.%f)", "time_rel(sec)", "velocity(m/s)"),
        "catalog_name": "apollo12_catalog_synthetic.csv", "keep_extension": False,
        "edge_s": (900.0, 2500.0), "min_sep_s": 3500.0,
        "event_types": {
            # mq_type: (fmin, fmax, rise range s, decay range s, snr range)
            "impact_mq": (0.4, 1.2, (30, 150), (700, 1800), (2.0, 40.0)),
            "deep_mq": (0.5, 1.0, (20, 80), (250, 600), (1.5, 12.0)),
            "shallow_mq": (0.7, 2.0, (10, 60), (400, 1200), (3.0, 60.0)),
        },
        "glitch": {"count": (3, 8), "dur_s": (1.0, 6.0), "snr": (30.0, 300.0)},
        "burst": {"count": (1, 3), "dur_s": (100.0, 600.0), "band": (1.0, 3.0), "snr": (4.0, 15.0)},
    },
    "mars": {
        "fs": 20.0, "duration_s": 3600, "sigma": 1.0, "start": "2022-01-01",
        "file_step": pd.Timedelta(hours=2),
        "cols": ("time(%Y-%m-%dT%H:%M:%S.%f)", "rel_time(sec)", "velocity(c/s)"),
        "catalog_name": "Mars_InSight_catalog_synthetic.csv", "keep_extension": True,
        "edge_s": (600.0, 600.0), "min_sep_s": 900.0,
        "event_types": {
            "high_frequency": (1.5, 3.0, (2, 10), (30, 150), (2.0, 30.0)),
            "low_frequency": (0.5, 1.5, (5, 20), (40, 200), (1.5, 20.0)),
        },
        "glitch": {"count": (2, 6), "dur_s": (5.0, 25.0), "snr": (20.0, 200.0)},
        "burst": {"count": (1, 3), "dur_s": (60.0, 400.0), "band": (0.1, 0.8), "snr": (5.0, 20.0)},
    },
}


def _bandlimited(n: int, fs: float, fmin: float, fmax: float, rng) -> np.ndarray:
    fmax = min(fmax, 0.45 * fs)
    y = butterworth_filter(rng.standard_normal(n), fs, fmax, fmin, order=4)
    return y / (np.std(y) + 1e-30)


def _taper(n: int, frac: float = 0.1) -> np.ndarray:
    w = np.ones(n)
    m = max(1, int(n * frac))
    ramp = 0.5 * (1 - np.cos(np.linspace(0, np.pi, m)))
    w[:m] = ramp
    w[-m:] = ramp[::-1]
    return w


def _add(trace: np.ndarray, start: int, wave: np.ndarray) -> None:
    stop = min(len(trace), start + len(wave))
    if stop > start:
        trace[start:stop] += wave[: stop - start]


def synthesize_trace(planet: str, rng, duration_s: float | None = None):
    """Return (velocity, fs, list of (onset_s, mq_type))."""
    spec = PLANET_SPECS[planet]
    fs = spec["fs"]
    duration_s = float(duration_s or spec["duration_s"])
    n = int(duration_s * fs)
    t = np.arange(n) / fs
    sigma = spec["sigma"]

    # background: coloured noise + slow drift (removed by detrending)
    x = sigma * (0.8 * _bandlimited(n, fs, 0.05, 0.45 * fs, rng) + 0.3 * rng.standard_normal(n))
    x += 3 * sigma * np.sin(2 * np.pi * t / rng.uniform(2000, 6000) + rng.uniform(0, 2 * np.pi))
    x += sigma * rng.uniform(-2, 2) * t / duration_s

    # catalogued quakes
    lo, hi = spec["edge_s"][0], duration_s - spec["edge_s"][1]
    if hi <= lo:  # very short trace: keep the quake in the middle part
        lo, hi = 0.25 * duration_s, 0.6 * duration_s
    n_events = 1 if rng.random() < 0.75 else 2
    onsets: list[float] = []
    for _ in range(50):
        if len(onsets) == n_events:
            break
        cand = rng.uniform(lo, hi)
        if all(abs(cand - o) >= spec["min_sep_s"] for o in onsets):
            onsets.append(cand)
    events = []
    for onset in sorted(onsets):
        mq_type = rng.choice(list(spec["event_types"]))
        fmin, fmax, rise_rng, decay_rng, snr_rng = spec["event_types"][mq_type]
        rise, decay = rng.uniform(*rise_rng), rng.uniform(*decay_rng)
        snr = np.exp(rng.uniform(np.log(snr_rng[0]), np.log(snr_rng[1])))
        m = int(min(6 * decay, duration_s - onset) * fs)
        te = np.arange(m) / fs
        env = (1 - np.exp(-te / rise)) * np.exp(-te / decay)
        env /= env.max()
        _add(x, int(onset * fs), snr * sigma * env * _bandlimited(m, fs, fmin, fmax, rng))
        events.append((float(onset), str(mq_type)))

    # glitches (short, impulsive, one-sided) - NOT in the catalogue
    g = spec["glitch"]
    for _ in range(rng.integers(g["count"][0], g["count"][1] + 1)):
        dur = rng.uniform(*g["dur_s"])
        m = max(3, int(dur * fs))
        pulse = np.exp(-0.5 * ((np.arange(m) - m / 2) / (m / 6)) ** 2)
        amp = rng.uniform(*g["snr"]) * sigma * rng.choice([-1, 1])
        _add(x, int(rng.uniform(0, duration_s - dur) * fs), amp * pulse)

    # box-shaped noise bursts (wind / thermal / instrument) - NOT in the catalogue
    b = spec["burst"]
    for _ in range(rng.integers(b["count"][0], b["count"][1] + 1)):
        dur = rng.uniform(*b["dur_s"])
        m = int(dur * fs)
        burst = _taper(m, 0.15) * _bandlimited(m, fs, b["band"][0], b["band"][1], rng)
        _add(x, int(rng.uniform(0, duration_s - dur) * fs), rng.uniform(*b["snr"]) * sigma * burst)

    # a few missing samples (exercise the cleaning step)
    x[rng.choice(n, size=max(1, n // 2000), replace=False)] = np.nan
    return x, fs, events


def generate_synthetic_dataset(out_dir: str, planet: str = "lunar", n_files: int = 20,
                               duration_s: float | None = None, seed: int = 0) -> tuple[str, str]:
    """Write ``n_files`` synthetic CSVs + a catalogue; return (data_dir, catalog_path)."""
    spec = PLANET_SPECS[planet]
    rng = np.random.default_rng(seed)
    data_dir = os.path.join(out_dir, planet, "data")
    os.makedirs(data_dir, exist_ok=True)
    abs_col, rel_col, vel_col = spec["cols"]
    catalog_rows = []
    evid = 0
    start0 = pd.Timestamp(spec["start"])
    for i in range(n_files):
        velocity, fs, events = synthesize_trace(planet, rng, duration_s)
        start = start0 + i * spec["file_step"]
        t_rel = np.arange(len(velocity)) / fs
        abs_times = start + pd.to_timedelta(t_rel, unit="s")
        evid += 1
        stem = f"synthetic_{planet}_{i + 1:03d}_evid{evid:05d}"
        df = pd.DataFrame({
            abs_col: abs_times.strftime("%Y-%m-%dT%H:%M:%S.%f"),
            rel_col: t_rel,
            vel_col: velocity,
        })
        df.to_csv(os.path.join(data_dir, f"{stem}.csv"), index=False)
        for onset, mq_type in events:
            catalog_rows.append({
                "filename": f"{stem}.csv" if spec["keep_extension"] else stem,
                "time_abs(%Y-%m-%dT%H:%M:%S.%f)": (start + pd.to_timedelta(onset, unit="s")).strftime(
                    "%Y-%m-%dT%H:%M:%S.%f"),
                "time_rel(sec)": onset,
                "evid": f"evid{evid:05d}",
                "mq_type": mq_type,
            })
    catalog_path = os.path.join(out_dir, planet, spec["catalog_name"])
    pd.DataFrame(catalog_rows).to_csv(catalog_path, index=False)
    return data_dir, catalog_path
