"""Build a labelled feature table (one row per candidate window) for ML training.

For every trace that appears in the event catalogue three kinds of samples are
produced:

1. **STA/LTA triggers** - every trigger of the classical detector.  A trigger
   whose onset lies within ``match_tolerance_sec`` of a catalogued arrival is a
   *true event* (label 1); a trigger far away from every arrival is a *false
   alarm* (label 0).  Triggers inside an event's coda are ambiguous and skipped.
   These rows are what lets the ML model learn to reject STA/LTA false alarms.
2. **Catalogue events** - ``positives_per_event`` windows anchored on each
   catalogued arrival (with a small random jitter), so that events missed by
   STA/LTA are still represented (label 1).
3. **Random noise** - ``negatives_per_trace`` windows drawn from quiet parts of
   the trace, away from every catalogued event (label 0).

Every row carries the source file in the ``group`` column so that train/test
splits can be made per recording (no leakage between overlapping windows).
"""

from __future__ import annotations

import glob
import os
from dataclasses import dataclass

import numpy as np
import pandas as pd

from .config import PlanetConfig
from .features import FEATURE_NAMES, extract_features, robust_noise_level
from .preprocessing import Trace, load_trace, preprocess
from .sta_lta import DetectionResult, run_sta_lta

_KNOWN_EXTENSIONS = (".csv", ".mseed", ".miniseed", ".sac")


def file_stem(name: str) -> str:
    """File name without directory and without a known data extension.

    ``os.path.splitext`` is not used on purpose: lunar file names contain dots
    (``xa.s12.00.mhz.1970-01-19HR00_evid00002``).
    """
    base = os.path.basename(str(name).strip())
    for ext in _KNOWN_EXTENSIONS:
        if base.lower().endswith(ext):
            return base[: -len(ext)]
    return base


def load_catalog(path: str) -> pd.DataFrame:
    """Load a NASA Space Apps event catalogue into a normalised DataFrame.

    Output columns: ``stem, arrival_rel, arrival_abs, evid, mq_type``.
    """
    cat = pd.read_csv(path)
    cols = {c.lower(): c for c in cat.columns}
    fname_col = next((cols[c] for c in cols if "file" in c), None)
    rel_col = next((cols[c] for c in cols if "rel" in c), None)
    abs_col = next((cols[c] for c in cols if "time" in c and "rel" not in c), None)
    if fname_col is None or (rel_col is None and abs_col is None):
        raise ValueError(f"{path}: catalogue needs a filename column and a time column")

    out = pd.DataFrame({"stem": cat[fname_col].map(file_stem)})
    out["arrival_rel"] = pd.to_numeric(cat[rel_col], errors="coerce") if rel_col else np.nan
    out["arrival_abs"] = pd.to_datetime(cat[abs_col], errors="coerce") if abs_col else pd.NaT
    out["evid"] = cat[cols["evid"]] if "evid" in cols else ""
    out["mq_type"] = cat[cols["mq_type"]] if "mq_type" in cols else ""
    return out


def arrivals_for_trace(trace: Trace, catalog: pd.DataFrame) -> pd.DataFrame:
    """Catalogue rows that belong to ``trace``, with ``arrival_rel`` filled in."""
    rows = catalog[catalog["stem"] == file_stem(trace.file)].copy()
    missing = rows["arrival_rel"].isna() & rows["arrival_abs"].notna()
    if missing.any() and trace.start_time is not None:
        rows.loc[missing, "arrival_rel"] = (
            rows.loc[missing, "arrival_abs"] - trace.start_time
        ).dt.total_seconds()
    return rows.dropna(subset=["arrival_rel"]).sort_values("arrival_rel")


def find_data_files(data_dir: str) -> list[str]:
    """All CSV files below ``data_dir`` (recursive), sorted."""
    if os.path.isfile(data_dir):
        return [data_dir]
    return sorted(glob.glob(os.path.join(data_dir, "**", "*.csv"), recursive=True))


@dataclass
class AnalysedTrace:
    trace: Trace
    x: np.ndarray               # preprocessed, normalised waveform
    det: DetectionResult
    noise_sigma: float


def analyse_trace(trace: Trace, cfg: PlanetConfig) -> AnalysedTrace:
    """Preprocess a trace and run STA/LTA on it (shared by training and inference)."""
    x = preprocess(trace.velocity, trace.fs, cfg)
    det = run_sta_lta(x, trace.fs, cfg)
    return AnalysedTrace(trace, x, det, robust_noise_level(x))


def _make_row(at: AnalysedTrace, cfg: PlanetConfig, anchor_idx: int, label: int, source: str,
              arrivals: np.ndarray, mq_types: np.ndarray) -> dict:
    tr = at.trace
    t_anchor = float(tr.time_rel[anchor_idx])
    nearest, mq = np.nan, ""
    if len(arrivals):
        j = int(np.argmin(np.abs(arrivals - t_anchor)))
        nearest = float(t_anchor - arrivals[j])
        mq = mq_types[j] if label == 1 else ""
    row = {
        "planet": cfg.name,
        "file": tr.file,
        "group": f"{cfg.name}:{tr.file}",
        "anchor_time_rel": t_anchor,
        "anchor_time_abs": tr.abs_time(t_anchor),
        "label": int(label),
        "source": source,
        "offset_from_nearest_arrival_s": nearest,
        "mq_type": mq,
        "threshold": at.det.threshold,
    }
    row.update(extract_features(at.x, at.det, tr.fs, anchor_idx, cfg, at.noise_sigma))
    return row


def build_samples_for_trace(at: AnalysedTrace, cfg: PlanetConfig, events: pd.DataFrame,
                            rng: np.random.Generator, positives_per_event: int = 3,
                            negatives_per_trace: int = 10) -> list[dict]:
    tr, det = at.trace, at.det
    fs, t = tr.fs, tr.time_rel
    n = len(t)
    arrivals = events["arrival_rel"].to_numpy(dtype=float)
    mq_types = events["mq_type"].astype(str).to_numpy()
    rows: list[dict] = []

    def in_exclusion(time_s: float) -> bool:
        return bool(np.any((time_s >= arrivals - cfg.exclusion_before_sec) &
                           (time_s <= arrivals + cfg.exclusion_after_sec)))

    # 1) STA/LTA triggers -> true detections / false alarms
    for trig in det.triggers:
        t_on = float(t[trig.onset])
        dist = np.abs(arrivals - t_on).min() if len(arrivals) else np.inf
        if dist <= cfg.match_tolerance_sec:
            rows.append(_make_row(at, cfg, trig.onset, 1, "sta_lta_trigger", arrivals, mq_types))
        elif not in_exclusion(t_on):
            rows.append(_make_row(at, cfg, trig.onset, 0, "sta_lta_trigger", arrivals, mq_types))

    # 2) catalogue events (with jitter) -> positives
    for arrival in arrivals:
        for p in range(positives_per_event):
            jitter = 0.0 if p == 0 else rng.uniform(-cfg.jitter_sec, cfg.jitter_sec)
            idx = int(np.clip(round((arrival + jitter) * fs), 0, n - 1))
            rows.append(_make_row(at, cfg, idx, 1, "catalog", arrivals, mq_types))

    # 3) random noise windows away from every event -> negatives
    if negatives_per_trace > 0:
        lo = det.n_lta
        hi = n - int(round((cfg.window_sec - cfg.pre_event_sec) * fs))
        if hi > lo:
            candidates = np.arange(lo, hi)
            if len(arrivals):
                ct = t[candidates]
                bad = np.zeros(len(candidates), dtype=bool)
                for a in arrivals:
                    bad |= (ct >= a - cfg.exclusion_before_sec) & (ct <= a + cfg.exclusion_after_sec)
                candidates = candidates[~bad]
            if len(candidates):
                k = min(negatives_per_trace, len(candidates))
                for idx in rng.choice(candidates, size=k, replace=False):
                    rows.append(_make_row(at, cfg, int(idx), 0, "random_noise", arrivals, mq_types))
    return rows


def build_dataset(data_dir: str, catalog_path: str, cfg: PlanetConfig, positives_per_event: int = 3,
                  negatives_per_trace: int = 10, max_files: int | None = None, seed: int = 42,
                  verbose: bool = True) -> pd.DataFrame:
    """Process every catalogued trace in ``data_dir`` and return the feature table."""
    catalog = load_catalog(catalog_path)
    known = set(catalog["stem"])
    files = [f for f in find_data_files(data_dir) if file_stem(f) in known]
    if max_files:
        files = files[:max_files]
    if not files:
        raise FileNotFoundError(
            f"No CSV files in '{data_dir}' match the catalogue '{catalog_path}'."
        )

    rng = np.random.default_rng(seed)
    all_rows: list[dict] = []
    for i, path in enumerate(files, 1):
        try:
            trace = load_trace(path)
        except Exception as exc:  # corrupt file: report and continue
            print(f"  [skip] {os.path.basename(path)}: {exc}")
            continue
        at = analyse_trace(trace, cfg)
        events = arrivals_for_trace(trace, catalog)
        rows = build_samples_for_trace(at, cfg, events, rng, positives_per_event, negatives_per_trace)
        all_rows.extend(rows)
        if verbose:
            n_pos = sum(r["label"] for r in rows)
            print(f"  [{i:3d}/{len(files)}] {trace.file}: fs={trace.fs:.3f} Hz, "
                  f"{len(at.det.triggers)} STA/LTA triggers, {len(events)} catalogued event(s) "
                  f"-> {n_pos} positive / {len(rows) - n_pos} negative windows")

    df = pd.DataFrame(all_rows)
    meta = [c for c in df.columns if c not in FEATURE_NAMES]
    return df[meta + FEATURE_NAMES]
