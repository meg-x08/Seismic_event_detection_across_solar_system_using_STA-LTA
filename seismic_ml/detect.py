"""Hybrid detector: STA/LTA proposes candidate events, the trained ML model confirms them.

For every STA/LTA trigger the same feature vector used during training is
computed and passed to the saved model.  The model's probability decides
whether the trigger is kept, and confirmed events are graded by strength:

    probability <  reject_threshold      -> "noise"        (false alarm, discarded)
    probability <  confident_threshold   -> "questionable"
    otherwise, by signal-to-noise ratio of the event window (dB):
        >= major_snr_db                  -> "major"
        >= moderate_snr_db               -> "moderate"
        else                             -> "minor"

SNR is used for the strength grade instead of the STA/LTA peak because the
classic STA/LTA ratio saturates at LTA/STA (e.g. 600 s / 120 s = 5 on the Moon),
so it cannot tell a strong quake from a moderate one.
"""

from __future__ import annotations

import os
import warnings

import joblib
import numpy as np
import pandas as pd

from .config import PlanetConfig, get_config
from .dataset import AnalysedTrace, analyse_trace
from .features import extract_features, window_bounds
from .preprocessing import load_trace


def load_model_bundle(path: str) -> dict:
    bundle = joblib.load(path)
    if not isinstance(bundle, dict) or "model" not in bundle:
        raise ValueError(f"{path} is not a model bundle produced by train_models.py")
    return bundle


def config_from_bundle(bundle: dict, planet: str) -> PlanetConfig:
    base = get_config(planet)
    stored = bundle.get("configs", {}).get(base.name)
    if stored is None:
        warnings.warn(f"The model was not trained on '{base.name}' data "
                      f"(trained on {bundle.get('planets')}); using default {base.name} settings.")
        return base
    return PlanetConfig(**stored)


def strength_class(snr_db: float, major_snr_db: float, moderate_snr_db: float) -> str:
    if snr_db >= major_snr_db:
        return "major"
    if snr_db >= moderate_snr_db:
        return "moderate"
    return "minor"


def classify_triggers(at: AnalysedTrace, cfg: PlanetConfig, bundle: dict,
                      reject_threshold: float = 0.5, confident_threshold: float = 0.7,
                      major_snr_db: float = 20.0, moderate_snr_db: float = 10.0) -> pd.DataFrame:
    tr, det, x = at.trace, at.det, at.x
    columns = ["file", "event_id", "onset_time_abs", "onset_time_rel", "peak_time_rel",
               "end_time_rel", "duration_s", "sta_lta_peak", "threshold", "peak_over_threshold",
               "peak_velocity", "rms_amplitude", "snr_db", "ml_probability", "classification"]
    if not det.triggers:
        return pd.DataFrame(columns=columns)

    feature_names = bundle["feature_names"]
    feats = [extract_features(x, det, tr.fs, trig.onset, cfg, at.noise_sigma) for trig in det.triggers]
    X = pd.DataFrame(feats)[feature_names].astype(float)
    model = bundle["model"]
    proba = model.predict_proba(X)[:, 1] if hasattr(model, "predict_proba") else model.decision_function(X)

    rows = []
    for i, (trig, f, p) in enumerate(zip(det.triggers, feats, proba), 1):
        ratio_peak = trig.peak_ratio / det.threshold if det.threshold > 0 else 0.0
        if p < reject_threshold:
            label = "noise"
        elif p < confident_threshold:
            label = "questionable"
        else:
            label = strength_class(f["snr_db"], major_snr_db, moderate_snr_db)
        start, stop = window_bounds(trig.onset, len(x), tr.fs, cfg)
        end_idx = max(trig.offset, trig.onset + 1)
        rows.append({
            "file": tr.file,
            "event_id": i,
            "onset_time_abs": tr.abs_time(tr.time_rel[trig.onset]),
            "onset_time_rel": float(tr.time_rel[trig.onset]),
            "peak_time_rel": float(tr.time_rel[trig.peak]),
            "end_time_rel": float(tr.time_rel[min(end_idx, len(x) - 1)]),
            "duration_s": float((end_idx - trig.onset) / tr.fs),
            "sta_lta_peak": float(trig.peak_ratio),
            "threshold": float(det.threshold),
            "peak_over_threshold": float(ratio_peak),
            "peak_velocity": float(np.max(np.abs(tr.velocity[start:stop]))),
            "rms_amplitude": f["rms_amplitude"],
            "snr_db": f["snr_db"],
            "ml_probability": float(p),
            "classification": label,
        })
    return pd.DataFrame(rows, columns=columns)


def detect_file(path: str, bundle: dict, cfg: PlanetConfig, **kwargs) -> tuple[AnalysedTrace, pd.DataFrame]:
    trace = load_trace(path)
    at = analyse_trace(trace, cfg)
    return at, classify_triggers(at, cfg, bundle, **kwargs)


def detect_directory(input_path: str, model_path: str, planet: str, out_dir: str,
                     make_plots: bool = True, **kwargs) -> pd.DataFrame:
    from . import plotting
    from .dataset import find_data_files

    bundle = load_model_bundle(model_path)
    cfg = config_from_bundle(bundle, planet)
    files = find_data_files(input_path)
    if not files:
        raise FileNotFoundError(f"No CSV files found at '{input_path}'")
    plot_dir = os.path.join(out_dir, "plots")
    os.makedirs(plot_dir, exist_ok=True)

    print(f"Model: {bundle['model_name']} (trained on {bundle.get('planets')}), planet config: {cfg.name}")
    all_events = []
    for i, path in enumerate(files, 1):
        try:
            at, events = detect_file(path, bundle, cfg, **kwargs)
        except Exception as exc:
            print(f"  [skip] {os.path.basename(path)}: {exc}")
            continue
        kept = events[events["classification"] != "noise"]
        print(f"  [{i:3d}/{len(files)}] {at.trace.file}: {len(events)} STA/LTA trigger(s) -> "
              f"{len(kept)} confirmed by ML "
              f"({', '.join(f'{v} {k}' for k, v in kept['classification'].value_counts().items()) or 'none'})")
        if make_plots:
            plotting.plot_detection(at, events, os.path.join(plot_dir, f"{at.trace.file}_hybrid_plot.png"),
                                    planet=cfg.name)
        all_events.append(events)

    result = pd.concat(all_events, ignore_index=True) if all_events else pd.DataFrame()
    os.makedirs(out_dir, exist_ok=True)
    result.to_csv(os.path.join(out_dir, "all_triggers_classified.csv"), index=False)
    if len(result):
        result[result["classification"] != "noise"].to_csv(
            os.path.join(out_dir, "detected_events.csv"), index=False)
    else:
        result.to_csv(os.path.join(out_dir, "detected_events.csv"), index=False)
    return result
