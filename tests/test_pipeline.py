"""Tests for the STA/LTA + ML pipeline. Run with:  pytest -q"""

import os

import joblib
import numpy as np
import pandas as pd
import pytest

from seismic_ml.config import get_config
from seismic_ml.dataset import build_dataset, file_stem, load_catalog
from seismic_ml.detect import detect_directory
from seismic_ml.features import FEATURE_NAMES
from seismic_ml.preprocessing import load_trace, preprocess
from seismic_ml.sta_lta import run_sta_lta, sta_lta_ratio, trigger_onsets
from seismic_ml.synthetic import generate_synthetic_dataset
from seismic_ml.training import train_and_evaluate


# ----------------------------------------------------------------------------- unit tests
def test_sta_lta_is_about_one_for_stationary_noise():
    rng = np.random.default_rng(0)
    x = rng.standard_normal(20000)
    r = sta_lta_ratio(x, 100, 1000)
    assert np.all(r[:999] == 0)                 # LTA warm-up
    assert abs(np.median(r[1000:]) - 1.0) < 0.1


def test_sta_lta_matches_brute_force():
    rng = np.random.default_rng(1)
    x = rng.standard_normal(500)
    r = sta_lta_ratio(x, 10, 50)
    i = 321
    expected = np.mean(x[i - 9:i + 1] ** 2) / np.mean(x[i - 49:i + 1] ** 2)
    assert r[i] == pytest.approx(expected)


def test_trigger_detects_burst_and_merges_close_triggers():
    rng = np.random.default_rng(2)
    x = rng.standard_normal(30000)
    x[15000:15500] *= 20                        # strong burst
    cfg = get_config("lunar", sta_sec=50, lta_sec=500, min_gap_sec=200)
    det = run_sta_lta(x, fs=1.0, cfg=cfg)
    assert any(14950 <= t.onset <= 15100 for t in det.triggers)

    ratio = np.zeros(100)
    ratio[[10, 11, 15, 16, 60]] = 5.0
    merged = trigger_onsets(ratio, thr_on=2.0, thr_off=1.0, min_gap=5)
    assert [(t.onset, t.offset) for t in merged] == [(10, 17), (60, 61)]


def test_file_stem_keeps_dots_in_lunar_names():
    assert file_stem("a/b/xa.s12.00.mhz.1970-01-19HR00_evid00002.csv") == "xa.s12.00.mhz.1970-01-19HR00_evid00002"
    assert file_stem("xa.s12.00.mhz.1970-01-19HR00_evid00002") == "xa.s12.00.mhz.1970-01-19HR00_evid00002"
    assert file_stem("XB.ELYSE.02.BHV.2022-02-03HR08_evid0005.csv") == "XB.ELYSE.02.BHV.2022-02-03HR08_evid0005"


@pytest.mark.parametrize("planet", ["lunar", "mars"])
def test_load_trace_reads_both_csv_layouts(tmp_path, planet):
    data_dir, catalog = generate_synthetic_dataset(str(tmp_path), planet, n_files=1, duration_s=1800, seed=3)
    path = os.path.join(data_dir, os.listdir(data_dir)[0])
    tr = load_trace(path)
    assert np.isfinite(tr.velocity).all()       # NaNs were interpolated
    assert tr.fs == pytest.approx(6.625 if planet == "lunar" else 20.0, rel=1e-6)
    assert tr.start_time is not None
    x = preprocess(tr.velocity, tr.fs, get_config(planet))
    assert np.max(np.abs(x)) == pytest.approx(1.0)
    cat = load_catalog(catalog)
    assert file_stem(tr.file) in set(cat["stem"])


# ----------------------------------------------------------------------------- end-to-end
@pytest.fixture(scope="module")
def synthetic_features(tmp_path_factory):
    root = tmp_path_factory.mktemp("synthetic")
    frames, dirs = [], {}
    for i, planet in enumerate(["lunar", "mars"]):
        duration = 3 * 3600 if planet == "lunar" else 3600
        data_dir, catalog = generate_synthetic_dataset(str(root), planet, n_files=8,
                                                       duration_s=duration, seed=10 + i)
        frames.append(build_dataset(data_dir, catalog, get_config(planet), verbose=False))
        dirs[planet] = data_dir
    return pd.concat(frames, ignore_index=True), dirs, root


def test_feature_table_is_complete(synthetic_features):
    df, _, _ = synthetic_features
    assert set(FEATURE_NAMES) <= set(df.columns)
    assert df[FEATURE_NAMES].notna().all().all()
    assert set(df["label"]) == {0, 1}
    assert {"sta_lta_trigger", "catalog", "random_noise"} <= set(df["source"])


def test_train_evaluate_and_detect(synthetic_features):
    df, dirs, root = synthetic_features
    out = os.path.join(str(root), "outputs")
    result = train_and_evaluate(df, out, model_names=["XGBoost", "Random Forest", "PCA + LogReg"],
                                cv_folds=3, verbose=False)

    for name in ["model_comparison.csv", "cross_validation.csv", "summary.md", "roc_curves.png",
                 "model_comparison.png", "confusion_matrices.png", "feature_importance.png"]:
        assert os.path.exists(os.path.join(out, "results", name)), name

    test = result["test"].set_index("model")
    assert "STA/LTA only" in test.index
    assert test.drop(index="STA/LTA only")["accuracy"].max() > 0.7

    bundle = joblib.load(result["model_path"])
    assert bundle["feature_names"] == FEATURE_NAMES
    assert set(bundle["planets"]) == {"lunar", "mars"}

    det_dir = os.path.join(out, "detections")
    events = detect_directory(dirs["lunar"], result["model_path"], "lunar", det_dir, make_plots=True)
    assert os.path.exists(os.path.join(det_dir, "detected_events.csv"))
    assert len(os.listdir(os.path.join(det_dir, "plots"))) == 8
    assert set(events["classification"]) <= {"major", "moderate", "minor", "questionable", "noise"}
    assert events["ml_probability"].between(0, 1).all()
