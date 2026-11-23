# Seismic Event Detection Across the Solar System using STA/LTA and Machine Learning

## Overview

This project addresses the challenge of identifying seismic events within noisy planetary seismic data, such as that collected by the Apollo missions and the Mars InSight Lander. Transmitting continuous seismic data from a distant lander to Earth is power-intensive, and not all of the data is scientifically useful. Therefore, this project focuses on creating a solution that can distinguish meaningful seismic events from noise and only transmit relevant data back to Earth.

The project has two phases:

1. **Phase 1 – STA/LTA detection** (`lunar_event_detection.py`, `mars_seismic_detection.py`, `demofile.py`):
   a classical Short-Term Average / Long-Term Average detector with a dynamic threshold.
2. **Phase 2 – Hybrid STA/LTA + Machine Learning** (`seismic_ml/` package): STA/LTA proposes candidate
   events, and supervised models (**XGBoost, LightGBM, Random Forest, Decision Tree, PCA + Logistic
   Regression, Deep Neural Net (MLP), SVM, KNN**) decide whether each candidate is a real quake or noise.
   Confirmed events are graded **major / moderate / minor / questionable**.

## Objective

The goal is to create an algorithm capable of identifying seismic events within the data. The algorithm is designed to process planetary seismic data from missions like Apollo and Mars InSight, filtering out noise, detecting seismic quakes, and highlighting significant events.

### Key Features
- **Preprocessing**: cleaning + linear interpolation of missing values, resampling to a uniform grid, detrending, zero-phase Butterworth filtering, amplitude normalisation.
- **STA/LTA detection**: sliding-window energy ratio with a dynamic threshold `mu + k*sigma`, on/off triggering and merging of nearby triggers.
- **Feature extraction**: 21 features per candidate window (duration, RMS, peak amplitude, mean velocity, STA/LTA peak, skewness, kurtosis, energy, SNR, spectral centroid, ...).
- **Machine learning**: 8 classifiers trained and compared with a 70/30 split, 5-fold cross-validation, accuracy / precision / recall / F1 / ROC-AUC, and a direct comparison against the STA/LTA-only detector.
- **Visualisation & reporting**: ROC curves, metric comparison charts, confusion matrices, feature importance, 3-panel detection plots, CSV reports.

## Data Sources

- **Apollo Seismic Data**: continuous seismic records from the Apollo Passive Seismic Experiment (1969–1977).
- **Mars InSight Lander Data**: SEIS seismometer recordings from the InSight lander (2018–2022).

Both come from the [NASA Space Apps 2024 "Seismic Detection Across the Solar System"](https://www.spaceappschallenge.org/2024/challenges/2024-challenges/seismic-detection-across-the-solar-system/) data package (`space_apps_2024_seismic_detection.zip`, ~1 GB, not stored in this repository). After unzipping, the relevant folders are:

```
space_apps_2024_seismic_detection/data/
├── lunar/training/data/S12_GradeA/*.csv                              # Apollo 12 training traces
├── lunar/training/catalogs/apollo12_catalog_GradeA_final.csv         # event arrival times + mq_type
├── lunar/test/data/...                                               # unlabelled test traces
├── mars/training/data/*.csv                                          # InSight training traces
├── mars/training/catalogs/Mars_InSight_training_catalog_final.csv
└── mars/test/data/...
```

CSV formats (both are detected automatically):

| Dataset | Columns |
|---|---|
| Lunar | `time_abs(%Y-%m-%dT%H:%M:%S.%f)`, `time_rel(sec)`, `velocity(m/s)` |
| Mars  | `time(%Y-%m-%dT%H:%M:%S.%f)`, `rel_time(sec)`, `velocity(c/s)` |

---

## Machine Learning Pipeline (Phase 2)

```
 raw CSV ──► preprocessing ──► STA/LTA triggers ──► feature extraction ──► ML classifier ──► labels
            (clean, resample,    (mu + k*sigma        (21 features per       (XGBoost, LightGBM,   (major / moderate /
             detrend, filter,     dynamic threshold)   candidate window)      RF, DT, PCA+LR, ...)  minor / questionable)
             normalise)
```

### Repository structure

```
├── seismic_ml/                  # the ML package
│   ├── config.py                # lunar / Mars parameters (filters, STA/LTA windows, k, window sizes)
│   ├── preprocessing.py         # CSV loading, cleaning, resampling, detrend, Butterworth filter, normalise
│   ├── sta_lta.py               # STA/LTA ratio, dynamic threshold, trigger on/off + merging
│   ├── features.py              # 21 features per candidate window
│   ├── dataset.py               # builds the labelled training table from traces + catalogue
│   ├── models.py                # XGBoost, LightGBM, RF, DT, PCA+LogReg, MLP, SVM, KNN
│   ├── training.py              # split, cross-validation, metrics, STA/LTA baseline, model saving
│   ├── detect.py                # hybrid STA/LTA + ML detection on new recordings
│   ├── plotting.py              # all figures
│   └── synthetic.py             # synthetic data generator for demos/tests
├── build_features.py            # step 1: traces + catalogue  -> feature CSV
├── train_models.py              # step 2: feature CSV(s)      -> model comparison + best model
├── detect_events.py             # step 3: new traces          -> classified events + plots
├── make_synthetic_data.py       # optional: data for a quick test run without the NASA package
├── tests/test_pipeline.py       # pytest suite
├── lunar_event_detection.py     # phase 1 STA/LTA scripts
├── mars_seismic_detection.py
├── demofile.py
└── training_dataset/            # phase 1 detection results and plots
```

### Installation

```bash
python -m venv .venv && source .venv/bin/activate     # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### Step 1 – Build the labelled feature table

```bash
python build_features.py --planet lunar \
    --data-dir space_apps_2024_seismic_detection/data/lunar/training/data/S12_GradeA \
    --catalog  space_apps_2024_seismic_detection/data/lunar/training/catalogs/apollo12_catalog_GradeA_final.csv \
    --out outputs/features_lunar.csv

python build_features.py --planet mars \
    --data-dir space_apps_2024_seismic_detection/data/mars/training/data \
    --catalog  space_apps_2024_seismic_detection/data/mars/training/catalogs/Mars_InSight_training_catalog_final.csv \
    --out outputs/features_mars.csv
```

How the labels are made (for every catalogued trace):

| Sample source | Label |
|---|---|
| STA/LTA trigger within ±tolerance of a catalogued arrival | **event (1)** – true detection |
| STA/LTA trigger far away from every catalogued arrival | **noise (0)** – false alarm |
| Window on a catalogued arrival (+ jittered copies) | **event (1)** – includes quakes STA/LTA missed |
| Random window in a quiet part of the trace | **noise (0)** |

Triggers inside a quake's coda are ambiguous and skipped. Each row also stores its source file so that
the train/test split can be done **per recording** (no overlapping windows on both sides of the split).

### Step 2 – Train and compare the models

```bash
python train_models.py --features outputs/features_lunar.csv outputs/features_mars.csv
# or a subset:  --models XGBoost LightGBM "Random Forest"
```

Evaluation protocol:
- 70 % / 30 % train/test split, grouped by recording file
- 5-fold cross-validation on the training part (stratified, grouped by file)
- Accuracy, precision, recall, F1-score and ROC-AUC for every model
- The **STA/LTA-only** detector (event if the ratio crossed the dynamic threshold) is scored on exactly the same windows
- The best model (highest mean cross-validated F1) is refit on all labelled data and saved

Outputs in `outputs/`:

| File | Content |
|---|---|
| `results/model_comparison.csv` / `summary.md` | test-set metrics for all models + STA/LTA baseline |
| `results/cross_validation.csv` | 5-fold CV mean ± std |
| `results/roc_curves.png` | ROC curves of all models |
| `results/model_comparison.png` | accuracy / precision / recall / F1 charts |
| `results/confusion_matrices.png` | confusion matrix per model |
| `results/feature_importance.png` | most useful features (tree model) |
| `results/pca_explained_variance.png` | PCA cumulative explained variance |
| `results/per_planet_metrics.csv` | best model's metrics on lunar vs. Martian windows |
| `models/best_model.joblib` | trained model + feature list + planet configs |

### Step 3 – Detect and classify events in new recordings

```bash
python detect_events.py --planet lunar \
    --input space_apps_2024_seismic_detection/data/lunar/test/data/S12_GradeB \
    --model outputs/models/best_model.joblib
```

Every STA/LTA trigger is scored by the model:

| ML probability | Event-window SNR | Label |
|---|---|---|
| < 0.5 | – | `noise` (false alarm, discarded) |
| 0.5 – 0.7 | – | `questionable` |
| ≥ 0.7 | ≥ 20 dB | `major` |
| ≥ 0.7 | 10 – 20 dB | `moderate` |
| ≥ 0.7 | < 10 dB | `minor` |

All thresholds are command-line options (`--reject-threshold`, `--confident-threshold`, `--major-snr-db`, `--moderate-snr-db`).
Outputs: `detected_events.csv`, `all_triggers_classified.csv` and a 3-panel plot per file
(raw waveform / STA/LTA ratio with threshold / classified events) in `outputs/detections_<planet>/`.

### Features

| Feature | Description |
|---|---|
| `duration_s` | time the smoothed envelope stays above 3× the noise level |
| `rms_amplitude`, `peak_amplitude` | RMS and maximum absolute amplitude of the window |
| `mean_abs_velocity` | mean absolute velocity |
| `sta_lta_peak`, `sta_lta_mean`, `sta_lta_peak_over_threshold`, `time_above_threshold_s` | STA/LTA statistics |
| `waveform_skewness`, `waveform_kurtosis`, `envelope_skewness`, `envelope_kurtosis` | shape statistics |
| `energy` | integral of the squared amplitude |
| `snr_db` | window RMS relative to the robust (MAD) noise level of the trace |
| `crest_factor` | peak / RMS – large for spikes and glitches |
| `rise_time_s`, `coda_energy_fraction` | envelope build-up and decay |
| `zero_crossing_rate`, `dominant_freq_hz`, `spectral_centroid_hz`, `spectral_bandwidth_hz` | spectral content |

### Planet settings

Defaults live in `seismic_ml/config.py` and can be overridden from the command line (`--sta-sec`, `--lta-sec`, `--k`, `--window-sec`).

| Setting | Lunar | Mars |
|---|---|---|
| Butterworth band | 0.5 – 1.0 Hz | 0.5 – 3.0 Hz |
| STA / LTA window | 120 s / 600 s | 30 s / 500 s |
| Threshold `mu + k*sigma` | k = 3.0 | k = 2.8 |
| Feature window | 1800 s | 300 s |

### Quick test without the NASA data

```bash
python make_synthetic_data.py --out-dir data/synthetic
python build_features.py --planet lunar --data-dir data/synthetic/lunar/data --catalog data/synthetic/lunar/apollo12_catalog_synthetic.csv
python build_features.py --planet mars  --data-dir data/synthetic/mars/data  --catalog data/synthetic/mars/Mars_InSight_catalog_synthetic.csv
python train_models.py --features outputs/features_lunar.csv outputs/features_mars.csv
python detect_events.py --planet lunar --input data/synthetic/lunar/data
```

> Synthetic traces only check that the code runs end to end. Scores measured on them say nothing about
> real Apollo/InSight performance – always report the numbers produced on the real data.

### Tests

```bash
pytest -q
```

### Notes and limitations
- The Mars training set in the Space Apps package contains only a couple of labelled recordings, so Mars is best trained together with the lunar data (as in Step 2). With fewer than 4 recordings the code falls back to a row-wise split and prints a warning.
- Catalogues only list the main events; un-catalogued quakes in a trace end up labelled as noise, which caps the measurable precision.
- All random steps use a fixed seed (`--seed`), so results are reproducible.

---

## Phase 1 – Running the original STA/LTA scripts

Requirements: `pip install pandas numpy matplotlib`

- `lunar_event_detection.py` – batch STA/LTA detection over a folder of lunar CSV files (set `data_dir` at the top of the script), writes `detected_event.csv`.
- `mars_seismic_detection.py` – the same for Mars data (different sampling rate, windows and `k`), also saves STA/LTA plots.
- `demofile.py` – single-file demo with a fixed threshold and plots.

Outputs: detected events (absolute time, relative time, file) and STA/LTA plots, as found in `training_dataset/`.

## License
This project is open-source. Feel free to modify and adapt it to suit your research needs.

---

## Contact

For further information or questions, feel free to contact:

- **Email**: megx0108@gmail.com
- **GitHub**: https://github.com/meg-x08/Seismic_event_detection_across_solar_system_using_STA-LTA

---

## Conclusion

This project provides an efficient solution for analyzing planetary seismic data, helping identify seismic events while minimizing data transmission. By combining the interpretability of STA/LTA with machine-learning classifiers that reject noise-induced triggers, the hybrid system reduces false alarms and grades the remaining events, which can significantly reduce the power and bandwidth requirements of planetary seismology missions.
