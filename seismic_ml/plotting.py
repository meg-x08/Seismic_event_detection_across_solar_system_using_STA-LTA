"""All figures produced by the pipeline (matplotlib, saved as PNG).

* ROC curves of every model + the STA/LTA baseline         (report Fig. 4.4 / 5.1)
* Accuracy / precision / recall / F1 comparison             (report Table 5.1)
* Confusion matrices, feature importance, PCA variance
* Three-panel STA/LTA + ML detection plot per recording     (report Fig. 4.2 / 5.2)

Colours: one fixed colour per model (colour follows the model, not its rank);
the STA/LTA baseline is always neutral grey.
"""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")  # headless backend: works on servers / CI

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402

from .models import MODEL_ORDER  # noqa: E402

SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_2 = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
AXIS = "#c3c2b7"
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
SEQ_BLUE = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]
BASELINE_NAME = "STA/LTA only"

MODEL_COLORS = {name: SERIES[i % len(SERIES)] for i, name in enumerate(MODEL_ORDER)}
MODEL_COLORS[BASELINE_NAME] = MUTED

# Event classes: ordinal blue ramp for strength, orange for "questionable"
CLASS_COLORS = {
    "major": "#104281",
    "moderate": "#2a78d6",
    "minor": "#86b6ef",
    "questionable": "#eb6834",
    "noise": MUTED,
}


def apply_style() -> None:
    plt.rcParams.update({
        "figure.facecolor": SURFACE,
        "axes.facecolor": SURFACE,
        "savefig.facecolor": SURFACE,
        "font.family": "sans-serif",
        "font.sans-serif": ["DejaVu Sans", "Segoe UI", "Helvetica", "Arial"],
        "font.size": 10,
        "axes.edgecolor": AXIS,
        "axes.labelcolor": INK_2,
        "axes.titlecolor": INK,
        "axes.titlesize": 11,
        "axes.titleweight": "bold",
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "axes.axisbelow": True,
        "grid.color": GRID,
        "grid.linestyle": "-",
        "grid.linewidth": 0.6,
        "xtick.color": MUTED,
        "ytick.color": MUTED,
        "xtick.labelcolor": INK_2,
        "ytick.labelcolor": INK_2,
        "legend.frameon": False,
        "legend.labelcolor": INK_2,
        "lines.linewidth": 2.0,
    })


def _save(fig, path: str) -> None:
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


# ----------------------------------------------------------------------------- evaluation plots
def plot_roc_curves(curves: dict, path: str, title: str = "ROC curves (held-out test set)") -> None:
    """``curves``: {name: (fpr, tpr, auc)}; the STA/LTA baseline is drawn dashed grey."""
    apply_style()
    fig, ax = plt.subplots(figsize=(7, 6))
    ax.plot([0, 1], [0, 1], color=AXIS, linewidth=1, label="_chance")
    names = [n for n in MODEL_ORDER if n in curves] + [n for n in curves if n not in MODEL_ORDER]
    for name in names:
        fpr, tpr, auc = curves[name]
        is_base = name == BASELINE_NAME
        ax.plot(fpr, tpr, color=MODEL_COLORS.get(name, INK_2),
                linestyle="--" if is_base else "-", linewidth=2,
                label=f"{name} (AUC = {auc:.3f})", zorder=2 if is_base else 3)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.01)
    ax.set_xlabel("False positive rate")
    ax.set_ylabel("True positive rate")
    ax.set_title(title, loc="left")
    ax.legend(loc="lower right", fontsize=9)
    _save(fig, path)


def plot_metric_comparison(results: pd.DataFrame, path: str) -> None:
    """Small multiples (one panel per metric), models sorted by F1."""
    apply_style()
    metrics = [("accuracy", "Accuracy"), ("precision", "Precision"), ("recall", "Recall"), ("f1", "F1-score")]
    order = results.sort_values("f1", ascending=True)["model"].tolist()
    fig, axes = plt.subplots(1, 4, figsize=(15, 0.45 * len(order) + 1.6), sharey=True)
    for ax, (col, label) in zip(axes, metrics):
        vals = results.set_index("model").loc[order, col].to_numpy(dtype=float)
        colors = [MUTED if m == BASELINE_NAME else SERIES[0] for m in order]
        ax.barh(order, vals, color=colors, height=0.6)
        for y, v in enumerate(vals):
            ax.text(min(v, 1.0) + 0.01, y, f"{v:.3f}", va="center", fontsize=8, color=INK_2)
        ax.set_xlim(0, 1.15)
        ax.set_xticks([0, 0.25, 0.5, 0.75, 1.0])
        ax.set_title(label, loc="left")
        ax.grid(axis="y", visible=False)
    fig.suptitle("Model performance on the held-out test set (grey = STA/LTA baseline)",
                 x=0.01, ha="left", fontsize=12, fontweight="bold", color=INK)
    fig.tight_layout()
    _save(fig, path)


def plot_confusion_matrices(matrices: dict, path: str) -> None:
    apply_style()
    names = [n for n in MODEL_ORDER if n in matrices] + [n for n in matrices if n not in MODEL_ORDER]
    ncols = min(3, len(names))
    nrows = int(np.ceil(len(names) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(3.6 * ncols, 3.3 * nrows), squeeze=False)
    cmap = LinearSegmentedColormap.from_list("seq_blue", SEQ_BLUE)
    for ax in axes.flat[len(names):]:
        ax.axis("off")
    for ax, name in zip(axes.flat, names):
        cm = np.asarray(matrices[name])
        ax.imshow(cm, cmap=cmap, vmin=0, vmax=max(cm.max(), 1))
        for (i, j), v in np.ndenumerate(cm):
            dark = v > 0.55 * cm.max()
            ax.text(j, i, str(int(v)), ha="center", va="center", fontsize=11,
                    color="#ffffff" if dark else INK)
        ax.set_xticks([0, 1], ["noise", "event"])
        ax.set_yticks([0, 1], ["noise", "event"])
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")
        ax.set_title(name, loc="left", fontsize=10)
        ax.grid(False)
    fig.tight_layout()
    _save(fig, path)


def plot_feature_importance(names, importances, model_name: str, path: str, top: int = 15) -> None:
    apply_style()
    imp = pd.Series(importances, index=names).sort_values(ascending=True).tail(top)
    fig, ax = plt.subplots(figsize=(7, 0.35 * len(imp) + 1.2))
    ax.barh(imp.index, imp.values, color=SERIES[0], height=0.6)
    ax.set_xlabel("Relative importance")
    ax.set_title(f"Top {len(imp)} features - {model_name}", loc="left")
    ax.grid(axis="y", visible=False)
    _save(fig, path)


def plot_pca_variance(explained_ratio: np.ndarray, path: str) -> None:
    apply_style()
    cum = np.cumsum(explained_ratio)
    comps = np.arange(1, len(cum) + 1)
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(comps, cum, color=SERIES[0], marker="o", markersize=5)
    ax.axhline(0.95, color=INK_2, linestyle="--", linewidth=1)
    ax.text(comps[-1], 0.955, "95 % variance", ha="right", va="bottom", fontsize=8, color=INK_2)
    ax.set_xlabel("Number of principal components")
    ax.set_ylabel("Cumulative explained variance")
    ax.set_ylim(0, 1.05)
    ax.set_title("PCA on the seismic feature set", loc="left")
    _save(fig, path)


# ----------------------------------------------------------------------------- detection plot
def _minmax_decimate(t: np.ndarray, y: np.ndarray, max_points: int = 20000):
    """Keep the visual envelope of a long trace while plotting far fewer points."""
    n = len(y)
    if n <= max_points:
        return t, y
    bucket = int(np.ceil(n / (max_points / 2)))
    m = n // bucket
    yb = y[: m * bucket].reshape(m, bucket)
    tb = t[: m * bucket].reshape(m, bucket)
    lo, hi = yb.argmin(axis=1), yb.argmax(axis=1)
    rows = np.arange(m)
    first = np.minimum(lo, hi)
    second = np.maximum(lo, hi)
    t_out = np.column_stack([tb[rows, first], tb[rows, second]]).ravel()
    y_out = np.column_stack([yb[rows, first], yb[rows, second]]).ravel()
    return t_out, y_out


def plot_detection(analysed, events: pd.DataFrame, path: str, planet: str = "") -> None:
    """Three-panel figure: raw waveform, STA/LTA ratio, classified events."""
    apply_style()
    tr, det, x = analysed.trace, analysed.det, analysed.x
    t = tr.time_rel
    fig, axes = plt.subplots(3, 1, figsize=(14, 10), sharex=True)

    # Panel 1 - raw waveform with event onsets
    ax = axes[0]
    td, vd = _minmax_decimate(t, tr.velocity)
    ax.plot(td, vd, color=INK_2, linewidth=0.5)
    kept = events[events["classification"] != "noise"] if len(events) else events
    for _, ev in kept.iterrows():
        ax.axvline(ev["onset_time_rel"], color=CLASS_COLORS[ev["classification"]], linewidth=1.5)
    ax.set_ylabel(f"Velocity ({tr.units})" if tr.units else "Velocity")
    ax.set_title("Raw seismic data with detected event onsets", loc="left")

    # Panel 2 - STA/LTA ratio, thresholds and triggers
    ax = axes[1]
    td, rd = _minmax_decimate(t, det.ratio)
    ax.plot(td, rd, color=SERIES[0], linewidth=1.0, label="STA/LTA ratio")
    ax.axhline(det.threshold, color=INK_2, linestyle="--", linewidth=1.2,
               label=f"Dynamic threshold (mu + k*sigma) = {det.threshold:.2f}")
    if len(events):
        for _, ev in events.iterrows():
            ax.axvspan(ev["onset_time_rel"], ev["end_time_rel"], color=GRID, alpha=0.6, linewidth=0)
        rejected = events[events["classification"] == "noise"]
        accepted = events[events["classification"] != "noise"]
        if len(rejected):
            ax.scatter(rejected["peak_time_rel"], rejected["sta_lta_peak"], marker="x", s=60,
                       color=MUTED, linewidths=2, label="Trigger rejected by ML (noise)", zorder=4)
        if len(accepted):
            ax.scatter(accepted["peak_time_rel"], accepted["sta_lta_peak"], marker="o", s=64,
                       color=[CLASS_COLORS[c] for c in accepted["classification"]],
                       edgecolors=SURFACE, linewidths=1.5, label="Trigger confirmed by ML", zorder=5)
    ax.set_ylabel("STA/LTA ratio")
    ax.set_title("STA/LTA detection analysis", loc="left")
    ax.legend(loc="lower right", bbox_to_anchor=(1.0, 1.0), ncol=4, fontsize=8, borderaxespad=0.2)

    # Panel 3 - filtered waveform with classified event windows
    ax = axes[2]
    td, xd = _minmax_decimate(t, x)
    ax.plot(td, xd, color=MUTED, linewidth=0.5)
    for _, ev in kept.iterrows():
        color = CLASS_COLORS[ev["classification"]]
        ax.axvspan(ev["onset_time_rel"], ev["end_time_rel"], color=color, alpha=0.25, linewidth=0)
        ax.text(ev["onset_time_rel"], 0.92, f" {ev['classification'].capitalize()} "
                f"(p={ev['ml_probability']:.2f})", transform=ax.get_xaxis_transform(),
                fontsize=8, color=INK, va="top", ha="left",
                bbox=dict(boxstyle="round,pad=0.2", facecolor=SURFACE, edgecolor=color, linewidth=1.2))
    counts = kept["classification"].value_counts().to_dict() if len(kept) else {}
    summary = [
        "Detection summary",
        f"Dataset: {planet.upper() or '-'}",
        f"Duration: {tr.duration:,.0f} s   fs = {tr.fs:.3f} Hz",
        f"STA/LTA triggers: {len(events)}",
        f"Confirmed events: {len(kept)}"
        + (" (" + ", ".join(f"{v} {k}" for k, v in counts.items()) + ")" if counts else ""),
        f"Threshold: {det.threshold:.2f}   max ratio: {det.ratio.max():.2f}",
    ]
    ax.text(0.01, 0.04, "\n".join(summary), transform=ax.transAxes, fontsize=8, color=INK_2,
            va="bottom", ha="left", family="monospace",
            bbox=dict(boxstyle="round,pad=0.4", facecolor=SURFACE, edgecolor=AXIS))
    ax.set_ylabel("Filtered, normalised")
    ax.set_xlabel("Time (seconds)")
    ax.set_title("Classified events (major / moderate / minor / questionable)", loc="left")

    fig.suptitle(f"STA/LTA + ML seismic analysis: {tr.file}", x=0.01, ha="left",
                 fontsize=13, fontweight="bold", color=INK)
    fig.tight_layout()
    _save(fig, path)
