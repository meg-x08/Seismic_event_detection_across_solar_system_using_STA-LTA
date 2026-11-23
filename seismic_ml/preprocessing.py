"""Loading and preprocessing of planetary seismic CSV files.

Supports both CSV layouts of the NASA Space Apps 2024 "Seismic Detection Across
the Solar System" data package:

* Lunar (Apollo 12):  ``time_abs(%Y-%m-%dT%H:%M:%S.%f), time_rel(sec), velocity(m/s)``
* Mars (InSight):     ``time(%Y-%m-%dT%H:%M:%S.%f), rel_time(sec), velocity(c/s)``

Preprocessing steps (report section 3.2.2-b / 4.2.3):
    1. data cleaning  - drop corrupt rows, linear interpolation of missing values
    2. resampling     - put irregularly sampled data on a uniform time grid
    3. detrending     - remove the linear baseline drift
    4. noise filtering - zero-phase Butterworth low-/band-pass filter
    5. normalisation  - scale the amplitude so that |x| lies in [0, 1]
"""

from __future__ import annotations

import os
from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy import signal

from .config import PlanetConfig


@dataclass
class Trace:
    """A single seismic recording on a uniform time grid."""

    file: str                    # file name (no directory)
    time_rel: np.ndarray         # seconds since the first sample
    velocity: np.ndarray         # cleaned velocity in the file's own units
    fs: float                    # sampling rate (Hz)
    start_time: pd.Timestamp | None = None
    units: str = ""

    @property
    def duration(self) -> float:
        return float(self.time_rel[-1] - self.time_rel[0]) if len(self.time_rel) else 0.0

    def abs_time(self, t_rel: float) -> pd.Timestamp | None:
        if self.start_time is None:
            return None
        return self.start_time + pd.to_timedelta(float(t_rel) - float(self.time_rel[0]), unit="s")


def _find_column(columns, *must_contain, exclude=()) -> str | None:
    for col in columns:
        low = col.lower()
        if all(m in low for m in must_contain) and not any(e in low for e in exclude):
            return col
    return None


def detect_columns(columns) -> tuple[str | None, str | None, str]:
    """Return (absolute-time column, relative-time column, velocity column)."""
    columns = list(columns)
    vel_col = _find_column(columns, "velocity")
    rel_col = _find_column(columns, "rel")
    abs_col = _find_column(columns, "time", exclude=("rel",))
    if vel_col is None:
        # Fall back to the last numeric-looking column
        vel_col = columns[-1]
    return abs_col, rel_col, vel_col


def load_trace(path: str, target_fs: float | None = None) -> Trace:
    """Read a seismic CSV file, clean it and return a uniformly sampled :class:`Trace`."""
    df = pd.read_csv(path, low_memory=False)
    abs_col, rel_col, vel_col = detect_columns(df.columns)

    velocity = pd.to_numeric(df[vel_col], errors="coerce")

    if rel_col is not None:
        t_rel = pd.to_numeric(df[rel_col], errors="coerce")
    elif abs_col is not None:
        abs_times = pd.to_datetime(df[abs_col], errors="coerce")
        t_rel = (abs_times - abs_times.dropna().iloc[0]).dt.total_seconds()
    else:
        raise ValueError(f"{path}: no relative or absolute time column found")

    frame = pd.DataFrame({"t": t_rel, "v": velocity})
    frame = frame.dropna(subset=["t"]).sort_values("t").drop_duplicates(subset="t")
    if frame["v"].notna().sum() < 2:
        raise ValueError(f"{path}: fewer than two valid velocity samples")

    # 1. cleaning: fill gaps / corrupt values by linear interpolation
    frame["v"] = frame["v"].interpolate(method="linear", limit_direction="both")

    t = frame["t"].to_numpy(dtype=float)
    v = frame["v"].to_numpy(dtype=float)

    # Absolute start time: parsing one timestamp is enough (day-long lunar files
    # have ~570k rows, parsing all of them would dominate the run time).
    start_time = None
    if abs_col is not None:
        for row in frame.index[:100]:
            stamp = pd.to_datetime(df.at[row, abs_col], errors="coerce")
            if pd.notna(stamp):
                start_time = stamp - pd.to_timedelta(frame.at[row, "t"] - t[0], unit="s")
                break

    dt = np.diff(t)
    median_dt = float(np.median(dt))
    fs = 1.0 / median_dt

    # 2. resampling onto a uniform grid if the sampling is irregular or a target is given
    irregular = np.any(np.abs(dt - median_dt) > 0.5 * median_dt)
    if target_fs is not None or irregular:
        fs = float(target_fs) if target_fs is not None else fs
        t_uniform = np.arange(t[0], t[-1] + 0.5 / fs, 1.0 / fs)
        v = np.interp(t_uniform, t, v)
        t = t_uniform

    units = vel_col[vel_col.find("(") + 1 : vel_col.find(")")] if "(" in vel_col else ""
    return Trace(
        file=os.path.basename(path),
        time_rel=t - t[0],
        velocity=v,
        fs=fs,
        start_time=start_time,
        units=units,
    )


def butterworth_filter(x: np.ndarray, fs: float, lowpass_hz: float | None,
                       highpass_hz: float | None, order: int = 4) -> np.ndarray:
    """Zero-phase Butterworth low-pass (or band-pass if ``highpass_hz`` is set)."""
    nyq = 0.5 * fs
    if lowpass_hz is not None and lowpass_hz >= 0.95 * nyq:
        lowpass_hz = None  # cut-off above Nyquist: low-pass is a no-op
    if highpass_hz is not None and highpass_hz <= 0:
        highpass_hz = None

    if lowpass_hz and highpass_hz:
        sos = signal.butter(order, [highpass_hz, lowpass_hz], btype="bandpass", fs=fs, output="sos")
    elif lowpass_hz:
        sos = signal.butter(order, lowpass_hz, btype="lowpass", fs=fs, output="sos")
    elif highpass_hz:
        sos = signal.butter(order, highpass_hz, btype="highpass", fs=fs, output="sos")
    else:
        return x.copy()

    padlen = 3 * (2 * len(sos) + 1)
    if len(x) > padlen:
        return signal.sosfiltfilt(sos, x)
    return signal.sosfilt(sos, x)


def preprocess(velocity: np.ndarray, fs: float, cfg: PlanetConfig, normalize: bool = True) -> np.ndarray:
    """Detrend, filter and (optionally) normalise a velocity series."""
    x = np.asarray(velocity, dtype=float)
    x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
    x = signal.detrend(x - x.mean(), type="linear")
    x = butterworth_filter(x, fs, cfg.lowpass_hz, cfg.highpass_hz, cfg.filter_order)
    if normalize:
        peak = np.max(np.abs(x))
        if peak > 0:
            x = x / peak
    return x
