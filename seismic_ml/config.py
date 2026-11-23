"""Planet-specific parameters for preprocessing, STA/LTA detection and windowing.

Lunar (Apollo PSE) and Martian (InSight SEIS) recordings differ a lot in sampling
rate, frequency content and event duration, so every tunable number lives here
instead of being hard-coded in the pipeline.  All window lengths are given in
seconds and converted to samples using the sampling rate measured from the data.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, replace


@dataclass(frozen=True)
class PlanetConfig:
    name: str

    # --- Preprocessing -------------------------------------------------------
    # Butterworth filter corner frequencies (Hz).  ``highpass_hz=None`` gives a
    # pure low-pass filter; setting both gives a band-pass filter.
    lowpass_hz: float | None
    highpass_hz: float | None
    filter_order: int = 4

    # --- STA/LTA -------------------------------------------------------------
    sta_sec: float = 120.0
    lta_sec: float = 600.0
    # Dynamic threshold  thr_on = mean(R) + k * std(R)
    k: float = 3.0
    # Trigger ends when the ratio falls below  mean(R) + k_off * std(R)
    k_off: float = 1.0
    # Triggers whose gap is shorter than this are merged into one event
    min_gap_sec: float = 300.0

    # --- Feature window ------------------------------------------------------
    # A feature window starts ``pre_event_sec`` before the anchor (trigger onset
    # or catalogue arrival) and lasts ``window_sec`` in total.
    pre_event_sec: float = 60.0
    window_sec: float = 1200.0

    # --- Labelling (training-set construction) -------------------------------
    # A trigger within +/- tolerance of a catalogue arrival is a true event.
    match_tolerance_sec: float = 600.0
    # Region around a catalogue arrival where "noise" windows must NOT be drawn
    # (the event coda can last a long time, especially on the Moon).
    exclusion_before_sec: float = 1200.0
    exclusion_after_sec: float = 5400.0
    # Jitter (+/- seconds) applied to positive windows built from the catalogue
    jitter_sec: float = 120.0

    def to_dict(self) -> dict:
        return asdict(self)


LUNAR = PlanetConfig(
    name="lunar",
    # Apollo 12 data are sampled at ~6.6 Hz; moonquake energy sits mostly in
    # 0.5-1.0 Hz (same band as the NASA Space Apps 2024 starter notebook).
    lowpass_hz=1.0,
    highpass_hz=0.5,
    sta_sec=120.0,
    lta_sec=600.0,
    k=3.0,
    k_off=1.0,
    min_gap_sec=600.0,
    pre_event_sec=120.0,
    window_sec=1800.0,
    match_tolerance_sec=900.0,
    exclusion_before_sec=1800.0,
    exclusion_after_sec=7200.0,
    jitter_sec=180.0,
)

MARS = PlanetConfig(
    name="mars",
    # InSight SEIS data are sampled at 20 Hz; marsquakes are shorter and richer
    # in higher frequencies than moonquakes, and wind noise is strong below 0.5 Hz.
    lowpass_hz=3.0,
    highpass_hz=0.5,
    sta_sec=30.0,
    lta_sec=500.0,
    k=2.8,
    k_off=1.0,
    min_gap_sec=120.0,
    pre_event_sec=20.0,
    window_sec=300.0,
    match_tolerance_sec=180.0,
    exclusion_before_sec=300.0,
    exclusion_after_sec=900.0,
    jitter_sec=30.0,
)

PLANET_CONFIGS: dict[str, PlanetConfig] = {"lunar": LUNAR, "mars": MARS}


def get_config(planet: str, **overrides) -> PlanetConfig:
    """Return the config for ``planet`` ('lunar' or 'mars'), optionally overridden."""
    key = planet.lower().strip()
    if key in {"moon", "apollo"}:
        key = "lunar"
    if key in {"insight", "martian"}:
        key = "mars"
    if key not in PLANET_CONFIGS:
        raise ValueError(f"Unknown planet '{planet}'. Choose from {sorted(PLANET_CONFIGS)}.")
    cfg = PLANET_CONFIGS[key]
    overrides = {k: v for k, v in overrides.items() if v is not None}
    return replace(cfg, **overrides) if overrides else cfg
