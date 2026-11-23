"""Hybrid STA/LTA + machine-learning seismic event detection for planetary data.

Pipeline (see README.md, section "Machine Learning Pipeline"):

    raw CSV  ->  preprocessing  ->  STA/LTA triggers  ->  feature extraction
             ->  ML classifier (XGBoost / LightGBM / RF / DT / PCA+LogReg / MLP / SVM / KNN)
             ->  event label (major / moderate / minor / questionable / noise)
"""

from .config import PLANET_CONFIGS, PlanetConfig, get_config

__all__ = ["PLANET_CONFIGS", "PlanetConfig", "get_config"]
__version__ = "1.0.0"
