"""Model zoo: every classifier compared in the report (Table 5.1 and Figure 5.1).

* XGBoost, LightGBM          - gradient-boosted trees
* Random Forest, Decision Tree
* PCA + Logistic Regression  - PCA keeps 95 % of the variance, then a linear model
* Deep Neural Net            - multi-layer perceptron (scikit-learn MLPClassifier)
* SVM, KNN                   - extra baselines shown in the ROC figure

XGBoost and LightGBM are optional imports: if a library is not installed the
model is skipped with a warning instead of crashing the whole pipeline.
"""

from __future__ import annotations

import warnings

from sklearn.calibration import CalibratedClassifierCV
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

try:
    from xgboost import XGBClassifier
except ImportError:  # pragma: no cover - depends on the environment
    XGBClassifier = None

try:
    from lightgbm import LGBMClassifier
except ImportError:  # pragma: no cover - depends on the environment
    LGBMClassifier = None

# Display order (also the colour order in every plot)
MODEL_ORDER = [
    "XGBoost",
    "LightGBM",
    "Random Forest",
    "Deep Neural Net",
    "Decision Tree",
    "PCA + LogReg",
    "SVM",
    "KNN",
]


def get_models(random_state: int = 42, pos_weight: float = 1.0,
               include: list[str] | None = None) -> dict:
    """Return ``{name: unfitted estimator}`` for the requested models.

    ``pos_weight`` = n_negative / n_positive, used by XGBoost to handle class
    imbalance (the other models use ``class_weight='balanced'``).
    """
    models: dict = {}

    if XGBClassifier is not None:
        models["XGBoost"] = XGBClassifier(
            n_estimators=400, max_depth=4, learning_rate=0.05, subsample=0.9,
            colsample_bytree=0.9, min_child_weight=1, reg_lambda=1.0,
            scale_pos_weight=pos_weight, eval_metric="logloss",
            random_state=random_state, n_jobs=-1,
        )
    else:
        warnings.warn("xgboost is not installed - skipping the XGBoost model")

    if LGBMClassifier is not None:
        models["LightGBM"] = LGBMClassifier(
            n_estimators=400, learning_rate=0.05, num_leaves=15, min_child_samples=10,
            subsample=0.9, subsample_freq=1, colsample_bytree=0.9,
            class_weight="balanced", random_state=random_state, verbose=-1, n_jobs=-1,
        )
    else:
        warnings.warn("lightgbm is not installed - skipping the LightGBM model")

    models["Random Forest"] = RandomForestClassifier(
        n_estimators=400, min_samples_leaf=2, class_weight="balanced",
        random_state=random_state, n_jobs=-1,
    )
    models["Deep Neural Net"] = Pipeline([
        ("scale", StandardScaler()),
        ("mlp", MLPClassifier(hidden_layer_sizes=(64, 32, 16), activation="relu", alpha=1e-3,
                              learning_rate_init=1e-3, max_iter=2000, early_stopping=True,
                              validation_fraction=0.15, n_iter_no_change=30,
                              random_state=random_state)),
    ])
    models["Decision Tree"] = DecisionTreeClassifier(
        max_depth=6, min_samples_leaf=3, class_weight="balanced", random_state=random_state,
    )
    models["PCA + LogReg"] = Pipeline([
        ("scale", StandardScaler()),
        ("pca", PCA(n_components=0.95, random_state=random_state)),
        ("logreg", LogisticRegression(class_weight="balanced", max_iter=5000)),
    ])
    # Platt-scaled probabilities (replaces the deprecated SVC(probability=True))
    models["SVM"] = Pipeline([
        ("scale", StandardScaler()),
        ("svm", CalibratedClassifierCV(
            SVC(kernel="rbf", C=1.0, gamma="scale", class_weight="balanced", random_state=random_state),
            method="sigmoid", cv=3, ensemble=False)),
    ])
    models["KNN"] = Pipeline([
        ("scale", StandardScaler()),
        ("knn", KNeighborsClassifier(n_neighbors=7, weights="distance")),
    ])

    if include:
        wanted = {name.lower() for name in include}
        models = {k: v for k, v in models.items() if k.lower() in wanted}
        if not models:
            raise ValueError(f"None of {include} matches the available models {MODEL_ORDER}")
    return {name: models[name] for name in MODEL_ORDER if name in models}
