"""Step 2 - train and compare XGBoost, LightGBM, Random Forest, Decision Tree,
PCA + Logistic Regression, a deep neural net (MLP), SVM and KNN against the
classical STA/LTA detector.

Example:
    python train_models.py --features outputs/features_lunar.csv outputs/features_mars.csv

Outputs (in --out-dir, default "outputs"):
    results/model_comparison.csv      accuracy / precision / recall / F1 / ROC-AUC on the test set
    results/cross_validation.csv      5-fold CV mean and std on the training set
    results/summary.md                the same numbers as Markdown tables
    results/roc_curves.png            ROC curves of all models + STA/LTA baseline
    results/model_comparison.png      metric bar charts
    results/confusion_matrices.png
    results/feature_importance.png    (tree model)
    results/pca_explained_variance.png
    models/best_model.joblib          best model (by CV F1), refit on all data
"""

import argparse
import json
import os

import pandas as pd

from seismic_ml.models import MODEL_ORDER


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--features", nargs="+", required=True, help="one or more feature CSVs from build_features.py")
    parser.add_argument("--out-dir", default="outputs")
    parser.add_argument("--models", nargs="+", default=None, metavar="NAME",
                        help=f"subset of models to run (default: all). Choices: {', '.join(MODEL_ORDER)}")
    parser.add_argument("--test-size", type=float, default=0.3)
    parser.add_argument("--cv-folds", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    from seismic_ml.training import train_and_evaluate

    frames, configs = [], {}
    for path in args.features:
        df = pd.read_csv(path)
        frames.append(df)
        cfg_path = path + ".config.json"
        if os.path.exists(cfg_path) and "planet" in df.columns and len(df):
            with open(cfg_path) as fh:
                configs[str(df["planet"].iloc[0])] = json.load(fh)
        print(f"Loaded {len(df)} windows from {path}")
    data = pd.concat(frames, ignore_index=True)

    result = train_and_evaluate(data, args.out_dir, args.models, args.test_size, args.cv_folds,
                                args.seed, configs=configs)
    print("\n" + result["summary"])


if __name__ == "__main__":
    main()
