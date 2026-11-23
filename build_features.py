"""Step 1 - preprocessing + STA/LTA + feature extraction -> labelled feature CSV.

Example (NASA Space Apps 2024 data package):
    python build_features.py --planet lunar \
        --data-dir space_apps_2024_seismic_detection/data/lunar/training/data/S12_GradeA \
        --catalog  space_apps_2024_seismic_detection/data/lunar/training/catalogs/apollo12_catalog_GradeA_final.csv \
        --out outputs/features_lunar.csv

    python build_features.py --planet mars \
        --data-dir space_apps_2024_seismic_detection/data/mars/training/data \
        --catalog  space_apps_2024_seismic_detection/data/mars/training/catalogs/Mars_InSight_training_catalog_final.csv \
        --out outputs/features_mars.csv
"""

import argparse
import json
import os
import time

from seismic_ml.config import get_config


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--planet", required=True, choices=["lunar", "mars"])
    parser.add_argument("--data-dir", required=True, help="folder with the seismic CSV files (searched recursively)")
    parser.add_argument("--catalog", required=True, help="event catalogue CSV (filename + arrival time)")
    parser.add_argument("--out", default=None, help="output CSV (default: outputs/features_<planet>.csv)")
    parser.add_argument("--positives-per-event", type=int, default=3,
                        help="windows per catalogued event (1 exact + jittered copies)")
    parser.add_argument("--negatives-per-trace", type=int, default=10, help="random noise windows per file")
    parser.add_argument("--max-files", type=int, default=None, help="only process the first N files")
    parser.add_argument("--seed", type=int, default=42)
    # optional STA/LTA overrides
    parser.add_argument("--sta-sec", type=float)
    parser.add_argument("--lta-sec", type=float)
    parser.add_argument("--k", type=float, help="threshold = mean + k*std of the STA/LTA ratio")
    parser.add_argument("--window-sec", type=float, help="feature window length")
    args = parser.parse_args()

    from seismic_ml.dataset import build_dataset

    cfg = get_config(args.planet, sta_sec=args.sta_sec, lta_sec=args.lta_sec, k=args.k,
                     window_sec=args.window_sec)
    out = args.out or os.path.join("outputs", f"features_{cfg.name}.csv")
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)

    print(f"Building {cfg.name} feature table from {args.data_dir}")
    t0 = time.perf_counter()
    df = build_dataset(args.data_dir, args.catalog, cfg, args.positives_per_event,
                       args.negatives_per_trace, args.max_files, args.seed)
    df.to_csv(out, index=False)
    with open(out + ".config.json", "w") as fh:  # remembered by train_models.py / detect_events.py
        json.dump(cfg.to_dict(), fh, indent=2)

    print(f"\nSaved {len(df)} windows ({int(df['label'].sum())} events / "
          f"{int((df['label'] == 0).sum())} noise) -> {out}  [{time.perf_counter() - t0:.1f} s]")
    trig = df[df["source"] == "sta_lta_trigger"]
    if len(trig):
        print(f"STA/LTA triggers on catalogued files: {len(trig)} "
              f"({int(trig['label'].sum())} true detections, {int((trig['label'] == 0).sum())} false alarms)")


if __name__ == "__main__":
    main()
