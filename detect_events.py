"""Step 3 - hybrid detection on new recordings: STA/LTA triggers -> ML confirmation
-> major / moderate / minor / questionable labels, CSV report and 3-panel plots.

Example:
    python detect_events.py --planet lunar \
        --input space_apps_2024_seismic_detection/data/lunar/test/data/S12_GradeB \
        --model outputs/models/best_model.joblib --out-dir outputs/detections_lunar

Outputs:
    detected_events.csv           events confirmed by the ML model
    all_triggers_classified.csv   every STA/LTA trigger with its ML probability
    plots/<file>_hybrid_plot.png  waveform / STA/LTA / classified events
"""

import argparse


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--planet", required=True, choices=["lunar", "mars"])
    parser.add_argument("--input", required=True, help="a CSV file or a folder of CSV files")
    parser.add_argument("--model", default="outputs/models/best_model.joblib")
    parser.add_argument("--out-dir", default=None, help="default: outputs/detections_<planet>")
    parser.add_argument("--no-plots", action="store_true")
    parser.add_argument("--reject-threshold", type=float, default=0.5,
                        help="ML probability below this -> trigger discarded as noise")
    parser.add_argument("--confident-threshold", type=float, default=0.7,
                        help="ML probability below this (but above reject) -> 'questionable'")
    parser.add_argument("--major-snr-db", type=float, default=20.0,
                        help="event-window SNR (dB) at or above this -> 'major'")
    parser.add_argument("--moderate-snr-db", type=float, default=10.0,
                        help="event-window SNR (dB) at or above this -> 'moderate' (else 'minor')")
    args = parser.parse_args()

    from seismic_ml.detect import detect_directory

    out_dir = args.out_dir or f"outputs/detections_{args.planet}"
    result = detect_directory(args.input, args.model, args.planet, out_dir, make_plots=not args.no_plots,
                              reject_threshold=args.reject_threshold,
                              confident_threshold=args.confident_threshold,
                              major_snr_db=args.major_snr_db, moderate_snr_db=args.moderate_snr_db)
    kept = result[result["classification"] != "noise"] if len(result) else result
    print(f"\n{len(result)} STA/LTA triggers, {len(kept)} confirmed events -> {out_dir}/detected_events.csv")
    if len(kept):
        print(kept["classification"].value_counts().to_string())


if __name__ == "__main__":
    main()
