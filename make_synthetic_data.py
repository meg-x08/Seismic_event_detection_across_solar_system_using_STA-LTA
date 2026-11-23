"""Generate small synthetic lunar / Martian datasets in the NASA Space Apps CSV format.

Use this to try the ML pipeline without downloading the real ~1 GB data package.
Numbers obtained on synthetic data are only a functional check of the code.

Example:
    python make_synthetic_data.py --planet lunar --n-files 20
    python make_synthetic_data.py --planet mars  --n-files 20
"""

import argparse

from seismic_ml.synthetic import generate_synthetic_dataset


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--planet", choices=["lunar", "mars", "both"], default="both")
    parser.add_argument("--out-dir", default="data/synthetic")
    parser.add_argument("--n-files", type=int, default=20)
    parser.add_argument("--duration-s", type=float, default=None,
                        help="length of each trace (default: 4 h lunar, 1 h Mars)")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    planets = ["lunar", "mars"] if args.planet == "both" else [args.planet]
    for i, planet in enumerate(planets):
        data_dir, catalog = generate_synthetic_dataset(args.out_dir, planet, args.n_files,
                                                       args.duration_s, args.seed + i)
        print(f"[{planet}] {args.n_files} files -> {data_dir}")
        print(f"[{planet}] catalogue      -> {catalog}")


if __name__ == "__main__":
    main()
