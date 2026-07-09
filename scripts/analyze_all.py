"""Run analyze_fields on every CAMELS field file present under --data-root.

Picks the transform per field (velocity fields -> asinh, else log10) and prints, for each:
  - raw-min  (if negative under log10, the script warns you to switch that field to asinh)
  - std percentiles + rejection impact  -> read min_std off the lower tail (~p10)

    python scripts/analyze_all.py --data-root /workspace/data

Use the printed p10 (and transform) to fill field_configs in train_fsdp.build_dataloader.
The raw-min line is authoritative for transform choice -- trust it over the name guess below.
"""
import argparse
import glob
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src", "data"))
import fields  # noqa: E402

# Known signed (velocity) fields need asinh; log10 would NaN on negatives. Everything else
# defaults to log10 -- but the raw-min warning in analyze_fields is the real authority.
SIGNED = {"Vgas", "Vcdm"}


def field_name(path):
    # Maps_<field>_<suite>_<set>_z=0.00.npy  ->  <field>
    return os.path.basename(path).split("_")[1]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", default="/workspace/data")
    ap.add_argument("--n", type=int, default=2000, help="maps sampled per field")
    args = ap.parse_args()

    files = sorted(glob.glob(os.path.join(args.data_root, "Maps_*_z=0.00.npy")))
    if not files:
        print(f"no Maps_*_z=0.00.npy files under {args.data_root}")
        return

    print(f"found {len(files)} field file(s) under {args.data_root}")
    for path in files:
        name = field_name(path)
        transform = "asinh" if name in SIGNED else "log10"
        print(f"\n{'#' * 70}\n# {name}   transform={transform}   {os.path.basename(path)}\n{'#' * 70}")
        fields.analyze_fields(path, n=args.n, transform=transform)


if __name__ == "__main__":
    main()
