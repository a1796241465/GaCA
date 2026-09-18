"""Create an ID-restricted structure directory for Foldseek."""

import argparse
import os
import shutil
from pathlib import Path

import pandas as pd


def materialize(source, target, mode):
    if mode == "copy":
        shutil.copy2(source, target)
    elif mode == "symlink":
        target.symlink_to(source.resolve())
    else:
        try:
            os.link(source, target)
        except OSError:
            shutil.copy2(source, target)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", type=Path, required=True)
    parser.add_argument("--source-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--mode", choices=("hardlink", "symlink", "copy"), default="hardlink")
    args = parser.parse_args()

    identifiers = set(pd.read_csv(args.csv, dtype={"Entry": str})["Entry"].astype(str))
    args.output_dir.mkdir(parents=True, exist_ok=True)
    stale = {path.stem for path in args.output_dir.glob("*.pdb")} - identifiers
    if stale:
        raise RuntimeError(
            f"Output directory contains {len(stale)} stale PDB files; use an empty directory"
        )
    missing = []
    for protein_id in sorted(identifiers):
        source = args.source_dir / f"{protein_id}.pdb"
        target = args.output_dir / source.name
        if not source.exists():
            missing.append(protein_id)
        elif not target.exists():
            materialize(source, target, args.mode)
    if missing:
        raise FileNotFoundError(f"Missing {len(missing)} structures; first IDs: {missing[:10]}")
    print(f"Materialized {len(identifiers)} structures in {args.output_dir}")


if __name__ == "__main__":
    main()