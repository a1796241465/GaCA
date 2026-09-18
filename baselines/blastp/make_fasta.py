"""Write a protein metadata CSV as a FASTA file."""

import argparse
from pathlib import Path

import pandas as pd


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--id-column", default="Entry")
    parser.add_argument("--sequence-column", default="Sequence")
    args = parser.parse_args()
    table = pd.read_csv(args.csv)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="ascii") as handle:
        for _, row in table.iterrows():
            protein_id = str(row[args.id_column]).strip()
            sequence = "".join(str(row[args.sequence_column]).split()).upper()
            handle.write(f">{protein_id}\n{sequence}\n")


if __name__ == "__main__":
    main()