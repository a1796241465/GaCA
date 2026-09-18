"""Validate sequence-structure agreement before residue embedding extraction."""

import argparse
import json
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from Bio import pairwise2
from Bio.PDB import PDBParser
from tqdm import tqdm

THREE_TO_ONE = {
    "ALA": "A", "CYS": "C", "ASP": "D", "GLU": "E", "PHE": "F",
    "GLY": "G", "HIS": "H", "ILE": "I", "LYS": "K", "LEU": "L",
    "MET": "M", "ASN": "N", "PRO": "P", "GLN": "Q", "ARG": "R",
    "SER": "S", "THR": "T", "VAL": "V", "TRP": "W", "TYR": "Y",
    "UNK": "X",
}


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", type=Path, required=True)
    parser.add_argument("--structures", type=Path, required=True)
    parser.add_argument("--output-pkl", type=Path, required=True)
    parser.add_argument("--output-csv", type=Path, required=True)
    parser.add_argument("--stats", type=Path, required=True)
    parser.add_argument("--min-identity", type=float, default=0.95)
    parser.add_argument("--min-length", type=int, default=50)
    parser.add_argument("--max-length", type=int, default=1000)
    return parser.parse_args()


def structure_sequence(path):
    structure = PDBParser(QUIET=True).get_structure("protein", path)
    coordinates = []
    sequence = []
    for model in structure:
        for chain in model:
            for residue in chain:
                if residue.id[0] != " " or "CA" not in residue:
                    continue
                residue_name = residue.get_resname()
                if residue_name not in THREE_TO_ONE:
                    continue
                coordinate = residue["CA"].get_coord()
                if not np.isfinite(coordinate).all() or np.abs(coordinate).max() > 500:
                    continue
                coordinates.append(coordinate)
                sequence.append(THREE_TO_ONE[residue_name])
    if not coordinates:
        raise ValueError("no valid CA coordinates")
    return "".join(sequence), np.asarray(coordinates, dtype=np.float32)


def alignment_bounds(csv_sequence, pdb_sequence):
    alignments = pairwise2.align.globalxx(csv_sequence, pdb_sequence)
    if not alignments:
        raise ValueError("sequence alignment failed")
    aligned_csv, aligned_pdb = alignments[0].seqA, alignments[0].seqB
    csv_index = pdb_index = 0
    csv_start = csv_end = pdb_start = pdb_end = None
    matches = total = 0
    for csv_residue, pdb_residue in zip(aligned_csv, aligned_pdb):
        if csv_residue == "-" or pdb_residue == "-":
            csv_index += csv_residue != "-"
            pdb_index += pdb_residue != "-"
            continue
        if csv_start is None:
            csv_start, pdb_start = csv_index, pdb_index
        csv_end, pdb_end = csv_index + 1, pdb_index + 1
        matches += csv_residue == pdb_residue
        total += 1
        csv_index += 1
        pdb_index += 1
    if total == 0:
        raise ValueError("alignment has no paired residues")
    return csv_start, csv_end, pdb_start, pdb_end, matches / total


def main():
    args = parse_args()
    frame = pd.read_csv(args.csv)
    required = {"Entry", "Sequence", "EC number"}
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"Missing columns: {sorted(missing)}")

    accepted = []
    accepted_rows = []
    failures = {
        "missing_structure": [],
        "structure_error": [],
        "low_identity": [],
        "length_mismatch": [],
        "length_outside_range": [],
    }
    for _, row in tqdm(frame.iterrows(), total=len(frame), desc="validate"):
        protein_id = str(row["Entry"])
        pdb_path = args.structures / f"{protein_id}.pdb"
        if not pdb_path.exists():
            failures["missing_structure"].append(protein_id)
            continue
        try:
            pdb_sequence, coordinates = structure_sequence(pdb_path)
            bounds = alignment_bounds(str(row["Sequence"]), pdb_sequence)
        except Exception as error:
            failures["structure_error"].append(
                {"protein_id": protein_id, "error": str(error)}
            )
            continue
        csv_start, csv_end, pdb_start, pdb_end, identity = bounds
        if identity < args.min_identity:
            failures["low_identity"].append(
                {"protein_id": protein_id, "identity": identity}
            )
            continue
        aligned_sequence = str(row["Sequence"])[csv_start:csv_end]
        aligned_coordinates = coordinates[pdb_start:pdb_end]
        if len(aligned_sequence) != len(aligned_coordinates):
            failures["length_mismatch"].append(protein_id)
            continue
        if not args.min_length <= len(aligned_sequence) <= args.max_length:
            failures["length_outside_range"].append(protein_id)
            continue
        accepted.append(
            {
                "protein_id": protein_id,
                "sequence_raw": aligned_sequence,
                "ec_number": str(row["EC number"]),
                "alignment_identity": float(identity),
            }
        )
        output_row = row.to_dict()
        output_row.update(
            {
                "aligned_sequence": aligned_sequence,
                "aligned_length": len(aligned_sequence),
                "alignment_identity": identity,
            }
        )
        accepted_rows.append(output_row)

    args.output_pkl.parent.mkdir(parents=True, exist_ok=True)
    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    args.stats.parent.mkdir(parents=True, exist_ok=True)
    with args.output_pkl.open("wb") as handle:
        pickle.dump(accepted, handle, protocol=pickle.HIGHEST_PROTOCOL)
    pd.DataFrame(accepted_rows).to_csv(args.output_csv, index=False)
    report = {
        "source_rows": len(frame),
        "accepted_rows": len(accepted),
        "min_identity": args.min_identity,
        "min_length": args.min_length,
        "max_length": args.max_length,
        "failure_counts": {key: len(value) for key, value in failures.items()},
        "failures": failures,
    }
    args.stats.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({key: value for key, value in report.items() if key != "failures"}, indent=2))


if __name__ == "__main__":
    main()