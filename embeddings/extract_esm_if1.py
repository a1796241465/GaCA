"""Extract residue-level ESM-IF1 encoder embeddings used by GaCA."""

import argparse
import pickle
from pathlib import Path

import biotite.structure as struc
import esm
import numpy as np
import torch
from biotite.structure.io import pdb
from tqdm import tqdm


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--structures", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def load_backbone(path):
    structure_file = pdb.PDBFile.read(path)
    if structure_file.get_model_count() == 0:
        raise ValueError("PDB file has no model")
    atoms = pdb.get_structure(structure_file, model=1)
    atoms = atoms[struc.filter_amino_acids(atoms)]
    starts = struc.get_residue_starts(atoms, add_exclusive_stop=True)
    coordinates = []
    for start, stop in zip(starts[:-1], starts[1:]):
        residue = atoms[start:stop]
        backbone = []
        for atom_name in ("N", "CA", "C"):
            matches = residue[residue.atom_name == atom_name]
            if len(matches) == 0:
                backbone = []
                break
            backbone.append(matches.coord[0])
        if backbone:
            matrix = np.asarray(backbone, dtype=np.float32)
            if np.isfinite(matrix).all():
                coordinates.append(matrix)
    if not coordinates:
        raise ValueError("PDB file has no complete backbone residue")
    return np.stack(coordinates, axis=0)

def encode(model, coordinates, device):
    coordinate_tensor = torch.tensor(
        coordinates, dtype=torch.float32, device=device
    ).unsqueeze(0)
    padding_mask = torch.zeros(
        (1, coordinates.shape[0]), dtype=torch.bool, device=device
    )
    confidence = torch.ones(
        (1, coordinates.shape[0]), dtype=torch.float32, device=device
    )
    with torch.no_grad():
        output = model.encoder.forward(
            coordinate_tensor,
            encoder_padding_mask=padding_mask,
            confidence=confidence,
            return_all_hiddens=False,
        )
        matrix = output["encoder_out"][0]
        if matrix.ndim == 3 and matrix.shape[1] == 1:
            matrix = matrix[:, 0, :]
        matrix = matrix / (matrix.norm(dim=-1, keepdim=True) + 1e-6)
    return matrix.cpu().numpy().reshape(-1, 512).astype(np.float16)


def main():
    args = parse_args()
    device = torch.device(args.device)
    if args.checkpoint:
        model, _ = esm.pretrained.load_model_and_alphabet_local(str(args.checkpoint))
    else:
        model, _ = esm.pretrained.esm_if1_gvp4_t16_142M_UR50()
    model = model.to(device).eval()

    embeddings = {}
    failures = []
    for path in tqdm(sorted(args.structures.glob("*.pdb")), desc="ESM-IF1"):
        try:
            embeddings[path.stem] = encode(model, load_backbone(path), device)
        except Exception as error:
            failures.append((path.stem, str(error)))

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("wb") as handle:
        pickle.dump(embeddings, handle, protocol=pickle.HIGHEST_PROTOCOL)
    failure_path = args.output.with_suffix(".failures.tsv")
    failure_path.write_text(
        "protein_id\terror\n" + "".join(f"{pid}\t{error}\n" for pid, error in failures),
        encoding="utf-8",
    )
    print(
        f"saved {len(embeddings)} proteins to {args.output}; "
        f"failures={len(failures)}"
    )


if __name__ == "__main__":
    main()