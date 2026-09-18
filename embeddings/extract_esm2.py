"""Extract residue-level ESM-2 embeddings used by GaCA."""

import argparse
import pickle
from pathlib import Path

import esm
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm


class ProteinSequences(Dataset):
    def __init__(self, records):
        self.records = records

    def __len__(self):
        return len(self.records)

    def __getitem__(self, index):
        return self.records[index]


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--id-column", default="Entry")
    parser.add_argument("--sequence-column", default="Sequence")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--max-length", type=int, default=1024)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def read_records(path, id_column, sequence_column):
    if path.suffix.lower() in {".pkl", ".pickle"}:
        with path.open("rb") as handle:
            data = pickle.load(handle)
        if isinstance(data, dict):
            return [(str(key), str(value)) for key, value in data.items()]
        records = []
        for index, item in enumerate(data):
            protein_id = item.get("protein_id", item.get(id_column, f"record_{index}"))
            sequence = item.get("sequence_raw", item.get(sequence_column))
            if sequence is None:
                raise ValueError(f"No sequence in record {index}")
            records.append((str(protein_id), str(sequence)))
        return records
    table = pd.read_csv(path)
    missing = {id_column, sequence_column}.difference(table.columns)
    if missing:
        raise ValueError(f"Missing columns: {sorted(missing)}")
    return list(
        zip(table[id_column].astype(str), table[sequence_column].astype(str))
    )


def main():
    args = parse_args()
    records = read_records(args.input, args.id_column, args.sequence_column)
    if len({protein_id for protein_id, _ in records}) != len(records):
        raise ValueError("Protein identifiers must be unique before embedding extraction")

    device = torch.device(args.device)
    model, alphabet = esm.pretrained.esm2_t33_650M_UR50D()
    model = model.to(device).eval()
    batch_converter = alphabet.get_batch_converter()
    loader = DataLoader(
        ProteinSequences(records),
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=lambda batch: batch,
    )

    embeddings = {}
    with torch.no_grad():
        for batch in tqdm(loader, desc="ESM-2"):
            identifiers, sequences = zip(*batch)
            truncated = [sequence[: args.max_length] for sequence in sequences]
            _, converted_sequences, tokens = batch_converter(
                list(zip(identifiers, truncated))
            )
            representations = model(
                tokens.to(device), repr_layers=[33], return_contacts=False
            )["representations"][33]
            for index, protein_id in enumerate(identifiers):
                length = len(converted_sequences[index])
                matrix = representations[index, 1 : length + 1]
                embeddings[protein_id] = matrix.cpu().numpy().astype(np.float16)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("wb") as handle:
        pickle.dump(embeddings, handle, protocol=pickle.HIGHEST_PROTOCOL)
    print(f"saved {len(embeddings)} proteins to {args.output}")


if __name__ == "__main__":
    main()