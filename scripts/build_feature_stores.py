"""Build configurable feature stores with the audited store implementation."""

import argparse
from pathlib import Path

from embeddings.build_store import DEFAULT_SPECS, build_store


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--embedding-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument(
        "--stores",
        nargs="+",
        choices=sorted(DEFAULT_SPECS),
        default=list(DEFAULT_SPECS),
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    args.output_root.mkdir(parents=True, exist_ok=True)
    for name in args.stores:
        default_source, feature_dim = DEFAULT_SPECS[name]
        build_store(
            name,
            args.embedding_root / default_source.name,
            feature_dim,
            args.output_root,
            overwrite=args.overwrite,
        )


if __name__ == "__main__":
    main()
