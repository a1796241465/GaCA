"""Download the upstream CARE repository that provides the benchmark splits."""

import argparse
import json
import subprocess
from pathlib import Path

CARE_PAPER = (
    "https://proceedings.neurips.cc/paper_files/paper/2024/hash/"
    "05a7ad45d75a3082d7a3a70de8743140-Abstract-Datasets_and_Benchmarks_Track.html"
)
CARE_REPOSITORY = "https://github.com/jsunn-y/CARE.git"
EXPECTED_SPLITS = (
    "protein_train50.csv",
    "30_protein_test.csv",
    "30-50_protein_test.csv",
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Clone CARE and locate the task-1 protein split files."
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--repository", default=CARE_REPOSITORY)
    parser.add_argument("--revision", help="Optional CARE commit or tag to check out.")
    return parser.parse_args()


def run(command, cwd=None):
    subprocess.run(command, cwd=cwd, check=True)


def main():
    args = parse_args()
    output = args.output_dir.resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(
            f"{output} is not empty. Use a new directory or update it with git directly."
        )
    output.parent.mkdir(parents=True, exist_ok=True)
    run(["git", "clone", "--depth", "1", args.repository, str(output)])
    if args.revision:
        run(["git", "fetch", "--depth", "1", "origin", args.revision], cwd=output)
        run(["git", "checkout", "FETCH_HEAD"], cwd=output)

    located = {}
    for filename in EXPECTED_SPLITS:
        matches = sorted(output.rglob(filename))
        located[filename] = [str(path.relative_to(output)) for path in matches]
    revision = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=output, text=True
    ).strip()
    record = {
        "care_paper": CARE_PAPER,
        "repository": args.repository,
        "revision": revision,
        "located_split_files": located,
    }
    (output / "gaca_source_record.json").write_text(
        json.dumps(record, indent=2), encoding="utf-8"
    )
    print(json.dumps(record, indent=2))
    if any(not paths for paths in located.values()):
        print(
            "One or more split files were not present in the cloned revision. "
            "Follow the upstream CARE data instructions and place the files under "
            "data/raw/care/splits/task1/."
        )


if __name__ == "__main__":
    main()