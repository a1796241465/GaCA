#!/usr/bin/env bash
set -euo pipefail

if [[ "${CONDA_DEFAULT_ENV:-}" != "gaca" ]]; then
  echo "Activate the gaca Conda environment before running this script." >&2
  echo "  conda activate gaca" >&2
  exit 1
fi

python -m pip install torch-scatter==2.1.2 --only-binary=:all: \
  -f https://data.pyg.org/whl/torch-2.2.0+cu121.html

python -c "import torch, torch_geometric, torch_scatter, esm.inverse_folding; assert torch.cuda.is_available(); print('PyTorch', torch.__version__, 'CUDA', torch.version.cuda)"
echo "GaCA dependencies are available."
