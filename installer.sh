#!/usr/bin/env bash
set -euo pipefail

ENV_NAME="plm_benchmark"

echo "=== PLM benchmark environment installer ==="

if ! command -v conda >/dev/null 2>&1; then
    echo "Error: conda was not found. Please install Miniconda or Anaconda first."
    exit 1
fi

# Enable conda activate inside this script
source "$(conda info --base)/etc/profile.d/conda.sh"

echo "Creating/updating conda environment: ${ENV_NAME}"

if conda env list | awk '{print $1}' | grep -qx "${ENV_NAME}"; then
    echo "Environment ${ENV_NAME} already exists. Updating..."
    conda env update -n "${ENV_NAME}" -f environment.yml --prune
else
    echo "Creating environment ${ENV_NAME}..."
    conda env create -n "${ENV_NAME}" -f environment.yml
fi

echo ""
echo "Installation complete."
echo "Activate the environment with:"
echo "conda activate ${ENV_NAME}"