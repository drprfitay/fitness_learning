#!/usr/bin/env bash
set -euo pipefail

DATA_DIR="data"

FILE1_URL="https://drive.google.com/file/d/1m9rkRSRbhzuOUwnp0Cicu3KgDvak-d4Q/view?usp=sharing"
FILE2_URL="https://drive.google.com/file/d/1pJkPKb7FTeATSgJVok28UOIFa1YLbb_L/view?usp=sharing"

FILE1_NAME="gcn4.tar.gz"
FILE2_NAME="pard3.tar.gz"

echo "=== Downloading benchmark files ==="

mkdir -p "${DATA_DIR}"

if ! command -v gdown >/dev/null 2>&1; then
    echo "Error: gdown is not installed."
    echo "Install it with:"
    echo "pip install gdown"
    exit 1
fi

download_and_extract () {
    URL="$1"
    OUTFILE="$2"

    echo ""
    echo "Downloading ${OUTFILE}..."
    gdown --fuzzy "${URL}" -O "${DATA_DIR}/${OUTFILE}"

    echo "Downloaded: ${DATA_DIR}/${OUTFILE}"

    if [[ "${OUTFILE}" == *.tar.gz ]]; then
        echo "Extracting ${OUTFILE}..."
        tar -xzvf "${DATA_DIR}/${OUTFILE}" -C "${DATA_DIR}"
    elif [[ "${OUTFILE}" == *.zip ]]; then
        echo "Extracting ${OUTFILE}..."
        unzip -o "${DATA_DIR}/${OUTFILE}" -d "${DATA_DIR}"
    else
        echo "No extraction rule for ${OUTFILE}; leaving file as-is."
    fi
}

download_and_extract "${FILE1_URL}" "${FILE1_NAME}"
download_and_extract "${FILE2_URL}" "${FILE2_NAME}"

echo ""
echo "Download complete."