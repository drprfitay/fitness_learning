#!/usr/bin/env bash
set -euo pipefail

DOWNLOAD_DIR=".downloads"
DATA_DIR="data"
NOTEBOOKS_DIR="notebooks"

FILE1_URL="https://drive.google.com/file/d/1m9rkRSRbhzuOUwnp0Cicu3KgDvak-d4Q/view?usp=sharing"
FILE2_URL="https://drive.google.com/file/d/1pJkPKb7FTeATSgJVok28UOIFa1YLbb_L/view?usp=sharing"
FILE3_URL="https://drive.google.com/file/d/1ZDh7WRpV4gYSCdTcu5eF_521pXpzu8Mz/view?usp=sharing"

FILE1_NAME="gcn4.tar.gz"
FILE2_NAME="pard3.tar.gz"
FILE3_NAME="notebook_results.tar.gz"

echo "=== Downloading benchmark files ==="

mkdir -p "${DOWNLOAD_DIR}"
mkdir -p "${DATA_DIR}"
mkdir -p "${NOTEBOOKS_DIR}"

if ! command -v gdown >/dev/null 2>&1; then
    echo "Error: gdown is not installed."
    echo "Install it with:"
    echo "pip install gdown"
    exit 1
fi

download_file () {
    URL="$1"
    OUTFILE="$2"

    echo ""
    echo "Downloading ${OUTFILE}..."
    gdown --fuzzy "${URL}" -O "${DOWNLOAD_DIR}/${OUTFILE}"
    echo "Downloaded: ${DOWNLOAD_DIR}/${OUTFILE}"
}

extract_to_dir () {
    ARCHIVE="$1"
    TARGET_DIR="$2"

    echo ""
    echo "Extracting ${ARCHIVE} into ${TARGET_DIR}..."

    if [[ "${ARCHIVE}" == *.tar.gz ]]; then
        tar -xzvf "${ARCHIVE}" -C "${TARGET_DIR}"
    elif [[ "${ARCHIVE}" == *.zip ]]; then
        unzip -o "${ARCHIVE}" -d "${TARGET_DIR}"
    else
        echo "No extraction rule for ${ARCHIVE}; leaving file as-is."
    fi
}

download_file "${FILE1_URL}" "${FILE1_NAME}"
download_file "${FILE2_URL}" "${FILE2_NAME}"
download_file "${FILE3_URL}" "${FILE3_NAME}"

# Extract dataset archives into data/
extract_to_dir "${DOWNLOAD_DIR}/${FILE1_NAME}" "${DATA_DIR}"
extract_to_dir "${DOWNLOAD_DIR}/${FILE2_NAME}" "${DATA_DIR}"

# Extract notebook result files into repository root.
# If the archive contains notebooks/*.pkl, they will land correctly.
# If it contains bare *.pkl files, move them into notebooks/ below.
extract_to_dir "${DOWNLOAD_DIR}/${FILE3_NAME}" "."

# Move bare notebook result files into notebooks/ if needed
for f in all_results.pkl best_extrapolation_results.pkl ProtGymMulti_muts.pkl; do
    if [[ -f "${f}" ]]; then
        echo "Moving ${f} to ${NOTEBOOKS_DIR}/"
        mv "${f}" "${NOTEBOOKS_DIR}/"
    fi
done

# Move data/ into notebooks/data/
if [[ -d "${DATA_DIR}" ]]; then
    echo ""
    echo "Moving ${DATA_DIR}/ to ${NOTEBOOKS_DIR}/${DATA_DIR}/"

    rm -rf "${NOTEBOOKS_DIR}/${DATA_DIR}"
    mv "${DATA_DIR}" "${NOTEBOOKS_DIR}/${DATA_DIR}"
fi

echo ""
echo "Cleaning temporary downloads..."
rm -rf "${DOWNLOAD_DIR}"

echo ""
echo "Download and extraction complete."
echo "Data are now in: ${NOTEBOOKS_DIR}/${DATA_DIR}/"
echo "Notebook result files should be in: ${NOTEBOOKS_DIR}/"