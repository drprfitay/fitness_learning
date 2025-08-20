#!/bin/bash

# Usage: ./train_triplet_backbone_wexac.sh "1,2,3,4" [--esm8|--esm35|--esm650]
# $1: comma-separated train mutation indices (e.g. "1,2,3,4")
# $2: optional, --esm8 (default), --esm35, or --esm650

if [ "$#" -lt 1 ] || [ "$#" -gt 2 ]; then
    echo "Usage: $0 <train_indices_comma_separated> [--esm8|--esm35|--esm650]"
    exit 1
fi

TRAIN_INDICES_CSV="$1"

# Default values
ESM_FLAG="--esm8"
ESM_DIR="esm8m"
PLM_NAME="esm2_t6_8M_UR50D"

# Parse optional argument for ESM model
if [ "$#" -eq 2 ]; then
    if [ "$2" == "--esm35" ]; then
        ESM_FLAG="--esm35"
        ESM_DIR="esm35m"
        PLM_NAME="esm2_t12_35M_UR50D"
    elif [ "$2" == "--esm650" ]; then
        ESM_FLAG="--esm650"
        ESM_DIR="esm650m"
        PLM_NAME="esm2_t33_650M_UR50D"
    elif [ "$2" == "--esm8" ]; then
        ESM_FLAG="--esm8"
        ESM_DIR="esm8m"
        PLM_NAME="esm2_t6_8M_UR50D"
    else
        # Default to esm8m if not recognized
        ESM_FLAG="--esm8"
        ESM_DIR="esm8m"
        PLM_NAME="esm2_t6_8M_UR50D"
    fi
fi

# Convert comma-separated to space-separated for passing as multiple arguments
TRAIN_INDICES_ARGS=(${TRAIN_INDICES_CSV//,/ })

# For path: join train indices with 'x' instead of nothing
TRAIN_MUTS_X=$(echo "$TRAIN_INDICES_CSV" | tr ',' 'x')

SAVE_PATH="pretraining/${ESM_DIR}/one_shot/triplet_backbone/train_${TRAIN_MUTS_X}"

SCRIPT_NAME="tmp/triplet_run_epinnet_${TRAIN_MUTS_X}_${ESM_DIR}_$$.sh"

# Set config file based on ESM_DIR
CONFIG_FILE="../configs/${ESM_DIR}_config.py"

cat <<EOF > "$SCRIPT_NAME"
#!/bin/bash
source ~/.bashrc
conda activate esm_env
python -u ../code/train_epinnet.py \\
    --train True \\
    --train_type triplet \\
    --evaluate_train False \\
    --evaluate_test False \\
    --save_path "$SAVE_PATH" \\
    --config $CONFIG_FILE \\
    --train_indices ${TRAIN_INDICES_ARGS[@]} \\
    --plm_name "$PLM_NAME"
EOF

chmod +x "$SCRIPT_NAME"

ERR_FILE="./triplet_err_file_${TRAIN_MUTS_X}_${ESM_DIR}_$$"
OUT_FILE="./triplet_out_file_${TRAIN_MUTS_X}_${ESM_DIR}_$$"
bsub -n 6 -gpu num=1:gmem=12G:aff=yes -R same[gpumodel] -R rusage[mem=64GB] -R span[ptile=6] -o "$ERR_FILE" -e "$OUT_FILE" -q short-gpu "./$SCRIPT_NAME" 

# Wait a moment to ensure job is submitted before deleting
#sleep 2
#rm "$SCRIPT_NAME"
