#!/bin/bash

# Usage: ./submit_epinnet.sh "1,2,3,4" "5,6,7,8" [--backbone] [--esm8|--esm35|--esm650]
# $1: comma-separated train mutation indices (e.g. "1,2,3,4")
# $2: comma-separated test mutation indices (e.g. "5,6,7,8")
# $3: optional, if set to --backbone, loads triplet backbone weights
# $4: optional, --esm8 (default), --esm35, or --esm650

if [ "$#" -lt 2 ] || [ "$#" -gt 4 ]; then
    echo "Usage: $0 <train_indices_comma_separated> <test_indices_comma_separated> [--backbone] [--esm8|--esm35|--esm650]"
    exit 1
fi

TRAIN_INDICES_CSV="$1"
TEST_INDICES_CSV="$2"
USE_BACKBONE=false

# Default values
ESM_FLAG="--esm8"
ESM_DIR="esm8m"
PLM_NAME="esm2_t6_8M_UR50D"

# Parse optional arguments
if [ "$#" -eq 3 ]; then
    if [ "$3" == "--backbone" ]; then
        USE_BACKBONE=true
    elif [ "$3" == "--esm35" ]; then
        ESM_FLAG="--esm35"
        ESM_DIR="esm35m"
        PLM_NAME="esm2_t12_35M_UR50D"
    elif [ "$3" == "--esm650" ]; then
        ESM_FLAG="--esm650"
        ESM_DIR="esm650m"
        PLM_NAME="esm2_t33_650M_UR50D"
    fi
elif [ "$#" -eq 4 ]; then
    if [ "$3" == "--backbone" ]; then
        USE_BACKBONE=true
    fi
    if [ "$4" == "--esm35" ]; then
        ESM_FLAG="--esm35"
        ESM_DIR="esm35m"
        PLM_NAME="esm2_t12_35M_UR50D"
    elif [ "$4" == "--esm650" ]; then
        ESM_FLAG="--esm650"
        ESM_DIR="esm650m"
        PLM_NAME="esm2_t33_650M_UR50D"
    elif [ "$4" == "--esm8" ]; then
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
TEST_INDICES_ARGS=(${TEST_INDICES_CSV//,/ })

# For path naming: join with 'x' instead of nothing
TRAIN_MUTS_PATH="${TRAIN_INDICES_CSV//,/'x'}"
TEST_MUTS_PATH="${TEST_INDICES_CSV//,/'x'}"

# Set config file based on ESM_DIR
CONFIG_FILE="../configs/${ESM_DIR}_config.yaml"

if [ "$USE_BACKBONE" = true ]; then
    # Save path for backbone mode
    SAVE_PATH="pretraining/${ESM_DIR}/one_shot/triplet_backbone/train_${TRAIN_MUTS_PATH}/train_${TRAIN_MUTS_PATH}_test_${TEST_MUTS_PATH}"
    WEIGHTS_PATH="pretraining/${ESM_DIR}/one_shot/triplet_backbone/train_${TRAIN_MUTS_PATH}/final_model.pt"
    EXTRA_ARGS="--load_weights True --weights_path \"$WEIGHTS_PATH\""
    SCRIPT_NAME="tmp/run_epinnet_${TRAIN_MUTS_PATH}_${TEST_MUTS_PATH}_backbone_${ESM_DIR}_$$.sh"
else
    # Default zero-shot save path
    SAVE_PATH="pretraining/${ESM_DIR}/zero_shot/train_${TRAIN_MUTS_PATH}_test_${TEST_MUTS_PATH}"
    EXTRA_ARGS=""
    SCRIPT_NAME="tmp/run_epinnet_${TRAIN_MUTS_PATH}_${TEST_MUTS_PATH}_${ESM_DIR}_$$.sh"
fi

cat <<EOF > "$SCRIPT_NAME"
#!/bin/bash
source ~/.bashrc
conda activate esm_env
python -u ../code/train_epinnet.py \\
    --train False \\
    --evaluate_test True \\
    --save_path "$SAVE_PATH" \\
    --config $CONFIG_FILE \\
    --train_indices ${TRAIN_INDICES_ARGS[@]} \\
    --test_indices ${TEST_INDICES_ARGS[@]} \\
    --plm_name "$PLM_NAME" \\
    $EXTRA_ARGS
EOF

chmod +x "$SCRIPT_NAME"

if [ "$USE_BACKBONE" = true ]; then
    ERR_FILE="./err_file_${TRAIN_MUTS_PATH}_${TEST_MUTS_PATH}_backbone_${ESM_DIR}_$$"
    OUT_FILE="./out_file_${TRAIN_MUTS_PATH}_${TEST_MUTS_PATH}_backbone_${ESM_DIR}_$$"
else
    ERR_FILE="./err_file_${TRAIN_MUTS_PATH}_${TEST_MUTS_PATH}_${ESM_DIR}_$$"
    OUT_FILE="./out_file_${TRAIN_MUTS_PATH}_${TEST_MUTS_PATH}_${ESM_DIR}_$$"
fi

bsub -n 6 -gpu num=1:gmem=12G:aff=yes -R same[gpumodel] -R rusage[mem=64GB] -R span[ptile=6] -e "$ERR_FILE" -o "$OUT_FILE" -q short-gpu "./$SCRIPT_NAME" 

# Wait a moment to ensure job is submitted before deleting
#sleep 2
#rm "$SCRIPT_NAME"
