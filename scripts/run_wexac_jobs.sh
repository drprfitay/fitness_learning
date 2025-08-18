#!/bin/bash

# Usage: ./submit_epinnet.sh "1,2,3,4" "5,6,7,8" [--backbone]
# $1: comma-separated train mutation indices (e.g. "1,2,3,4")
# $2: comma-separated test mutation indices (e.g. "5,6,7,8")
# $3: optional, if set to --backbone, loads triplet backbone weights

if [ "$#" -lt 2 ] || [ "$#" -gt 3 ]; then
    echo "Usage: $0 <train_indices_comma_separated> <test_indices_comma_separated> [--backbone]"
    exit 1
fi

TRAIN_INDICES_CSV="$1"
TEST_INDICES_CSV="$2"
USE_BACKBONE=false

if [ "$#" -eq 3 ] && [ "$3" == "--backbone" ]; then
    USE_BACKBONE=true
fi

# Convert comma-separated to space-separated for passing as multiple arguments
TRAIN_INDICES_ARGS=(${TRAIN_INDICES_CSV//,/ })
TEST_INDICES_ARGS=(${TEST_INDICES_CSV//,/ })

TRAIN_MUTS="${TRAIN_INDICES_CSV//,/}"
TEST_MUTS="${TEST_INDICES_CSV//,/}"

if [ "$USE_BACKBONE" = true ]; then
    # Save path for backbone mode
    SAVE_PATH="pretraining/esm8m/one_shot/triplet_backbone/train_${TRAIN_MUTS}/train_${TRAIN_MUTS}_test_${TEST_MUTS}"
    WEIGHTS_PATH="pretraining/esm8m/one_shot/triplet_backbone/train_${TRAIN_MUTS}/final_model.pt"
    EXTRA_ARGS="--load_weights True --weights_path \"$WEIGHTS_PATH\""
    SCRIPT_NAME="tmp/run_epinnet_${TRAIN_MUTS}_${TEST_MUTS}_backbone_$$.sh"
else
    # Default zero-shot save path
    SAVE_PATH="pretraining/esm8m/zero_shot/train_${TRAIN_MUTS}_test_${TEST_MUTS}"
    EXTRA_ARGS=""
    SCRIPT_NAME="tmp/run_epinnet_${TRAIN_MUTS}_${TEST_MUTS}_$$.sh"
fi

cat <<EOF > "$SCRIPT_NAME"
#!/bin/bash
source ~/.bashrc
conda activate esm_env
python -u ../code/train_epinnet.py \\
    --train False \\
    --evaluate_train True \\
    --evaluate_test True \\
    --save_path "$SAVE_PATH" \\
    --config ../code/config.yaml \\
    --train_indices ${TRAIN_INDICES_ARGS[@]} \\
    --test_indices ${TEST_INDICES_ARGS[@]} \\
    $EXTRA_ARGS
EOF

chmod +x "$SCRIPT_NAME"

if [ "$USE_BACKBONE" = true ]; then
    ERR_FILE="./err_file_${TRAIN_MUTS}_${TEST_MUTS}_backbone_$$"
    OUT_FILE="./out_file_${TRAIN_MUTS}_${TEST_MUTS}_backbone_$$"
else
    ERR_FILE="./err_file_${TRAIN_MUTS}_${TEST_MUTS}_$$"
    OUT_FILE="./out_file_${TRAIN_MUTS}_${TEST_MUTS}_$$"
fi

bsub -n 6 -gpu num=1:gmem=12G:aff=yes -R same[gpumodel] -R rusage[mem=64GB] -R span[ptile=6] -o "$ERR_FILE" -e "$OUT_FILE" -q short-gpu "./$SCRIPT_NAME" 

# Wait a moment to ensure job is submitted before deleting
#sleep 2
#rm "$SCRIPT_NAME"
