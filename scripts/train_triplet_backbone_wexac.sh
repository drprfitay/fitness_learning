#!/bin/bash

# Usage: ./submit_epinnet.sh "1,2,3,4" 
# $1: comma-separated train mutation indices (e.g. "1,2,3,4")

if [ "$#" -ne 2 ]; then
    echo "Usage: $0 <train_indices_comma_separated> 
    exit 1
fi

TRAIN_INDICES_CSV="$1"

# Convert comma-separated to space-separated for passing as multiple arguments
TRAIN_INDICES_ARGS=(${TRAIN_INDICES_CSV//,/ })

TRAIN_MUTS="${TRAIN_INDICES_CSV//,/}"

SAVE_PATH="pretraining/esm8m/one_shot/triplet_backbone/train_${TRAIN_MUTS}"

SCRIPT_NAME="tmp/run_epinnet_${TRAIN_MUTS}_$$.sh"

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
    --config ../code/config.yaml \\
    --train_indices ${TRAIN_INDICES_ARGS[@]}
    
EOF

chmod +x "$SCRIPT_NAME"

ERR_FILE="./triplet_err_file_${TRAIN_MUTS}_$$"
OUT_FILE="./triplet_out_file_${TRAIN_MUTS}_$$"
bsub -n 6 -gpu num=1:gmem=12G:aff=yes -R same[gpumodel] -R rusage[mem=64GB] -R span[ptile=6] -o "$ERR_FILE" -e "$OUT_FILE" -q short-gpu "./$SCRIPT_NAME" 

# Wait a moment to ensure job is submitted before deleting
#sleep 2
#rm "$SCRIPT_NAME"
