#!/bin/bash

# Usage: ./submit_epinnet.sh "1,2,3,4" "5,6,7,8"
# $1: comma-separated train mutation indices (e.g. "1,2,3,4")
# $2: comma-separated test mutation indices (e.g. "5,6,7,8")

if [ "$#" -ne 2 ]; then
    echo "Usage: $0 <train_indices_comma_separated> <test_indices_comma_separated>"
    exit 1
fi

TRAIN_INDICES_CSV="$1"
TEST_INDICES_CSV="$2"

# Convert comma-separated to python list string
TRAIN_INDICES_LIST="[$TRAIN_INDICES_CSV]"
TEST_INDICES_LIST="[$TEST_INDICES_CSV]"

TRAIN_MUTS="${TRAIN_INDICES_CSV//,/}"
TEST_MUTS="${TEST_INDICES_CSV//,/}"

SAVE_PATH="pretraining/esm8m/zero_shot/train_${TRAIN_MUTS}_test_${TEST_MUTS}"

SCRIPT_NAME="tmp/run_epinnet_${TRAIN_MUTS}_${TEST_MUTS}_$$.sh"

cat <<EOF > "$SCRIPT_NAME"
#!/bin/bash
source ~/.bashrc
conda activate esm_env
python ../code/train_epinnet.py \\
    --train False \\
    --evaluate_train True \\
    --evaluate_test True \\
    --save_path "$SAVE_PATH" \\
    --train_indices "$TRAIN_INDICES_LIST" \\
    --test_indices "$TEST_INDICES_LIST" 
EOF

chmod +x "$SCRIPT_NAME"

bsub -n 6 -gpu num=1gmem=12G:aff=yes -R same[gpumodel] -R span[ptile=6] -q short-gpu "./$SCRIPT_NAME"

# Wait a moment to ensure job is submitted before deleting
sleep 2
rm "$SCRIPT_NAME"
