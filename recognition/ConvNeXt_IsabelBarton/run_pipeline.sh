#!/usr/bin/env bash
# Full pipeline for one run on the GPU:
#   1. check the data + recompute the training-set normalisation (check_data.py)
#   2. train, validating every epoch, then test the best validation checkpoint once (main.py)
#   3. analyse the test predictions: patient level, confusion matrix, ROC, calibration, reject curve, examples (predict.py)
#   4. plot the training curves and learning-rate schedule (plot_log.py)
#
# Usage (from anywhere):
#     ./run_pipeline.sh                      # full run with main.py's defaults, into ~/Desktop/results/<NAME>
#     NAME=lr1e-3 ./run_pipeline.sh --lr 1e-3   # any extra arguments are passed to main.py
#     QUICK=1 ./run_pipeline.sh              # smoke test: 1 epoch into ~/Desktop/results/quick_test (~2 min)
#
# Long runs (keeps going after the terminal closes):
#     nohup ./run_pipeline.sh > pipeline.log 2>&1 &
#
# Re-running the same NAME resumes from the last checkpoint (main.py does this automatically); a run that already
# has test_results.json skips training and only redoes steps 3-4.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
CODE_DIR="$SCRIPT_DIR/../recognition/ConvNeXt_IsabelBarton"

PYTHON=${PYTHON:-python3}
DATA=${DATA:-$HOME/Desktop/data/ADNI_patientleakagefix/AD_NC}
RESULTS=${RESULTS:-$HOME/Desktop/results}
QUICK=${QUICK:-0}

if [ "$QUICK" = 1 ]; then
    NAME=${NAME:-quick_test}
    # 1 epoch, no warmup, and keep the run small; delete the folder first so it never resumes
    EXTRA=(--epochs 1 --warmup_epochs 0 --save_ckpt_num 1)
    rm -rf "$RESULTS/$NAME"
else
    NAME=${NAME:-convnext_t}
    EXTRA=()
fi
OUT="$RESULTS/$NAME"
mkdir -p "$OUT"


log() { echo "[$(date '+%F %T')] $*"; }

log "run: $NAME -> $OUT"
"$PYTHON" -c "import torch, sys; sys.exit(0 if torch.cuda.is_available() else 'No CUDA GPU visible to PyTorch')"

# 1. data check + normalisation (mean/std from the training split only)
log "step 1/4: checking data and recomputing normalisation stats"
"$PYTHON" "$SCRIPT_DIR/check_data.py" --data_root "$DATA" --recompute_stats 2>&1 | tee "$OUT/data_check.log"

# 2. train + validate every epoch + test the best checkpoint
if [ -f "$OUT/test_results.json" ]; then
    log "step 2/4: $OUT/test_results.json exists, skipping training"
else
    log "step 2/4: training (follow with: tail -f $OUT/train.log)"
    (cd "$CODE_DIR" && "$PYTHON" -u main.py \
        --data_path "$DATA/train" --val_data_path "$DATA/validation" --eval_data_path "$DATA/test" \
        --output_dir "$OUT" \
        "${EXTRA[@]}" "$@") >> "$OUT/train.log" 2>&1
    log "training finished"
    "$PYTHON" -c "import json; r = json.load(open('$OUT/test_results.json')); \
print(f\"  test accuracy {r['test_acc1']:.2f}%, AD recall {r['test_ad_recall']:.3f}, NC recall {r['test_nc_recall']:.3f}\")"
fi

# 3. analysis of the saved test predictions (no model needed)
log "step 3/4: analysing test predictions"
(cd "$CODE_DIR" && "$PYTHON" predict.py --predictions "$OUT/test_predictions.csv" --output_dir "$OUT") \
    2>&1 | tee "$OUT/predict.log"

# 4. training curves
log "step 4/4: plotting training curves"
"$PYTHON" "$SCRIPT_DIR/plot_log.py" "$OUT"

log "done: results in $OUT"
