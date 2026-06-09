#!/bin/bash
#OAR -n paddle-ocr
#OAR -q production
#OAR -l host=1,walltime=10:00:00
#OAR -O paddle.log
#OAR -E paddle.log
#
# Passive OAR batch job: run PaddleOCR PP-DocLayout-L layout detection over a folder
# of PDFs on a Grid5000 (Nancy) GPU node.
#
# Submit with:
#   oarsub -S ./g5k/run_paddle_g5k.sh
#
# The #OAR directives above set the job name, queue, resources (1 host / 1 GPU,
# 2h walltime) and log files. Edit walltime / gpu count there if needed.
#
# Override the defaults below by editing this file, or run setup first with
# setup_env_g5k.sh so the venv exists at $VENV_DIR.
#
set -euo pipefail

# --- Config (override by editing) -------------------------------------------
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="${REPO_DIR:-$(dirname "$SCRIPT_DIR")}"
VENV_DIR="${VENV_DIR:-REPO_DIR/.venv}"
PDF_DIR="${PDF_DIR:-${BASH_SOURCE[1]}/pdfs}"
OUT_DIR="${OUT_DIR:-${BASH_SOURCE[1]}/output}"
MODEL_NAME="${MODEL_NAME:-PP-DocLayout-L}"   # the "L" (large) layout model
WORKERS="${WORKERS:-8}"                       # batch script caps workers at 4

# --- Job context (for the log) ----------------------------------------------
echo "==> Job ${OAR_JOB_ID:-<interactive>} on host $(hostname)"
echo "==> Repo:    $REPO_DIR"
echo "==> PDFs:    $PDF_DIR"
echo "==> Output:  $OUT_DIR"
echo "==> Model:   $MODEL_NAME   Workers: $WORKERS"
echo "==> GPU:"
nvidia-smi || echo "!!! nvidia-smi failed — is this a GPU node?"

# --- Run --------------------------------------------------------------------
# shellcheck disable=SC1091
source "$VENV_DIR/bin/activate"
cd "$REPO_DIR"

python -m local.paddle_inference_batch "$PDF_DIR" \
    -o "$OUT_DIR" \
    --model-name "$MODEL_NAME" \
    --workers "$WORKERS"

echo "==> Finished. Results in $OUT_DIR"
