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
# Positional arguments (all optional, with defaults):
#   $1  PDF_DIR     input folder of PDFs        (default: <repo>/pdfs)
#   $2  OUT_DIR     output folder              (default: <repo>/output)
#   $3  MODEL_NAME  layout model               (default: PP-DocLayout-L)
#   $4  WORKERS     parallel workers           (default: 8)
#
# Submit with (-S reads the #OAR directives; args go after the script path):
#   oarsub -S "./g5k/run_paddle_g5k.sh ./pdfs ./output PP-DocLayout-L 8"
#
# The #OAR directives above set the job name, queue, resources and log files.
# Edit walltime / gpu count there if needed.
#
set -euo pipefail

# --- Config -----------------------------------------------------------------
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(dirname "$SCRIPT_DIR")"
VENV_DIR="$REPO_DIR/../.venv"
PDF_DIR="${1:-$REPO_DIR/pdfs}"
OUT_DIR="${2:-$REPO_DIR/output}"
MODEL_NAME="${3:-PP-DocLayout-L}"   # the "L" (large) layout model
WORKERS="${4:-8}"                    # batch script caps workers at 4

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
#source "$VENV_DIR/bin/activate"
module load conda
conda activate paddle
cd "$REPO_DIR"

python -m local.paddle_inference_batch "$PDF_DIR" \
    -o "$OUT_DIR" \
    --model-name "$MODEL_NAME" \
    --workers "$WORKERS"

echo "==> Finished. Results in $OUT_DIR"
