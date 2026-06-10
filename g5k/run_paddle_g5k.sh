#!/bin/bash
#OAR -n paddle-ocr
#OAR -q production
#OAR -p gpu_count>0 AND gpu_compute_capability_major>=5
#OAR -l host=1,walltime=24:00:00
#OAR -O paddle.%jobid%.log
#OAR -E paddle.%jobid%.log
#
# ============================================================================
# Passive OAR batch job: run PaddleOCR layout detection (PP-DocLayout-L by
# default) over a folder of PDFs on a Grid5000 (Nancy) GPU node.
#
# It simply activates the prepared environment and calls the existing pipeline
# `python -m local.paddle_inference_batch`. See local/readme.md for the full
# behaviour of that pipeline.
# ============================================================================
#
# PREREQUISITES (set up by you, not by this script):
#   * A conda env named `paddle` containing paddlepaddle-gpu, paddleocr and
#     pdf2image, with the poppler binaries (pdftoppm) available on PATH.
#   * This repo checked out on the site's shared ~/home so the node can see it.
#
# OAR DIRECTIVES (the `#OAR` lines above — read only when submitted with -S):
#   -n  job name (shown in `oarstat`).
#   -q  queue; `production` is Nancy's queue for its GPU clusters.
#   -p  node-property filter: only nodes with a GPU of compute capability >= 5.
#   -l  resources: `host=1` reserves a whole node (so all its GPUs are usable),
#       `walltime=HH:MM:SS` is the max run time before OAR kills the job.
#   -O / -E  stdout / stderr log files (both -> paddle.log here).
#   Edit these lines to change walltime, queue, or to pin a GPU count.
#
# POSITIONAL ARGUMENTS (all optional; pass them after the script path):
#   $1  PDF_DIR     input folder of PDFs          (default: <repo>/pdfs)
#   $2  OUT_DIR     output folder                 (default: <repo>/output)
#   $3  MODEL_NAME  layout model, one of:         (default: PP-DocLayout-L)
#                     PP-DocLayout-S  small  (fastest)
#                     PP-DocLayout-M  medium (balanced)
#                     PP-DocLayout-L  large  (most accurate)
#                     PP-DocLayoutV2 / PP-DocBlockLayout
#   $4  WORKERS     parallel worker processes     (default: 8)
#                   NOTE: when GPUs are present the pipeline overrides this and
#                   runs exactly one worker (one model) per GPU — 1 worker for a
#                   single GPU, N for N GPUs. WORKERS only applies on CPU-only
#                   runs (capped at 4).
#
# BEHAVIOUR:
#   * One worker per GPU: 1 GPU -> 1 worker, N GPUs -> N workers, allocated
#     equally across the GPUs. Automatic, no flag needed.
#   * Verbose: this job passes --verbose so each PDF being processed is logged.
#   * Resumable: PDFs whose `<name>.json` already exists in OUT_DIR are skipped,
#     so resubmitting after a walltime kill continues where it left off. Add
#     `--force` to the python line below to reprocess everything.
#
# SUBMIT (with -S, so the #OAR directives are read; args go inside the quotes):
#   oarsub -S ./g5k/run_paddle_g5k.sh                                  # defaults
#   oarsub -S "./g5k/run_paddle_g5k.sh ./pdfs ./output PP-DocLayout-L 8"
#
# MONITOR:
#   oarstat -u            # your jobs and their state
#   tail -f paddle.log    # live progress
# ============================================================================
#
set -euo pipefail

# --- Config -----------------------------------------------------------------
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(dirname "$SCRIPT_DIR")"
VENV_DIR="$REPO_DIR/../.venv"
PDF_DIR="${1:-$REPO_DIR/pdfs}"
OUT_DIR="${2:-$REPO_DIR/output}"
MODEL_NAME="${3:-PP-DocLayout-L}"   # the "L" (large) layout model
WORKERS="${4:-8}"                    # 1 worker/GPU if multi-GPU; else capped at 4

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
    --workers "$WORKERS" \
    --cleanup-images \
    --verbose

echo "==> Finished. Results in $OUT_DIR"
