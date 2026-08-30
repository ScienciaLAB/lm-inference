#!/bin/bash
# GLM-OCR / FireRed-OCR batch inference as a SLURM array (one GPU per task).
#
# Each array task processes every N-th PDF via the runner's --shard option;
# resume is per-shard CSV, so a re-submit skips finished documents.
#
# Submit (DFKI — adjust partition/account to the cluster you are on):
#   mkdir -p logs
#   MODEL=glm-ocr    INPUT=/path/to/pdfs OUTPUT=/path/to/out sbatch --array=0-3 slurm/run_vlm_ocr.sbatch
#   MODEL=firered-ocr INPUT=/path/to/pdfs OUTPUT=/path/to/out sbatch --array=0-3 slurm/run_vlm_ocr.sbatch
#
#SBATCH --job-name=vlm-ocr
#SBATCH --partition=A100-40GB        # DFKI: e.g. A100-40GB, A100-80GB, RTXA6000
##SBATCH --account=<project>         # uncomment if the cluster requires it
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --output=logs/%x-%A_%a.out
set -euo pipefail

MODEL=${MODEL:?set MODEL=glm-ocr|firered-ocr}
INPUT=${INPUT:?set INPUT=/path/to/pdfs}
OUTPUT=${OUTPUT:?set OUTPUT=/path/to/output}
REPO=${REPO:-$(cd "$(dirname "$0")/.." && pwd)}   # lm-inference checkout

# --- environment -------------------------------------------------------------
# DFKI note: if the cluster mandates enroot/pyxis containers, replace this block
# with the srun --container-image=... invocation and drop the venv.
VENV=${VENV:-$REPO/.venv-vlm}
if [ ! -x "$VENV/bin/python" ]; then
  python3 -m venv "$VENV"
  "$VENV/bin/pip" install --upgrade pip
  "$VENV/bin/pip" install "vllm==0.11.0" pymupdf
fi
export HF_HOME=${HF_HOME:-$OUTPUT/hf-cache}      # shared model cache across tasks

N=${SLURM_ARRAY_TASK_COUNT:-1}
K=${SLURM_ARRAY_TASK_ID:-0}
mkdir -p "$OUTPUT/$MODEL"

srun "$VENV/bin/python" "$REPO/local/vlm_ocr_batch.py" "$INPUT" \
  -o "$OUTPUT/$MODEL" --model "$MODEL" --shard "$K/$N" \
  --csv "$OUTPUT/$MODEL/${MODEL}_results_shard${K}.csv"
