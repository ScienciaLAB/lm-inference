#!/bin/bash
# PP-StructureV3 batch inference as a SLURM array (one GPU per task).
#
# The runner (local/ppstructurev3_inference_batch.py) has no shard option, so
# each task builds a symlink shard of every N-th PDF and processes it into its
# own output subdirectory; the runner's --resume makes re-submits safe.
#
# Submit (DFKI — adjust partition/account):
#   mkdir -p logs
#   INPUT=/path/to/pdfs OUTPUT=/path/to/out sbatch --array=0-3 slurm/run_ppstructurev3.sbatch
#
#SBATCH --job-name=ppsv3
#SBATCH --partition=A100-40GB        # DFKI: e.g. A100-40GB, A100-80GB, RTXA6000
##SBATCH --account=<project>
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --output=logs/%x-%A_%a.out
set -euo pipefail

INPUT=${INPUT:?set INPUT=/path/to/pdfs}
OUTPUT=${OUTPUT:?set OUTPUT=/path/to/output}
REPO=${REPO:-$(cd "$(dirname "$0")/.." && pwd)}

VENV=${VENV:-$REPO/.venv-ppsv3}
if [ ! -x "$VENV/bin/python" ]; then
  python3 -m venv "$VENV"
  "$VENV/bin/pip" install --upgrade pip
  # GPU paddle wheel from PaddlePaddle's own index (cu126), then paddleocr.
  "$VENV/bin/pip" install paddlepaddle-gpu==3.2.2 \
      --index-url https://www.paddlepaddle.org.cn/packages/stable/cu126/
  "$VENV/bin/pip" install "paddleocr[all]>=3.2.0" pdf2image
fi
export PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK=True
export PADDLE_PDX_CACHE_HOME=${PADDLE_PDX_CACHE_HOME:-$OUTPUT/paddlex-cache}

N=${SLURM_ARRAY_TASK_COUNT:-1}
K=${SLURM_ARRAY_TASK_ID:-0}
SHARD_DIR="$OUTPUT/shards/shard_$K"
mkdir -p "$SHARD_DIR" "$OUTPUT/out_shard_$K"
i=0
for f in "$INPUT"/*.pdf; do
  [ $((i % N)) -eq "$K" ] && ln -sf "$f" "$SHARD_DIR/"
  i=$((i + 1))
done

srun "$VENV/bin/python" "$REPO/local/ppstructurev3_inference_batch.py" \
  "$SHARD_DIR" -o "$OUTPUT/out_shard_$K" --devices gpu:0
