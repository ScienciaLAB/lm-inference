#!/bin/bash
#
# One-time environment bootstrap for running PaddleOCR layout detection on Grid5000.
# Run this once on a Nancy frontend (or interactively on a node) before submitting
# the OAR job in run_paddle_g5k.sh.
#
# Usage:
#   bash g5k/setup_env_g5k.sh            # CPU paddlepaddle wheel (repo default)
#   GPU_WHEEL=1 bash g5k/setup_env_g5k.sh   # swap in the CUDA paddlepaddle-gpu wheel
#
set -euo pipefail

# --- Config -----------------------------------------------------------------
VENV_DIR="${VENV_DIR:-$HOME/venvs/paddle-g5k}"
# Repo root = parent of the directory holding this script.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="${REPO_DIR:-$(dirname "$SCRIPT_DIR")}"
GPU_WHEEL="${GPU_WHEEL:-0}"
REQUIREMENTS="$REPO_DIR/local/requirement.paddle.ladas.txt"

echo "==> Repo dir:    $REPO_DIR"
echo "==> Venv dir:    $VENV_DIR"
echo "==> GPU wheel:   $GPU_WHEEL"

# --- Python version sanity check (repo targets 3.11) ------------------------
echo "==> python3 version: $(python3 --version 2>&1)"

# --- Create / reuse virtualenv ----------------------------------------------
if [ ! -d "$VENV_DIR" ]; then
    echo "==> Creating virtualenv at $VENV_DIR"
    python3 -m venv "$VENV_DIR"
else
    echo "==> Reusing existing virtualenv at $VENV_DIR"
fi

# shellcheck disable=SC1091
source "$VENV_DIR/bin/activate"

# --- Install dependencies ---------------------------------------------------
echo "==> Upgrading pip"
pip install --upgrade pip

echo "==> Installing requirements from $REQUIREMENTS"
pip install -r "$REQUIREMENTS"

if [ "$GPU_WHEEL" = "1" ]; then
    # The repo pins the CPU wheel (paddlepaddle==3.2.0). On a reserved GPU node you
    # usually want the CUDA build instead. The cuXXX index must match the node's
    # NVIDIA driver/CUDA — cu126 works on most current Nancy GPU clusters; check
    # `nvidia-smi` and switch to cu118/cu123/etc. if needed.
    echo "==> Installing CUDA paddlepaddle-gpu wheel (overrides the CPU wheel)"
    pip install paddlepaddle-gpu==3.2.0 \
        -i https://www.paddlepaddle.org.cn/packages/stable/cu126/
fi

# --- Poppler check (pdf2image needs the pdftoppm binary) --------------------
# pdf2image shells out to poppler's `pdftoppm`. On Grid5000's default (non-deploy)
# production environment you cannot `apt install`, so detect it up front and give
# actionable guidance instead of failing deep inside inference.
if command -v pdftoppm >/dev/null 2>&1; then
    echo "==> poppler found: $(command -v pdftoppm)"
else
    echo "!!! WARNING: 'pdftoppm' (poppler) NOT found — pdf2image will fail at runtime."
    echo "    Options to get poppler without root on Grid5000:"
    echo "      * module avail poppler   &&  module load poppler"
    echo "      * conda install -c conda-forge poppler"
    echo "      * guix install poppler"
    echo "      * or run inside a deploy job and 'sudo-g5k apt-get install poppler-utils'"
fi

echo "==> Done. Activate later with: source $VENV_DIR/bin/activate"
