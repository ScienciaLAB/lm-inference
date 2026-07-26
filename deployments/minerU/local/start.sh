#!/usr/bin/env bash
# Boot the two long-lived MinerU 2.5 processes:
#   1. mineru-vllm-server  -> the vLLM OpenAI engine hosting the VLM
#   2. mineru-api          -> MinerU's HTTP API, talking to (1) as an http-client
# No `mineru` CLI is invoked per request: the engine stays warm and mineru-api
# routes each upload to it.
set -euo pipefail

MODEL="${MINERU_VLM_MODEL:-opendatalab/MinerU2.5-2509-1.2B}"
VLLM_PORT="${VLLM_PORT:-30000}"
API_PORT="${API_PORT:-8000}"
GPU_MEM_UTIL="${GPU_MEM_UTIL:-0.90}"
TP_SIZE="${TP_SIZE:-1}"

echo ">> [1/2] Starting MinerU vLLM server (model=${MODEL}) on :${VLLM_PORT}"
# --model overrides MinerU's default (which is now the newer Pro variant); pin
# MinerU2.5-2509 for reproducibility. mineru-vllm-server injects the MinerU
# logits processor automatically.
mineru-vllm-server \
  --model "${MODEL}" \
  --host 0.0.0.0 --port "${VLLM_PORT}" \
  --gpu-memory-utilization "${GPU_MEM_UTIL}" \
  --tensor-parallel-size "${TP_SIZE}" &
VLLM_PID=$!

echo ">> Waiting for vLLM /health (first run downloads the ~2.4GB weights)..."
for i in $(seq 1 900); do
  if curl -sf "http://localhost:${VLLM_PORT}/health" >/dev/null 2>&1; then
    echo ">> vLLM server ready after ${i}s"
    break
  fi
  if ! kill -0 "${VLLM_PID}" 2>/dev/null; then
    echo "!! vLLM server exited during startup" >&2
    exit 1
  fi
  sleep 2
done

echo ">> [2/2] Starting MinerU API on :${API_PORT} (backend=vlm-http-client -> vLLM)"
# NOTE: confirm the flag name with `mineru-api --help` on your MinerU version
# (some releases use --server-url, others --server_url).
exec mineru-api \
  --host 0.0.0.0 --port "${API_PORT}" \
  --backend vlm-http-client \
  --server-url "http://localhost:${VLLM_PORT}"
