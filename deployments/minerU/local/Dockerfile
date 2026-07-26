# MinerU 2.5 inference server (vLLM) for a local Linux host with a Blackwell GPU.
#
# Architecture: the vLLM OpenAI engine (mineru-vllm-server) stays warm and hosts
# MinerU2.5-2509; MinerU's HTTP API (mineru-api) routes each PDF upload to it over
# HTTP. No `mineru` CLI is spawned per request.
#
# Blackwell notes (B100/B200 = sm_100, RTX 50-series = sm_120):
#   * Needs NVIDIA driver >= 570 on the host and the nvidia-container-toolkit.
#   * Needs a vLLM/torch build carrying Blackwell (sm_100/sm_120) kernels — i.e.
#     CUDA 12.8+. We base on vLLM's official image and REUSE its vLLM so the GPU
#     kernels match your card; pick a tag known to support your GPU (MinerU's own
#     Blackwell reference stack is vllm 0.11.2 + torch 2.9 on CUDA 12.8).
#
# Build:  docker build --build-arg VLLM_TAG=v0.11.2 -t mineru-vllm-local .
ARG VLLM_TAG=latest
FROM vllm/vllm-openai:${VLLM_TAG}

# Curl is used by start.sh to poll the vLLM health endpoint.
RUN (command -v curl >/dev/null 2>&1) || (apt-get update && apt-get install -y --no-install-recommends curl && rm -rf /var/lib/apt/lists/*)

# MinerU core provides the `mineru`, `mineru-vllm-server` and `mineru-api` console
# scripts plus mineru_vl_utils (the MinerULogitsProcessor). We deliberately do NOT
# install the [vllm] extra, so the base image's Blackwell-capable vLLM is reused
# instead of pip pulling a (possibly non-Blackwell) pinned build over it.
RUN pip install --no-cache-dir "mineru[core]"

# Work around a prometheus_fastapi_instrumentator bug that 500s EVERY route on the
# vLLM OpenAI server (its route-name lookup does `route.path` on routes lacking it
# -> AttributeError: '_IncludedRouter' object has no attribute 'path'), which
# otherwise makes /health and inference return 500 and the server never ready.
RUN P=$(python -c "import os,prometheus_fastapi_instrumentator as m;print(os.path.join(os.path.dirname(m.__file__),'routing.py'))") \
    && sed -i 's/route_name = route\.path/route_name = getattr(route, "path", "")/' "$P" \
    && grep -n 'getattr(route' "$P"

ENV MINERU_MODEL_SOURCE=huggingface \
    HF_HOME=/root/.cache/huggingface \
    OMP_NUM_THREADS=1

COPY start.sh /usr/local/bin/start.sh
RUN chmod +x /usr/local/bin/start.sh

EXPOSE 8000
# Override the base image's vLLM entrypoint with our two-server launcher.
ENTRYPOINT ["/usr/local/bin/start.sh"]
