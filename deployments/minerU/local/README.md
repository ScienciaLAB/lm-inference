# MinerU 2.5 — local vLLM inference server (Blackwell GPU)

Self-hosted alternative to the Modal deployment. It runs MinerU 2.5 as a **warm
vLLM server + MinerU's HTTP API** on a local Linux box with a Blackwell GPU —
**no `mineru` CLI is spawned per request**.

```
PDF ──HTTP──▶ mineru-api (:8000) ──http-client──▶ mineru-vllm-server (:30000, vLLM engine)
                                                     └─ opendatalab/MinerU2.5-2509-1.2B
```

## Why two servers
- `mineru-vllm-server` — wraps `vllm serve` with MinerU's defaults and the
  `MinerULogitsProcessor`; loads the VLM once and keeps it warm.
- `mineru-api` — MinerU's FastAPI service; with `--backend vlm-http-client` it
  forwards each upload to the vLLM engine. This replaces the per-request CLI.

## Prerequisites (Blackwell: B100/B200 = sm_100, RTX 50-series = sm_120)
- NVIDIA driver **≥ 570** on the host (`nvidia-smi` works).
- Docker + **nvidia-container-toolkit** (`docker run --rm --gpus all nvidia/cuda:12.8.1-base-ubuntu22.04 nvidia-smi` succeeds).
- A vLLM/torch build with Blackwell kernels — i.e. **CUDA 12.8+**. This image
  reuses vLLM from the `vllm/vllm-openai` base, so pick a base tag that supports
  your card (MinerU's reference Blackwell stack is **vllm 0.11.2 + torch 2.9**).

## Run (docker compose)
```bash
cd deployments/minerU/local
# VLLM_TAG in docker-compose.yml selects the vLLM base image (default v0.11.2)
docker compose up --build
```
First boot downloads the ~2.4 GB weights into the `hf-cache` volume (persisted).
When you see `vLLM server ready`, the API is live on `:8000`.

## Run (plain Docker)
```bash
docker build --build-arg VLLM_TAG=v0.11.2 -t mineru-vllm-local .
docker run --rm --gpus all -p 8000:8000 --shm-size=16g --ipc=host \
  -v mineru-hf-cache:/root/.cache/huggingface \
  -e MINERU_VLM_MODEL=opendatalab/MinerU2.5-2509-1.2B \
  mineru-vllm-local
```

## Run (bare metal, no Docker)
```bash
uv pip install "mineru[core,vllm]"            # or into a venv/conda env
# apply the prometheus patch once (see Dockerfile) if /health returns 500:
P=$(python -c "import os,prometheus_fastapi_instrumentator as m;print(os.path.join(os.path.dirname(m.__file__),'routing.py'))")
sed -i 's/route_name = route\.path/route_name = getattr(route, "path", "")/' "$P"
MINERU_MODEL_SOURCE=huggingface ./start.sh
```

## Usage
`mineru-api` exposes MinerU's standard endpoints. Confirm the exact routes with
your version's OpenAPI docs at `http://localhost:8000/docs`. Typical call:
```bash
curl -X POST "http://localhost:8000/file_parse" \
  -F "files=@paper.pdf" \
  -F "backend=vlm-http-client" \
  -F "server_url=http://localhost:30000"
```

## Configuration (env vars, see `start.sh` / `docker-compose.yml`)
| Var | Default | Meaning |
|---|---|---|
| `MINERU_VLM_MODEL` | `opendatalab/MinerU2.5-2509-1.2B` | VLM model (pinned to the Sept-2025 release, not the newer Pro variant) |
| `VLLM_PORT` | `30000` | vLLM engine port |
| `API_PORT` | `8000` | MinerU HTTP API port |
| `GPU_MEM_UTIL` | `0.90` | vLLM `--gpu-memory-utilization` |
| `TP_SIZE` | `1` | tensor-parallel size (number of GPUs) |

## Gotchas
- **prometheus_fastapi_instrumentator crash.** MinerU's vLLM server 500s on every
  route (incl. `/health`) because the metrics middleware does `route.path` on a
  route type that lacks it (`_IncludedRouter`). The Dockerfile patches that line;
  this is independent of the vLLM version.
- **`mineru-api` flag names.** Some MinerU releases use `--server-url`, others
  `--server_url`. If startup errors on the flag, run `mineru-api --help` and fix
  `start.sh` accordingly.
- **Model variant.** `--model` overrides MinerU's default (now `MinerU2.5-Pro-2605`,
  a 2026 model). If you also need the matching logits processor for an older model,
  pin `mineru-vl-utils` to the corresponding release.
- **Blackwell kernels.** If vLLM errors with "no kernel image is available for
  execution on the device", the base image's CUDA/torch predates your GPU — pick a
  newer `VLLM_TAG` (CUDA 12.8+).
