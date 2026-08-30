"""
FireRed-OCR served on Modal (vLLM, A100-40GB).

Deployment shape mirrors deployments/minerU and deployments/pp_structure_v3 so
the cost numbers are directly comparable: one document per GPU container
(`max_inputs=1`), billed-lifetime accounting, `duration_seconds x
cost_per_second` echoed in each response.

!!! VERIFY BEFORE A REAL RUN !!!
MODEL_ID and PROMPT must be checked against the official FireRed-OCR model card
(FireRedTeam/FireRed-OCR, arXiv 2603.01840). FireRed-OCR is a full-page
generative parser (Qwen3-VL-based, 2B), so full-page-through-vLLM is the
intended serving mode, but the prompt and any structured-output constraints
must come from the model card / GitHub repo.

Deploy (from lm-inference/):
    modal deploy deployments/firered_ocr/inference_firered_ocr.py
Client:
    python deployments/vlm_ocr_client.py --endpoint https://<ws>--firered-ocr-app-parse-document-endpoint.modal.run ...
"""

import base64
import subprocess
import time

import modal
from fastapi import HTTPException, Request

APP_NAME = "firered-ocr-app"
MODEL_ID = "FireRedTeam/FireRed-OCR"  # VERIFY against the model card
PROMPT = (
    "Convert this document page to Markdown. Preserve the reading order, "
    "headings, paragraphs, tables (as Markdown tables) and formulas (as LaTeX). "
    "Output only the Markdown content."
)  # VERIFY against the model card
GPU_TYPE = "A100-40GB"
COST_PER_SECOND = 0.000583  # Modal A100-40GB, USD/s; keep in sync with lm_inference_utils.py
MAX_TOKENS = 8192
RENDER_DPI = 200

HF_CACHE = modal.Volume.from_name("firered-ocr-hf-cache", create_if_missing=True)
HF_CACHE_PATH = "/root/.cache/huggingface"

image = (
    modal.Image.from_registry("nvidia/cuda:12.4.1-devel-ubuntu22.04", add_python="3.11")
    .apt_install("git", "wget")
    .pip_install(
        "vllm==0.11.0",
        extra_options="--extra-index-url https://download.pytorch.org/whl/cu128 --no-cache-dir",
    )
    .pip_install(
        "pymupdf", "pillow", "httpx", "fastapi[standard]",
        extra_options="--no-cache-dir",
    )
    .env({"HF_HOME": HF_CACHE_PATH})
)

app = modal.App(APP_NAME, image=image)


@app.cls(
    gpu=GPU_TYPE,
    scaledown_window=300,
    max_containers=4,
    min_containers=1,
    timeout=1800,
    volumes={HF_CACHE_PATH: HF_CACHE},
)
# One document per container: concurrent requests fan out one-doc-per-GPU, so
# the per-document cost is not divided by a sharing factor.
@modal.concurrent(max_inputs=1)
class VlmOcrService:
    @modal.enter()
    def start_server(self):
        print(f"Starting vLLM server for {MODEL_ID} ...")
        self.server_process = subprocess.Popen(
            [
                "vllm", "serve", MODEL_ID,
                "--trust-remote-code",
                "--host", "0.0.0.0", "--port", "8000",
                "--served-model-name", "model",
                "--gpu-memory-utilization", "0.92",
                "--max-model-len", "16384",
            ]
        )
        import httpx

        for i in range(300):
            try:
                if httpx.get("http://localhost:8000/health", timeout=2).status_code == 200:
                    print("vLLM server ready.")
                    HF_CACHE.commit()  # persist the model download
                    return
            except Exception:
                pass
            time.sleep(1)
            if i and i % 30 == 0:
                print(f"... waiting for vLLM ({i}s)")
        raise RuntimeError("vLLM failed to start in 300s")

    @modal.exit()
    def stop_server(self):
        if getattr(self, "server_process", None):
            self.server_process.terminate()

    @modal.method()
    def parse_document(self, file_bytes: bytes, original_filename: str):
        import fitz  # pymupdf
        import httpx

        pages = []
        with fitz.open(stream=file_bytes, filetype="pdf") as doc:
            for page in doc:
                pix = page.get_pixmap(dpi=RENDER_DPI)
                pages.append(base64.b64encode(pix.tobytes("png")).decode("ascii"))
        print(f"{original_filename}: {len(pages)} pages")

        markdown_pages = []
        with httpx.Client(timeout=600) as client:
            for b64 in pages:
                payload = {
                    "model": "model",
                    "temperature": 0.0,
                    "max_tokens": MAX_TOKENS,
                    "messages": [
                        {
                            "role": "user",
                            "content": [
                                {"type": "image_url",
                                 "image_url": {"url": f"data:image/png;base64,{b64}"}},
                                {"type": "text", "text": PROMPT},
                            ],
                        }
                    ],
                }
                r = client.post("http://localhost:8000/v1/chat/completions", json=payload)
                r.raise_for_status()
                markdown_pages.append(r.json()["choices"][0]["message"]["content"].strip())

        return {
            "markdown_content": "\n\n".join(markdown_pages),
            "num_pages": len(pages),
        }


shared_service = VlmOcrService()


@app.function(cpu=2, memory=4096, timeout=1800, max_containers=1)
@modal.concurrent(max_inputs=10)
@modal.fastapi_endpoint(method="POST")
async def parse_document_endpoint(request: Request):
    from starlette.datastructures import UploadFile as StarletteUploadFile

    form = await request.form()
    upload = form.get("file")
    if not isinstance(upload, StarletteUploadFile):
        raise HTTPException(status_code=400, detail="multipart field 'file' required")
    file_bytes = await upload.read()

    t0 = time.time()
    try:
        result = await shared_service.parse_document.remote.aio(
            file_bytes, upload.filename or "document.pdf"
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)) from e
    duration = time.time() - t0
    result["duration_seconds"] = round(duration, 2)
    result["cost_usd"] = round(duration * COST_PER_SECOND, 6)
    return result
