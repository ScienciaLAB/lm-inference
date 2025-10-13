import asyncio
import base64
import json
import os
import time
from pathlib import Path
from typing import BinaryIO, Dict, List, Union
from uuid import uuid4

import fitz  # PyMuPDF
import modal
from fastapi import FastAPI, UploadFile, File, HTTPException
from openai import AsyncOpenAI

# ----------------------------------------------------------------------
# Modal image (unchanged except for a couple of package version pins)
# ----------------------------------------------------------------------
cuda_version = "12.8.0"
flavor = "devel"
operating_sys = "ubuntu22.04"
tag = f"{cuda_version}-{flavor}-{operating_sys}"

image = (
    modal.Image.from_registry(f"nvidia/cuda:{tag}", add_python="3.11")
    .pip_install(
        "vllm>=0.4.0",
        "huggingface_hub[hf_transfer]==0.26.2",
        "flashinfer-python==0.2.0.post2",
        "PyMuPDF",
        "fastapi[standard]==0.115.4",
        "openai",
        "requests==2.32.3",
        "pydantic==2.9.2",
        "transformers==4.54.1",
        "torch==2.7.1",
        "numpy<2",
        extra_index_url="https://flashinfer.ai/whl/cu124/torch2.5",
    )
    .env({
        "HF_HUB_ENABLE_HF_TRANSFER": "1",
        "TOKENIZERS_PARALLELISM": "false",
        "PYTORCH_CUDA_ALLOC_CONF": "max_split_size_mb:512",
    })
)

# ----------------------------------------------------------------------
# Global config
# ----------------------------------------------------------------------
MODEL_PATH = "Qwen/Qwen2.5-VL-7B-Instruct"
GPU_TYPE = os.environ.get("GPU_TYPE", "l40s")
GPU_COUNT = os.environ.get("GPU_COUNT", 1)

GPU_CONFIG = f"{GPU_TYPE}:{GPU_COUNT}"
GPU_MEMORY_UTILIZATION = 0.9
VLLM_PORT = 8000
MINUTES = 60

# Volumes
MODEL_VOL = modal.Volume.from_name("sgl-cache", create_if_missing=True)
HF_CACHE_VOL = modal.Volume.from_name("huggingface-cache", create_if_missing=True)

# App
app = modal.App("qwen-2.5-vl-7b-instruct-vllm")

# ----------------------------------------------------------------------
# Helper: extract PDF pages (pure-CPU, runs on a cheap function)
# ----------------------------------------------------------------------
@app.function(
    image=image,
    cpu=4,
    memory=4096,
    timeout=600,
    volumes={"/root/.cache/huggingface": HF_CACHE_VOL},
    secrets=[modal.Secret.from_name("document-qa-api-key")],
)
def extract_pdf_pages(pdf_content: bytes, dpi: int = 150) -> List[Dict]:
    """Return a list of dicts: {page_number, image_base64, width, height, dpi}."""
    try:
        doc = fitz.open(stream=pdf_content, filetype="pdf")
        pages = []
        zoom = dpi / 72.0
        mat = fitz.Matrix(zoom, zoom)

        for page_no in range(len(doc)):
            page = doc.load_page(page_no)
            pix = page.get_pixmap(matrix=mat)
            img_bytes = pix.tobytes("png")
            pages.append(
                {
                    "page_number": page_no + 1,
                    "image_base64": base64.b64encode(img_bytes).decode(),
                    "width": pix.width,
                    "height": pix.height,
                    "dpi": dpi,
                }
            )
        doc.close()
        return pages
    except Exception as e:
        print(f"PDF extraction error: {e}")
        return []


# ----------------------------------------------------------------------
# vLLM model class (GPU)
# ----------------------------------------------------------------------
@app.cls(
    gpu=GPU_CONFIG,
    memory=16384,
    timeout=20 * MINUTES,
    # NEW: autoscaling window (replaces container_idle_timeout)
    scaledown_window=20 * MINUTES,
    concurrency_limit=100,          # replaces @modal.concurrent
    image=image,
    volumes={"/cache": MODEL_VOL, "/root/.cache/huggingface": HF_CACHE_VOL},
    secrets=[modal.Secret.from_name("document-qa-api-key")],
)
class VLMMModel:
    # NO __init__ – everything is done in @enter
    @modal.enter()
    async def start_vllm_client(self):
        self.client = AsyncOpenAI(
            base_url=f"http://localhost:{VLLM_PORT}/v1",
            api_key=os.environ.get("API_KEY"),
        )
        print("vLLM client ready")

    # ------------------------------------------------------------------
    # Helper that talks to the model for a *single* page
    # ------------------------------------------------------------------
    async def _extract_page_structure(self, image_b64: str, page_number: int) -> Dict:
        prompt = """
        ... (your long prompt – unchanged, omitted here for brevity) ...
        """
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt.replace("<page_number>", str(page_number))},
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/png;base64,{image_b64}"},
                    },
                ],
            }
        ]

        try:
            resp = await self.client.chat.completions.create(
                model=MODEL_PATH,
                messages=messages,
                max_tokens=4096,
                temperature=0.1,
            )
            raw = resp.choices[0].message.content

            # ---- robust JSON extraction ----
            start = raw.find("{")
            end = raw.rfind("}") + 1
            json_part = raw[start:end] if start != -1 and end > 0 else raw
            data = json.loads(json_part)

            # guarantee shape
            for sec in ("header", "body", "annex"):
                data.setdefault(sec, {"passages": []})
            return {"success": True, "data": data, "raw_response": raw}

        except Exception as exc:
            return {
                "success": False,
                "error": str(exc),
                "page_number": page_number,
                "data": {"header": {"passages": []}, "body": {"passages": []}, "annex": {"passages": []}},
           }

    # ------------------------------------------------------------------
    # Public method used by the CPU-side orchestrator
    # ------------------------------------------------------------------
    async def extract_document_structure(self, pages: List[Dict]) -> Dict:
        """Run per-page extraction concurrently."""
        tasks = [
            self._extract_page_structure(p["image_base64"], p["page_number"])
            for p in pages
        ]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        out = {
            "document_info": {
                "total_pages": len(pages),
                "extraction_timestamp": time.time(),
                "model_used": MODEL_PATH,
            },
            "pages": [],
            "summary": {
                "total_passages": 0,
                "total_references": 0,
                "structure_counts": {"header_passages": 0, "body_passages": 0, "annex_passages": 0},
            },
        }

        for idx, res in enumerate(results):
            page_num = pages[idx]["page_number"]
            if isinstance(res, Exception):
                page_res = {
                    "page_number": page_num,
                    "success": False,
                    "error": str(res),
                    "header": {"passages": []},
                    "body": {"passages": []},
                    "annex": {"passages": []},
                }
            else:
                page_res = {
                    "page_number": page_num,
                    "success": res.get("success", False),
                    "error": res.get("error"),
                    "header": res.get("data", {}).get("header", {"passages": []}),
                    "body": res.get("data", {}).get("body", {"passages": []}),
                    "annex": res.get("data", {}).get("annex", {"passages": []}),
                }
                # ---- counting stats (only on success) ----
                if res.get("success"):
                    for sec in ("header", "body", "annex"):
                        passages = page_res[sec].get("passages", [])
                        out["summary"]["structure_counts"][f"{sec}_passages"] += len(passages)
                        out["summary"]["total_passages"] += len(passages)
                        for p in passages:
                            out["summary"]["total_references"] += len(p.get("references", []))

            out["pages"].append(page_res)

        return out


# ----------------------------------------------------------------------
# Orchestrator – runs on a cheap CPU worker, calls the GPU class
# ----------------------------------------------------------------------
@app.function(
    image=image,
    cpu=4,
    memory=8192,
    timeout=1200,
    volumes={"/root/.cache/huggingface": HF_CACHE_VOL},
    secrets=[modal.Secret.from_name("document-qa-api-key")],
)
async def process_pdf_document_structure(pdf: UploadFile, dpi: int = 150) -> Dict:
    """
    Entry point used by the HTTP wrapper.
    Returns the full document-structure JSON.
    """
    contents = await pdf.read()
    pages = extract_pdf_pages.remote(contents, dpi)

    if not pages:
        raise HTTPException(status_code=400, detail="Could not extract pages from PDF")

    # instantiate the GPU class locally in the same container
    model = VLMMModel()
    await model.start_vllm_client.remote()          # warm-up
    structure = await model.extract_document_structure.remote(pages)

    structure["document_info"]["filename"] = pdf.filename
    structure["document_info"]["dpi"] = dpi
    structure["success"] = True
    return structure


# ----------------------------------------------------------------------
# FastAPI ASGI app (exposed as the HTTP endpoint)
# ----------------------------------------------------------------------
web_app = FastAPI(
    title="Qwen-VL PDF Structure Extraction",
    version="1.0",
)


@web_app.post("/process-pdf")
async def http_process_pdf(pdf: UploadFile = File(...), dpi: int = 150):
    """
    Upload a PDF → receive the structured JSON.
    """
    # Modal's remote call is *awaitable* because the function is async
    result = await process_pdf_document_structure.remote(pdf, dpi)
    return result

@modal.asgi_app()
def fastapi_app():
    return web_app


# ----------------------------------------------------------------------
# (Optional) Simple image-QA endpoint you had earlier
# ----------------------------------------------------------------------
@VLMMModel.fastapi_endpoint(method="POST", path="/generate")
async def generate(model: VLMMModel, question: str, image: UploadFile = File(...)):
    """
    Very small wrapper that re-uses the same VLMMModel class.
    """
    img_bytes = await image.read()
    b64 = base64.b64encode(img_bytes).decode()
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": question},
                {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64}"}},
            ],
        }
    ]
    resp = await model.client.chat.completions.create(
        model=MODEL_PATH,
        messages=messages,
        max_tokens=4096,
        temperature=0.1,
    )
    return {"answer": resp.choices[0].message.content}


# ----------------------------------------------------------------------
# Run locally (for testing)
# ----------------------------------------------------------------------
if __name__ == "__main__":
    # modal run inference_qwen_VL_7B_vllm.py
    import uvicorn

    uvicorn.run(web_app, host="0.0.0.0", port=8000)