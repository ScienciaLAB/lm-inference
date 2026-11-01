# olmocr_v2_modal.py
import modal
import subprocess
import tempfile
import os
import threading
import time
from pathlib import Path
from fastapi import Request, HTTPException
from starlette.datastructures import UploadFile as StarletteUploadFile

from lm_inference_utils import get_cost_per_second

VOLUME = modal.Volume.from_name("olmocr-v2-cache", create_if_missing=True)
MODEL_CACHE = "/model_cache"
# Build image
image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install(
        "poppler-utils",
        "fonts-crosextra-caladea",
        "fonts-crosextra-carlito",
        "gsfonts",
        "lcdf-typetools",
        "fontconfig",
        "fonts-liberation",
    )
    .run_commands("fc-cache -fv")
    .pip_install(
        "olmocr[gpu]",
        "fastapi",
        "uvicorn",
        extra_index_url="https://download.pytorch.org/whl/cu128",
    )
    .add_local_file("lm_inference_utils.py", "/root/lm_inference_utils.py")
)



# Documents served concurrently per GPU container. The campaign of the paper's
# cost table ran at 1 (one document per A100, pages batched only within the
# document). The re-measurement for the EACL response runs at 8, so that the
# vLLM server batches page requests across documents, which is what the
# vendors' throughput figures assume. Override at deploy time: DOCS_PER_GPU=<n>.
DOCS_PER_GPU = int(os.environ.get("DOCS_PER_GPU", "8"))

app = modal.App("olmocr-v2-inference")


@app.cls(
    gpu="A100-40GB",
    image=image,
    volumes={MODEL_CACHE: VOLUME},
    timeout=1200,
    scaledown_window=300,
    min_containers=1,
    max_containers=1,
)
# One GPU container with DOCS_PER_GPU documents in flight: the olmocr pipeline
# processes run side by side and the single vLLM server batches their page
# requests. With max_containers=1 the billed GPU time is this container's span
# (see [GPU_SPAN] in the logs). The paper's campaign used max_inputs=1 and
# max_containers=4, one document per GPU.
@modal.concurrent(max_inputs=DOCS_PER_GPU)
class OlmOcrService:
    @modal.enter()
    def start_vllm(self):
        print("🚀 Starting vLLM server for olmOCR v2")
        # GPU-time accounting for the concurrency re-measurement: with one
        # container (max_containers=1) the billed GPU time is this container's
        # span, so the span from the first document's start to the last
        # document's end, divided by the documents served, is the GPU-seconds
        # per document that the per-request latency no longer measures.
        self._t0 = time.time()
        self._ready = None
        self._docs = 0
        self._first = None
        self._last = None
        self._lock = threading.Lock()
        os.environ["HF_HOME"] = MODEL_CACHE
        self.vllm_proc = subprocess.Popen(
            [
                "vllm",
                "serve",
                "allenai/olmOCR-2-7B-1025-FP8",
                "--served-model-name",
                "olmocr",
                "--max-model-len",
                "16384",
                "--gpu-memory-utilization",
                "0.90",
                "--port",
                "8000",
                "--host",
                "0.0.0.0",
                "--dtype",
                "auto",
            ]
        )

        import httpx

        for i in range(300):
            try:
                if (
                    httpx.get("http://localhost:8000/health", timeout=2).status_code
                    == 200
                ):
                    self._ready = time.time()
                    print("✅ vLLM server ready!")
                    return
            except Exception:
                pass
            time.sleep(1)
            if i % 30 == 0:
                print(f"... waiting ({i}s elapsed)")
        raise RuntimeError("vLLM failed to start — check logs for model load errors")

    @modal.method()
    def convert(self, pdf_bytes: bytes, filename: str) -> str:
        t_start = time.time()
        try:
            return self._convert(pdf_bytes, filename)
        finally:
            with self._lock:
                self._docs += 1
                self._first = self._first or t_start
                self._last = time.time()
                span = self._last - self._first
                startup = (self._ready or self._t0) - self._t0
                print(
                    f"[GPU_SPAN] docs={self._docs} active_span_s={span:.1f} "
                    f"startup_s={startup:.1f} gpu_s_per_doc={span / self._docs:.2f} "
                    f"docs_per_gpu={DOCS_PER_GPU}"
                )

    def _convert(self, pdf_bytes: bytes, filename: str) -> str:
        import subprocess
        import sys
        import json

        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)
            output_dir = tmpdir / "output"
            output_dir.mkdir()

            pdf_path = tmpdir / "input.pdf"
            pdf_path.write_bytes(pdf_bytes)

            cmd = [
                sys.executable,
                "-m",
                "olmocr.pipeline",
                str(output_dir),
                "--markdown",
                "--pdfs",
                str(pdf_path),
                "--server",
                "http://localhost:8000/v1",
                "--model",
                "olmocr",
                "--gpu-memory-utilization",
                "0.85",
                "--max_model_len",
                "16384",
            ]

            result = subprocess.run(cmd, capture_output=True, text=True)
            if result.returncode != 0:
                raise RuntimeError(
                    f"olmOCR failed:\nSTDOUT:\n{result.stdout}\nSTDERR:\n{result.stderr}"
                )

            # Locate the .jsonl result file
            results_dir = output_dir / "results"
            jsonl_files = list(results_dir.glob("*.jsonl"))
            if not jsonl_files:
                raise FileNotFoundError(
                    "No .jsonl result file found in 'results' directory"
                )

            jsonl_file = jsonl_files[0]
            with open(jsonl_file, "r", encoding="utf-8") as f:
                first_line = f.readline().strip()
                if not first_line:
                    raise ValueError("JSONL file is empty")

            try:
                record = json.loads(first_line)
            except json.JSONDecodeError as e:
                raise ValueError(f"Failed to parse JSONL: {e}")

            if "text" not in record:
                raise KeyError("Expected 'text' field not found in JSONL output")

            markdown_text = record["text"]
            if not markdown_text.strip():
                raise ValueError("Extracted markdown text is empty")
            # print(markdown_text)
            return markdown_text


# Shared instance
service = OlmOcrService()


@app.function(
    image=image,
    cpu=2,
    memory=4096,
    max_containers=10,
    volumes={MODEL_CACHE: VOLUME},
    timeout=1800,
)
@modal.concurrent(max_inputs=64)
@modal.fastapi_endpoint(method="POST")
async def upload(request: Request):
    try:
        # Start timing immediately when request is received
        start_time = time.perf_counter()

        form = await request.form()
        file = form.get("file")
        if not isinstance(file, StarletteUploadFile):
            raise HTTPException(status_code=400, detail="No valid 'file' uploaded")

        filename = file.filename or "unknown.pdf"
        if not filename.lower().endswith((".pdf", ".png", ".jpg", ".jpeg")):
            raise HTTPException(status_code=400, detail="Only PDF/PNG/JPG allowed")

        contents = await file.read()

        # Call the olmOCR v2 Modal service
        results = await service.convert.remote.aio(contents, filename)

        # Stop timing after processing finishes
        duration = time.perf_counter() - start_time

        # Calculate cost
        GPU_TYPE = "A100_40GB"
        # The endpoint container is CPU-only; one GPU is used (the old value 2
        # double-counted it, see docs/run-logs/samesample/comparison.md). With
        # DOCS_PER_GPU documents sharing the GPU, latency x rate is no longer
        # the cost of a document; the nominal share below is kept for the client
        # CSVs, the billing basis is the container span ([GPU_SPAN]).
        NUM_GPUS_USED = 1
        cost_per_sec = get_cost_per_second(GPU_TYPE)
        total_cost = duration * cost_per_sec * NUM_GPUS_USED / DOCS_PER_GPU
        cost_info = {
            "duration_seconds": round(duration, 2),
            "cost_usd": round(total_cost, 6),
            "cost_basis": f"latency x A100-40GB rate / {DOCS_PER_GPU} docs per GPU (nominal)",
        }

        print(
            f"[COST_LOG] File: {filename}, Duration: {cost_info['duration_seconds']}s, Cost: ${cost_info['cost_usd']:.6f}"
        )

        # Return both the OCR result and cost info
        return {"result": results, "cost_info": cost_info}

    except Exception as e:
        print(f"Error in upload endpoint: {e}")
        raise HTTPException(status_code=500, detail=str(e))
