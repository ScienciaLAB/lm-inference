# olmocr_v2_modal.py
import modal
import subprocess
import tempfile
import os
import time
from pathlib import Path
from fastapi import FastAPI, File, Request, UploadFile, HTTPException
from fastapi.responses import PlainTextResponse
from starlette.datastructures import UploadFile as StarletteUploadFile

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
    .pip_install("olmocr[gpu]", extra_index_url="https://download.pytorch.org/whl/cu128")
)

app = modal.App("olmocr-v2-inference")

@app.cls(
    gpu="A100-40GB",
    image=image,
    volumes={MODEL_CACHE: VOLUME},
    timeout=1200,
    scaledown_window=300,
    min_containers=1
)
class OlmOcrService:
    @modal.enter()
    def start_vllm(self):
        print("🚀 Starting vLLM server for olmOCR v2")
        os.environ["HF_HOME"] = MODEL_CACHE
        self.vllm_proc = subprocess.Popen([
            "vllm", "serve",
            "allenai/olmOCR-2-7B-1025-FP8",
            "--served-model-name", "olmocr",
            "--max-model-len", "16384",
            "--gpu-memory-utilization", "0.90",
            "--port", "8000",
            "--host", "0.0.0.0",
             "--dtype", "auto",
        ])

        import httpx, time
        for i in range(300): 
            try:
                if httpx.get("http://localhost:8000/health", timeout=2).status_code == 200:
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
        import tempfile
        from pathlib import Path
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
                sys.executable, "-m", "olmocr.pipeline",
                str(output_dir),
                "--markdown",
                "--pdfs", str(pdf_path),
                "--server", "http://localhost:8000/v1",
                "--model", "olmocr",
                "--gpu-memory-utilization", "0.85",
                "--max_model_len", "16384",
            ]

            result = subprocess.run(cmd, capture_output=True, text=True)
            if result.returncode != 0:
                raise RuntimeError(f"olmOCR failed:\nSTDOUT:\n{result.stdout}\nSTDERR:\n{result.stderr}")

            # Locate the .jsonl result file
            results_dir = output_dir / "results"
            jsonl_files = list(results_dir.glob("*.jsonl"))
            if not jsonl_files:
                raise FileNotFoundError("No .jsonl result file found in 'results' directory")

            jsonl_file = jsonl_files[0]
            with open(jsonl_file, 'r', encoding='utf-8') as f:
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
            #print(markdown_text)
            return markdown_text

# Shared instance
service = OlmOcrService()

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import PlainTextResponse

@app.function(
    gpu="A100-40GB",
    image=image,
    volumes={MODEL_CACHE: VOLUME},
    timeout=1800,  
)
@modal.fastapi_endpoint(method="POST")
async def upload(request: Request):
    try:
        form = await request.form()
        file = form.get("file")
        if not isinstance(file, StarletteUploadFile):
            raise HTTPException(status_code=400, detail="No valid 'file' uploaded")

        filename = file.filename or "unknown.pdf"
        if not filename.lower().endswith(('.pdf', '.png', '.jpg', '.jpeg')):
            raise HTTPException(status_code=400, detail="Only PDF/PNG/JPG allowed")

        contents = await file.read()

        # Call your Modal class method
        markdown = await service.convert.remote.aio(contents, filename)

        return PlainTextResponse(markdown)

    except Exception as e:
        print(f"Error in upload endpoint: {e}")
        raise HTTPException(status_code=500, detail=str(e))
