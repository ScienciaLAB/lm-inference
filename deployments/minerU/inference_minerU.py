import modal
import tempfile
import subprocess
import json
from pathlib import Path
import time
from fastapi import HTTPException, Request

from lm_inference_utils import get_cost_per_second

# MinerU 2.5 VLM model served through an OpenAI-compatible vLLM server.
# https://github.com/opendatalab/MinerU  -  https://huggingface.co/opendatalab/MinerU2.5-2509-1.2B
MODEL_NAME = "opendatalab/MinerU2.5-2509-1.2B"
VLLM_PORT = 30000

# Persist the downloaded model weights across container starts.
HF_CACHE = modal.Volume.from_name("mineru-hf-cache", create_if_missing=True)
HF_CACHE_PATH = "/root/.cache/huggingface"

# Modal image with MinerU 2.5 and its vLLM backend.
image = (
    modal.Image.from_registry("nvidia/cuda:12.4.1-devel-ubuntu22.04", add_python="3.11")
    .apt_install("git", "wget", "libgl1", "libglib2.0-0")
    # Pin vLLM to MinerU's recommended 0.10.1.1. The auto-resolved 0.21.x breaks
    # the OpenAI server: prometheus_fastapi_instrumentator crashes on every route
    # ("'_IncludedRouter' object has no attribute 'path'"), so /health and the
    # inference endpoints all return 500 and the container never becomes ready.
    .pip_install(
        "mineru[core,vllm]",
        "vllm==0.10.1.1",
        extra_options="--no-cache-dir",
    )
    .env(
        {
            # Download weights from HuggingFace (default is modelscope).
            "MINERU_MODEL_SOURCE": "huggingface",
            "HF_HOME": HF_CACHE_PATH,
        }
    )
    # Patch a prometheus_fastapi_instrumentator bug that makes MinerU's vLLM
    # OpenAI server 500 on EVERY request (incl. /health and inference): its
    # route-name lookup does `route.path` on routes that lack it (`_IncludedRouter`),
    # raising AttributeError. Guard the attribute access so the middleware no
    # longer crashes. (vLLM version is unrelated — 0.10.1.1 and 0.21 both hit this.)
    .run_commands(
        "P=$(python -c 'import os,prometheus_fastapi_instrumentator as m;print(os.path.join(os.path.dirname(m.__file__),\"routing.py\"))') && "
        'sed -i \'s/route_name = route\\.path/route_name = getattr(route, "path", "")/\' "$P" && '
        "grep -n 'getattr(route' \"$P\""
    )
    .run_commands(
        "mineru --help",
        "python -c 'import vllm; print(\"vLLM:\", vllm.__version__)'",
    )
    .add_local_file("lm_inference_utils.py", "/root/lm_inference_utils.py")
)


app = modal.App("mineru-vllm-official-app", image=image)


# Modal class to manage the MinerU vLLM server lifecycle.
@app.cls(
    gpu="A100-40GB",
    scaledown_window=300,
    max_containers=4,
    min_containers=1,
    volumes={HF_CACHE_PATH: HF_CACHE},
)
# One document per container: N concurrent client requests then fan out to N
# separate GPU containers (up to max_containers) instead of piling onto one.
# With max_inputs>1 a single container absorbs all the concurrency, so the pool
# never scales and requests contend/get cancelled ("Missing request").
@modal.concurrent(max_inputs=1)
class MinerUService:
    @modal.enter()  # Runs once when the container starts
    def start_server(self):
        print("Starting MinerU vLLM server...")

        # mineru-vllm-server wraps `vllm serve` with the MinerU2.5 defaults and
        # accepts vLLM passthrough flags. It exposes an OpenAI-compatible API.
        # `--model` pins the VLM to MinerU2.5-2509 (Sept 2025); without it the
        # server defaults to the newer MinerU2.5-Pro-2605 (2026), which would
        # break temporal parity with our Sept-2025 Docling comparison.
        self.server_process = subprocess.Popen(
            [
                "mineru-vllm-server",
                "--model",
                MODEL_NAME,
                "--host",
                "0.0.0.0",
                "--port",
                str(VLLM_PORT),
                "--gpu-memory-utilization",
                "0.90",
                "--tensor-parallel-size",
                "1",  # 1 GPU
            ]
        )

        print("Waiting for MinerU vLLM server to start...")

        import httpx

        # First boot downloads the weights, so allow a generous timeout.
        for i in range(600):
            try:
                resp = httpx.get(f"http://localhost:{VLLM_PORT}/health", timeout=2)
                if resp.status_code == 200:
                    print("✅ MinerU vLLM server is ready.")
                    return
            except Exception:
                pass
            time.sleep(1)
            if i % 30 == 0:
                print(f"... waiting ({i}s elapsed)")

        raise RuntimeError("MinerU vLLM server failed to start in 600s")

    @modal.exit()  # Runs when the container shuts down
    def stop_server(self):
        if hasattr(self, "server_process") and self.server_process:
            print("Stopping MinerU vLLM server...")
            self.server_process.terminate()
            try:
                self.server_process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                print("Server did not shut down gracefully, killing...")
                self.server_process.kill()  # Force kill if it doesn't terminate
            print("MinerU vLLM server stopped.")

    @modal.method()
    def parse_document(
        self,
        file_bytes: bytes,
        original_filename: str,
        lang: str = "ch",
    ):
        import shutil

        suffix = Path(original_filename).suffix.lower()
        if suffix not in {".pdf", ".jpg", ".jpeg", ".png"}:
            suffix = ".pdf"

        work_dir = tempfile.mkdtemp(prefix="mineru_")
        input_path = Path(work_dir) / f"input{suffix}"
        input_path.write_bytes(file_bytes)
        output_dir = Path(work_dir) / "output"
        output_dir.mkdir()

        print(f"Processing file: {input_path}")

        # The client talks to the warm vLLM server over HTTP (vlm-http-client).
        cmd = [
            "mineru",
            "-p",
            str(input_path),
            "-o",
            str(output_dir),
            "-b",
            "vlm-http-client",
            "-u",
            f"http://localhost:{VLLM_PORT}",
            "--lang",
            lang,
        ]

        try:
            result = subprocess.run(
                cmd,
                check=True,
                capture_output=True,
                text=True,
                timeout=900,
            )
            print("MinerU STDOUT:", result.stdout)
            print("MinerU STDERR:", result.stderr)

            # MinerU writes to <output_dir>/<stem>/vlm/ (markdown, content_list, ...).
            outputs = {}

            md_files = sorted(output_dir.glob("**/*.md"))
            if md_files:
                outputs["markdown_content"] = md_files[0].read_text(encoding="utf-8")

            content_list = sorted(output_dir.glob("**/*_content_list.json"))
            if content_list:
                outputs["json_content"] = json.loads(
                    content_list[0].read_text(encoding="utf-8")
                )

            middle = sorted(output_dir.glob("**/*_middle.json"))
            if middle:
                outputs["middle_json"] = json.loads(
                    middle[0].read_text(encoding="utf-8")
                )

            if not outputs:
                raise RuntimeError(
                    "MinerU produced no output files. "
                    f"Output tree: {[str(p) for p in output_dir.glob('**/*')]}"
                )

            print("Successfully parsed document. Output keys:", list(outputs.keys()))
            return outputs

        except subprocess.CalledProcessError as e:
            print("MinerU failed!")
            print("STDOUT:", e.stdout)
            print("STDERR:", e.stderr)
            raise RuntimeError(f"MinerU error: {e.stderr}")
        except Exception as e:
            print(f"Unexpected error in parse_document: {e}")
            raise
        finally:
            shutil.rmtree(work_dir, ignore_errors=True)


shared_service = MinerUService()


@app.function(cpu=2, memory=4096, timeout=1000, max_containers=1)
@modal.concurrent(max_inputs=10)
@modal.fastapi_endpoint(method="POST")
async def parse_document_endpoint(request: Request):
    from starlette.datastructures import UploadFile as StarletteUploadFile

    try:
        form = await request.form()
        if "file" not in form or not isinstance(form["file"], StarletteUploadFile):
            raise HTTPException(
                status_code=400, detail="No 'file' part in the request or invalid file."
            )
        file: StarletteUploadFile = form["file"]
        file_content = await file.read()
        original_filename = file.filename
        lang = form.get("lang", "ch")
        output_format = form.get("output_format", "markdown_content")

        parse_kwargs = {
            "file_bytes": file_content,
            "original_filename": original_filename,
            "lang": lang,
        }

        start_time = time.perf_counter()

        results_future = shared_service.parse_document.remote(**parse_kwargs)
        results = results_future.get(output_format)

        duration = time.perf_counter() - start_time
        cost_per_sec = get_cost_per_second("A100_40GB")
        NUM_GPUS_USED = 1

        total_cost = duration * cost_per_sec * NUM_GPUS_USED
        cost_result = {
            "duration_seconds": round(duration, 2),
            "cost_usd": round(total_cost, 6),
        }

        print(
            f"[COST_LOG] File: {original_filename}, Duration: {cost_result['duration_seconds']}s, Cost: ${cost_result['cost_usd']:.6f}"
        )

        return {"result": results, "cost_info": cost_result}

    except Exception as e:
        print(f"Error in web endpoint: {e}")
        raise HTTPException(
            status_code=500, detail=f"Error processing document: {str(e)}"
        )
