import modal
import tempfile
import os
import subprocess
import json
from pathlib import Path
import time
from fastapi import HTTPException, Request, File, UploadFile
# Modal GPU Pricing
GPU_COST_PER_SECOND = {
    "A10G": 0.000264,  
    "A10": 0.000306,
    "L4": 0.000222,
    "L40S": 0.000542,
    "A100_40GB": 0.000583,
    "A100_80GB": 0.000694,
    "H100": 0.001097,
    "H200": 0.001261,
    "B200": 0.001736,
    "T4": 0.000164,
}
# Modal image with necessary dependencies
image = (
    modal.Image.from_registry("nvidia/cuda:12.4.1-devel-ubuntu22.04", add_python="3.11") 
    .apt_install("git", "wget", "build-essential", "libopenmpi-dev") 
    .pip_install(
        ["torch==2.7.0", "torchvision==0.22.0", "torchaudio==2.7.0"],
        extra_options="--index-url https://download.pytorch.org/whl/cu128 --no-cache-dir",
    )
    .pip_install("packaging", "wheel", "numpy") 
    .pip_install(
        "vllm==0.11.0",
        extra_options="--extra-index-url https://download.pytorch.org/whl/cu128 --no-cache-dir",
    )
    # other dependencies required by dots.ocr parser script
    .pip_install(
        "qwen-vl-utils", 
        "pillow", "requests", "pydantic", "pyyaml", "pymupdf", # Common dependencies
        "transformers",
        extra_options="--no-cache-dir",
    )
    # Clone the dots.ocr repository
    .run_commands(
        "git clone https://github.com/rednote-hilab/dots.ocr.git /dots_ocr_repo",

    ).run_commands("cd /dots_ocr_repo && pip install --no-deps -e .")
    # Download model
    .run_commands("cd /dots_ocr_repo && python tools/download_model.py")
    .run_commands(
        "python -c 'import dots_ocr; print(\"dots_ocr imported\")'",
        "python -c 'import vllm; print(\"vLLM:\", vllm.__version__)'"
    )
)

app = modal.App("dots-ocr-vllm-official-app", image=image)
# cost calculation 
def get_cost_per_second(gpu_type: str) -> float:
    """Retrieves the cost per second for a given GPU type."""
    if gpu_type not in GPU_COST_PER_SECOND:
        available = ", ".join(GPU_COST_PER_SECOND.keys())
        raise ValueError(f"Unknown GPU type '{gpu_type}'. Available types: {available}")
    return GPU_COST_PER_SECOND[gpu_type]

# Modal class to manage the vLLM server lifecycle
@app.cls(gpu="A10G",scaledown_window=300,max_containers=1,min_containers=1)
class DotsOCRService:
    @modal.enter()  # Runs once when the container starts
    def start_server(self):
        print("Starting vLLM server for dots.ocr...")

        model_path = "/dots_ocr_repo/weights/DotsOCR"  

        if not os.path.exists(model_path):
            raise RuntimeError(f"Model path not found: {model_path}. "
                            "Check that download_model.py saved the model correctly.")

        # Launch vLLM with the correct model directory
        self.server_process = subprocess.Popen([
            "vllm", "serve",
            model_path,
            "--trust-remote-code",
            "--gpu-memory-utilization", "0.95",
            "--host", "0.0.0.0",
            "--port", "8000",
            "--served-model-name", "model" 
        ])

        print("Waiting for vLLM server to start...")

        import httpx
        for i in range(180):
            try:
                resp = httpx.get("http://localhost:8000/health", timeout=2)
                if resp.status_code == 200:
                    print("✅ vLLM server is ready.")
                    return
            except Exception:
                pass
            time.sleep(1)
            if i % 30 == 0:
                print(f"... waiting ({i}s elapsed)")

        raise RuntimeError("vLLM failed to start in 180s")

    @modal.exit() # Runs when the container shuts down
    def stop_server(self):
        if hasattr(self, 'server_process') and self.server_process:
            print("Stopping vLLM server...")
            self.server_process.terminate()
            try:
                self.server_process.wait(timeout=10) 
            except subprocess.TimeoutExpired:
                print("Server did not shut down gracefully, killing...")
                self.server_process.kill() # Force kill if it doesn't terminate
            print("vLLM server stopped.")

    @modal.method()
    def parse_document(self, file_bytes: bytes, original_filename: str, prompt_mode: str = "prompt_layout_all_en", num_threads: int = 64):
        import json
        import shutil
        from pathlib import Path

        os.chdir("/dots_ocr_repo")
        os.environ["VLLM_ENDPOINT"] = "http://localhost:8000"
        os.environ["VLLM_MODEL_NAME"] = "model"

        suffix = Path(original_filename).suffix.lower()
        if suffix not in {'.pdf', '.jpg', '.jpeg', '.png'}:
            suffix = '.pdf'

        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp_file:
            tmp_file.write(file_bytes)
            local_input_path = tmp_file.name

        print(f"Processing file: {local_input_path} with prompt: {prompt_mode}")

        cmd = [
            "python3", "dots_ocr/parser.py", local_input_path,
            "--prompt", prompt_mode,
            "--num_thread", str(num_threads)
        ]

        try:
            result = subprocess.run(
                cmd,
                cwd="/dots_ocr_repo",
                check=True,
                capture_output=True,
                text=True,
                timeout=900
            )
            print("Parser STDOUT:", result.stdout)
            print("Parser STDERR:", result.stderr)

            base_name = Path(local_input_path).stem
            output_subdir = Path("/dots_ocr_repo/output") / base_name
            outputs = {}

            if output_subdir.exists():
                # Merge all page JSONs into one list
                page_jsons = []
                for page_file in sorted(output_subdir.glob(f"{base_name}_page_*.json")):
                    try:
                        with open(page_file, 'r', encoding='utf-8') as f:
                            data = json.load(f)
                            if isinstance(data, list):
                                page_jsons.extend(data)
                            else:
                                print(f" Unexpected JSON structure in {page_file}")
                    except Exception as e:
                        print(f"Failed to read JSON {page_file}: {e}")
                if page_jsons:
                    outputs['json_content'] = page_jsons

                # Concatenate Markdown files
                def read_and_concat(pattern_suffix):
                    parts = []
                    for md_file in sorted(output_subdir.glob(f"{base_name}_page_*{pattern_suffix}")):
                        try:
                            with open(md_file, 'r', encoding='utf-8') as f:
                                parts.append(f.read())
                        except Exception as e:
                            print(f"Failed to read Markdown {md_file}: {e}")
                    return "\n\n--- PAGE BREAK ---\n\n".join(parts) if parts else None

                md = read_and_concat(".md")
                if md:
                    outputs['markdown_content'] = md

                md_nohf = read_and_concat("_nohf.md")
                if md_nohf:
                    outputs['markdown_nohf_content'] = md_nohf

            #Cleanup 
            os.unlink(local_input_path)
            if output_subdir.exists():
                shutil.rmtree(output_subdir)

            print("Successfully parsed document. Output keys:", list(outputs.keys()))
            return outputs

        except subprocess.CalledProcessError as e:
            print("Parser failed!")
            print("STDOUT:", e.stdout)
            print("STDERR:", e.stderr)
            raise RuntimeError(f"Parser error: {e.stderr}")
        except Exception as e:
            print(f"Unexpected error in parse_document: {e}")
            raise

shared_service = DotsOCRService()

    
@app.function(gpu="A10G", timeout=1000 ,scaledown_window=300,max_containers=1)
@modal.fastapi_endpoint(method="POST")
async def parse_document_endpoint(request: Request):
    from starlette.datastructures import UploadFile as StarletteUploadFile
    try:
        form = await request.form()
        if 'file' not in form or not isinstance(form['file'], StarletteUploadFile):
            raise HTTPException(status_code=400, detail="No 'file' part in the request or invalid file.")
        file: StarletteUploadFile = form['file']
        file_content = await file.read()
        original_filename = file.filename
        prompt_mode = form.get('prompt_mode', 'prompt_layout_all_en')
        num_threads = int(form.get('num_threads', '64'))
        output_format=form.get('output_format', 'json_content')

        parse_kwargs = {
            "file_bytes": file_content,
            "original_filename": original_filename,
            "prompt_mode": prompt_mode,
            "num_threads": num_threads
        }

        start_time = time.perf_counter()

        # Make the remote call and wait for the result (this is where processing happens)
        results_future = shared_service.parse_document.remote(**parse_kwargs)
        results = results_future.get(output_format) # This line blocks until processing is done

        # Calculate duration and cost after processing is complete
        duration = time.perf_counter() - start_time
        cost_per_sec = get_cost_per_second("A10G") # Use your specific GPU type
        NUM_GPUS_USED = 2  # 1 for the class, 1 for the endpoint

        total_cost = duration * cost_per_sec * NUM_GPUS_USED
        cost_result = {
            "duration_seconds": round(duration, 2),
            "cost_usd": round(total_cost, 6)
        }

        # Log the calculated cost
        print(f"[COST_LOG] File: {original_filename}, Duration: {cost_result['duration_seconds']}s, Cost: ${cost_result['cost_usd']:.6f}")

        # Optionally, return the cost info along with the results
        # This returns a dictionary containing both the parsed results and the cost info
        return {
            "result": results,
            "cost_info": cost_result
        }

    except ValueError:
        raise HTTPException(status_code=400, detail="num_threads must be an integer.")
    except Exception as e:
        print(f"Error in web endpoint: {e}")
        raise HTTPException(status_code=500, detail=f"Error processing document: {str(e)}")
