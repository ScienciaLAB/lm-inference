"""
Baguette-Software-Dataset (608M) served on Modal (vLLM, L4).

Two-step extraction of dataset and software mentions from a paper, following
the authors' inference_example.py:

  step 1 (validated):  paragraph -> dataset / software mentions
  step 2 (indicative): mentions  -> article-level record + PLOS-OSI summary

The weights are read from the public Hugging Face bucket
dataesr/Baguette-Software-Dataset, not from the model repository named in the
authors' example (PleIAs/Baguette-Software-Dataset), which is not public.
They are downloaded once into a Modal volume.

The tokenizer ships no chat template, so the endpoint calls the raw
completions API with the exact training templates.

Deploy (from lm-inference/):
    modal deploy deployments/baguette/inference_baguette.py
Smoke test (from lm-inference/):
    modal run deployments/baguette/inference_baguette.py
"""

import json
import os
import subprocess
import time
from concurrent.futures import ThreadPoolExecutor

import modal
from fastapi import HTTPException, Request

from lm_inference_utils import get_cost_per_second

APP_NAME = "baguette-software-dataset-app"
BUCKET_URL = (
    "https://huggingface.co/buckets/dataesr/Baguette-Software-Dataset/resolve"
)
MODEL_FILES = [
    "config.json",
    "special_tokens_map.json",
    "tokenizer.json",
    "tokenizer_config.json",
    "model.safetensors",
]
GPU_TYPE = "L4"
MAX_MODEL_LEN = 8192
MAX_TOKENS = 1024
STOP = ["<|im_end|>"]
# Requests sent to vLLM at the same time; vLLM batches them on the GPU.
MAX_PARALLEL_REQUESTS = 32

MODEL_VOLUME = modal.Volume.from_name("baguette-model", create_if_missing=True)
MODEL_VOLUME_PATH = "/models"
MODEL_PATH = f"{MODEL_VOLUME_PATH}/Baguette-Software-Dataset"

image = (
    modal.Image.from_registry("nvidia/cuda:12.4.1-devel-ubuntu22.04", add_python="3.11")
    .apt_install("git", "wget")
    .pip_install(
        "vllm==0.11.0",
        extra_options="--extra-index-url https://download.pytorch.org/whl/cu128 --no-cache-dir",
    )
    .pip_install("httpx", "fastapi[standard]", extra_options="--no-cache-dir")
    .add_local_file("lm_inference_utils.py", "/root/lm_inference_utils.py")
)

app = modal.App(APP_NAME, image=image)


# Exact training templates, from the model card.
def extraction_prompt(paragraph: str) -> str:
    return f"<|im_start|>user\n<text>{paragraph}</text><|im_end|>\n<|im_start|>assistant\n"


def analysis_prompt(mentions: str) -> str:
    return (
        f"<|im_start|>user\n<mentions>{mentions}</mentions><|im_end|>\n"
        "<|im_start|>assistant\n"
    )


def parse_json(text: str) -> dict:
    """The model is not constrained-decoded: keep the outermost JSON object."""
    text = text.strip()
    start, end = text.find("{"), text.rfind("}")
    try:
        parsed = json.loads(text[start : end + 1])
    except Exception:
        return {}
    return parsed if isinstance(parsed, dict) else {}


def aggregate_mentions(per_paragraph: list[dict]) -> dict:
    """Merge the per-paragraph mentions, deduplicated by lower-cased name."""
    datasets, software, seen = [], [], set()
    for extraction in per_paragraph:
        for kind, target in (("datasets", datasets), ("software", software)):
            for mention in extraction.get(kind) or []:
                if not isinstance(mention, dict):
                    continue
                name = (mention.get("name") or "").lower()
                if name and (kind, name) not in seen:
                    seen.add((kind, name))
                    target.append(mention)
    return {"datasets": datasets, "software": software}


def download_model():
    """Fetch the model files from the bucket into the volume, once."""
    import httpx

    os.makedirs(MODEL_PATH, exist_ok=True)
    downloaded = False
    for name in MODEL_FILES:
        target = f"{MODEL_PATH}/{name}"
        if os.path.exists(target):
            continue
        print(f"Downloading {name} ...")
        partial = f"{target}.part"
        with httpx.stream(
            "GET", f"{BUCKET_URL}/{name}", follow_redirects=True, timeout=600
        ) as response:
            response.raise_for_status()
            with open(partial, "wb") as f:
                for chunk in response.iter_bytes(chunk_size=1 << 20):
                    f.write(chunk)
        os.replace(partial, target)
        downloaded = True
    if downloaded:
        MODEL_VOLUME.commit()


@app.cls(
    gpu=GPU_TYPE,
    scaledown_window=300,
    max_containers=4,
    timeout=1800,
    volumes={MODEL_VOLUME_PATH: MODEL_VOLUME},
)
# One article per container, as in the other deployments, so the cost of an
# article is not divided by a sharing factor.
@modal.concurrent(max_inputs=1)
class BaguetteService:
    @modal.enter()
    def start_server(self):
        import httpx

        download_model()
        print(f"Starting vLLM server for {MODEL_PATH} ...")
        self.server_process = subprocess.Popen(
            [
                "vllm", "serve", MODEL_PATH,
                "--host", "0.0.0.0", "--port", "8000",
                "--served-model-name", "model",
                "--dtype", "bfloat16",
                "--gpu-memory-utilization", "0.90",
                "--max-model-len", str(MAX_MODEL_LEN),
            ]
        )  # fmt: skip

        for i in range(300):
            if self.server_process.poll() is not None:
                raise RuntimeError("vLLM exited during startup")
            try:
                if httpx.get("http://localhost:8000/health", timeout=2).status_code == 200:
                    print("vLLM server ready.")
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

    def _complete(self, client, prompt: str) -> dict:
        """Greedy completion of one prompt. Never raises: a failure, such as a
        prompt longer than the context, is reported for that prompt only."""
        payload = {
            "model": "model",
            "prompt": prompt,
            "temperature": 0.0,
            "max_tokens": MAX_TOKENS,
            "stop": STOP,
        }
        try:
            r = client.post("http://localhost:8000/v1/completions", json=payload)
            if r.status_code != 200:
                return {"error": r.text[:500]}
            return {"text": r.json()["choices"][0]["text"]}
        except Exception as e:
            return {"error": str(e)}

    @modal.method()
    def extract(self, paragraphs: list[str], analyze: bool = True):
        import httpx

        with httpx.Client(timeout=600) as client:
            # Step 1: one extraction per paragraph.
            prompts = [extraction_prompt(p) for p in paragraphs]
            with ThreadPoolExecutor(max_workers=MAX_PARALLEL_REQUESTS) as pool:
                outputs = list(pool.map(lambda p: self._complete(client, p), prompts))

            per_paragraph = []
            for index, output in enumerate(outputs):
                extraction = parse_json(output.get("text", ""))
                item = {
                    "index": index,
                    "is_boilerplate": extraction.get("is_boilerplate"),
                    "datasets": extraction.get("datasets") or [],
                    "software": extraction.get("software") or [],
                }
                if "error" in output:
                    item["error"] = output["error"]
                elif not extraction:
                    item["error"] = "output is not a JSON object"
                    item["raw_output"] = output["text"]
                per_paragraph.append(item)

            mentions = aggregate_mentions(per_paragraph)
            result = {
                "num_paragraphs": len(paragraphs),
                "paragraphs": per_paragraph,
                "mentions": mentions,
            }

            # Step 2: article-level record, only when there is something to analyse.
            if analyze and (mentions["datasets"] or mentions["software"]):
                output = self._complete(
                    client, analysis_prompt(json.dumps(mentions, ensure_ascii=False))
                )
                result["record"] = parse_json(output.get("text", ""))
                if "error" in output:
                    result["record_error"] = output["error"]
                elif not result["record"]:
                    result["record_error"] = "output is not a JSON object"
                    result["record_raw_output"] = output["text"]

        return result


shared_service = BaguetteService()


@app.function(cpu=2, memory=4096, timeout=1800, max_containers=1)
@modal.concurrent(max_inputs=10)
@modal.fastapi_endpoint(method="POST")
async def extract_endpoint(request: Request):
    """Body: {"paragraphs": ["...", "..."], "analyze": true}"""
    try:
        body = await request.json()
    except Exception as e:
        raise HTTPException(status_code=400, detail="JSON body required") from e
    paragraphs = body.get("paragraphs") if isinstance(body, dict) else None
    if (
        not isinstance(paragraphs, list)
        or not paragraphs
        or not all(isinstance(p, str) for p in paragraphs)
    ):
        raise HTTPException(
            status_code=400, detail="'paragraphs' must be a non-empty list of strings"
        )

    t0 = time.time()
    try:
        result = await shared_service.extract.remote.aio(
            paragraphs, bool(body.get("analyze", True))
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)) from e
    duration = time.time() - t0
    result["duration_seconds"] = round(duration, 2)
    result["cost_usd"] = round(duration * get_cost_per_second(GPU_TYPE), 6)
    return result


@app.local_entrypoint()
def main():
    """Smoke test on the paragraphs of the authors' example."""
    paragraphs = [
        "For the data acquisition, a T420 FLIR thermal camera with a 0.1 degree C "
        "thermal sensitivity was used. The data was acquired using the ResearchIR "
        "MAX 4.0 software by connecting the camera to a computer.",
        "Results and discussion. Figure 9 shows the infrared images obtained on the "
        "slabs containing either sound or corroded sensors.",
        "As can be seen, the intensity of the hot spot is higher for the sound "
        "sensor than the corroded one.",
    ]
    result = BaguetteService().extract.remote(paragraphs)
    print(json.dumps(result, ensure_ascii=False, indent=1))
