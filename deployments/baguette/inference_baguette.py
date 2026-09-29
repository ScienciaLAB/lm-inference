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

The endpoint takes a list of paragraphs, a text, or a file (text, TEI, JSON).
All the splitting is done here, on the server. The model only works on inputs
of the size of a paragraph: on a long input it finds nothing, or writes text
that is not JSON. So:
  - a text is split into paragraphs at blank lines, then into pieces of at
    most INPUT_TOKEN_BUDGET tokens, at sentence boundaries; each piece is one
    paragraph of the response;
  - a given paragraph (TEI, JSON) longer than the budget is split the same
    way, and the mentions of its chunks are merged back into one item;
  - a chunk whose output is not JSON, or is cut at MAX_TOKENS, is split in
    two and extracted again.

Deploy (from lm-inference/):
    modal deploy deployments/baguette/inference_baguette.py
Smoke test (from lm-inference/):
    modal run deployments/baguette/inference_baguette.py
Client:
    python deployments/baguette/baguette_client.py --endpoint https://<ws>--baguette-software-dataset-app-extract-endpoint.modal.run ...
"""

import json
import os
import re
import subprocess
import time
import xml.etree.ElementTree as ET
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
MAX_TOKENS = 2048
STOP = ["<|im_end|>"]
# Tokens of text sent in one prompt. The context allows far more, but the model
# does not: measured on CPU, it misses mentions from about 600 tokens, finds
# nothing from 800, and writes text that is not JSON from 2000. On one paper
# given as a single block, pieces of 200 tokens found the same software as its
# TEI paragraphs (177 tokens on average); pieces of 300 and 400 found less.
INPUT_TOKEN_BUDGET = 200
# A failed chunk is split in two and extracted again, this many times at most,
# and only while it has more tokens than MIN_RETRY_TOKENS.
MAX_RETRY_DEPTH = 3
MIN_RETRY_TOKENS = 40
SENTENCE_END = re.compile(r"[.!?;:]\s+")
BLANK_LINES = re.compile(r"\n[ \t\r]*\n")
TEI = "{http://www.tei-c.org/ns/1.0}"
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
        # vLLM 0.11.0 only requires transformers>=4.55.2, so pip takes 5.x, whose
        # tokenizers have no all_special_tokens_extended: vLLM exits at startup.
        "transformers==4.57.1",
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


def clean(text: str) -> str:
    return " ".join(text.split())


def split_text(text: str) -> list[str]:
    """Paragraphs of a plain text: the blocks separated by blank lines."""
    return [clean(block) for block in BLANK_LINES.split(text) if block.strip()]


def read_tei(content: bytes) -> list[str]:
    """Paragraphs of a TEI document (GROBID): the <p> elements of the abstract,
    body and back."""
    root = ET.fromstring(content)
    paragraphs = []
    for section in (f".//{TEI}abstract", f"./{TEI}text"):
        for parent in root.iterfind(section):
            for p in parent.iter(f"{TEI}p"):
                paragraphs.append(clean("".join(p.itertext())))
    return [p for p in paragraphs if p]


def count_tokens(text: str, tokenizer) -> int:
    return len(tokenizer.encode(text, add_special_tokens=False).ids)


def read_paragraphs_field(paragraphs) -> list[str]:
    if not isinstance(paragraphs, list) or not all(
        isinstance(p, str) for p in paragraphs
    ):
        raise ValueError("'paragraphs' must be a list of strings")
    return [clean(p) for p in paragraphs if p.strip()]


def read_json(data) -> tuple[list[str], bool]:
    """Paragraphs of a JSON input: a list of strings, or an object with a
    'paragraphs' list or a 'text'. The flag tells a text from paragraphs."""
    if isinstance(data, list):
        return read_paragraphs_field(data), False
    if isinstance(data, dict) and "paragraphs" in data:
        return read_paragraphs_field(data["paragraphs"]), False
    if isinstance(data, dict) and isinstance(data.get("text"), str):
        return split_text(data["text"]), True
    raise ValueError("expected 'paragraphs' (list of strings) or 'text' (string)")


def read_file(filename: str, content: bytes) -> tuple[list[str], bool]:
    """Paragraphs of an uploaded file, according to its extension. The flag
    tells a text from paragraphs."""
    suffix = os.path.splitext(filename)[1].lower()
    if suffix == ".xml":
        return read_tei(content), False
    text = content.decode("utf-8-sig", errors="replace")
    if suffix == ".json":
        return read_json(json.loads(text))
    return split_text(text), True


def split_paragraph(text: str, tokenizer, budget: int = INPUT_TOKEN_BUDGET) -> list[str]:
    """Split a paragraph into chunks of at most `budget` tokens. The cut is at
    the last sentence end of the second half of the window, else at the last
    whitespace, else at the token limit."""
    chunks = []
    rest = text.strip()
    while rest:
        encoding = tokenizer.encode(rest, add_special_tokens=False)
        if len(encoding.ids) <= budget:
            chunks.append(rest)
            break
        # Start of the first token beyond the budget, in characters.
        limit = max(encoding.offsets[budget][0], 1)
        window_start = limit // 2
        window = rest[window_start:limit]
        sentence_ends = list(SENTENCE_END.finditer(window))
        if sentence_ends:
            cut = window_start + sentence_ends[-1].end()
        else:
            spaces = list(re.finditer(r"\s+", window))
            cut = window_start + spaces[-1].end() if spaces else limit
        chunk = rest[:cut].strip()
        if chunk:
            chunks.append(chunk)
        rest = rest[cut:].strip()
    return chunks


def merge_chunks(index: int, outputs: list[dict]) -> dict:
    """One item for a paragraph, from the outputs of its chunks."""
    item = {"index": index, "is_boilerplate": None, "datasets": [], "software": []}
    if len(outputs) > 1:
        item["chunks"] = len(outputs)
    errors, raw_outputs, boilerplate = [], [], []
    for output in outputs:
        reason = failure(output)
        if reason is not None:
            errors.append(reason)
            raw = output.get("raw_output") or output.get("text")
            if raw:
                raw_outputs.append(raw[:2000])
            continue
        extraction = parse_json(output["text"])
        boilerplate.append(extraction.get("is_boilerplate"))
        item["datasets"] += extraction.get("datasets") or []
        item["software"] += extraction.get("software") or []
    if boilerplate:
        # Boilerplate only when every chunk is.
        item["is_boilerplate"] = (
            boilerplate[0] if len(boilerplate) == 1 else all(boilerplate)
        )
    if errors:
        item["error"] = "; ".join(errors)
    if raw_outputs:
        item["raw_output"] = raw_outputs
    return item


def failure(output: dict) -> str | None:
    """Why the output of a chunk cannot be used, or None."""
    if "error" in output:
        return output["error"]
    if output.get("finish_reason") == "length":
        return f"output cut at {MAX_TOKENS} tokens"
    if not parse_json(output["text"]):
        return "output is not a JSON object"
    return None


def extract_chunk(text: str, tokenizer, complete, depth: int = 0) -> list[dict]:
    """Outputs for one chunk: one output, or, when it failed, the outputs of
    its two halves. `complete` sends one prompt to the model."""
    output = complete(extraction_prompt(text))
    reason = failure(output)
    if reason is None:
        return [output]
    tokens = count_tokens(text, tokenizer)
    if depth < MAX_RETRY_DEPTH and tokens > MIN_RETRY_TOKENS:
        halves = split_paragraph(text, tokenizer, tokens // 2 + 1)
        if len(halves) > 1:
            return [
                o
                for half in halves
                for o in extract_chunk(half, tokenizer, complete, depth + 1)
            ]
    return [{"error": reason, "raw_output": output.get("text", "")}]


def extract_paragraphs(
    paragraphs: list[str], tokenizer, complete, is_text: bool, pool
) -> list[dict]:
    """Step 1. One item per paragraph; for a text, the paragraphs are its
    pieces of at most INPUT_TOKEN_BUDGET tokens."""
    if is_text:
        paragraphs = [c for p in paragraphs for c in split_paragraph(p, tokenizer)]
    groups = [split_paragraph(p, tokenizer) for p in paragraphs]
    chunks = [c for group in groups for c in group]
    outputs = list(pool.map(lambda c: extract_chunk(c, tokenizer, complete), chunks))

    items, position = [], 0
    for index, group in enumerate(groups):
        group_outputs = [
            o for chunk in outputs[position : position + len(group)] for o in chunk
        ]
        position += len(group)
        item = merge_chunks(index, group_outputs or [{"error": "empty paragraph"}])
        item["text"] = paragraphs[index]
        items.append(item)
    return items


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
        from tokenizers import Tokenizer

        download_model()
        self.tokenizer = Tokenizer.from_file(f"{MODEL_PATH}/tokenizer.json")
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
        """Greedy completion of one prompt. Never raises: a failure is reported
        for that prompt only."""
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
            choice = r.json()["choices"][0]
            return {"text": choice["text"], "finish_reason": choice["finish_reason"]}
        except Exception as e:
            return {"error": str(e)}

    @modal.method()
    def extract(self, paragraphs: list[str], analyze: bool = True, is_text: bool = False):
        import httpx

        with httpx.Client(timeout=600) as client:
            with ThreadPoolExecutor(max_workers=MAX_PARALLEL_REQUESTS) as pool:
                per_paragraph = extract_paragraphs(
                    paragraphs,
                    self.tokenizer,
                    lambda prompt: self._complete(client, prompt),
                    is_text,
                    pool,
                )

            mentions = aggregate_mentions(per_paragraph)
            result = {
                "num_paragraphs": len(per_paragraph),
                "paragraphs": per_paragraph,
                "mentions": mentions,
            }

            # Step 2: article-level record, only when there is something to analyse.
            if analyze and (mentions["datasets"] or mentions["software"]):
                output = self._complete(
                    client, analysis_prompt(json.dumps(mentions, ensure_ascii=False))
                )
                result["record"] = parse_json(output.get("text", ""))
                reason = failure(output)
                if reason is not None:
                    result["record_error"] = reason
                    result["record_raw_output"] = output.get("text", "")[:2000]

        return result


shared_service = BaguetteService()


def as_bool(value) -> bool:
    if isinstance(value, str):
        return value.strip().lower() not in ("false", "0", "no", "")
    return bool(value)


async def read_request(request: Request) -> tuple[list[str], bool, bool]:
    """The paragraphs of a request, whether the input is a text, and the
    'analyze' flag. The request is one of: a multipart form with a 'file' or a
    'text', a JSON body, a plain text body."""
    content_type = request.headers.get("content-type", "").lower()
    if content_type.startswith("multipart/form-data"):
        form = await request.form()
        analyze = as_bool(form.get("analyze", True))
        upload = form.get("file")
        if hasattr(upload, "read"):
            content = await upload.read()
            return *read_file(upload.filename or "document.txt", content), analyze
        if isinstance(form.get("text"), str):
            return split_text(form["text"]), True, analyze
        raise ValueError("multipart field 'file' or 'text' required")
    analyze = as_bool(request.query_params.get("analyze", True))
    if content_type.startswith("text/plain"):
        content = await request.body()
        text = content.decode("utf-8-sig", errors="replace")
        return split_text(text), True, analyze
    body = await request.json()
    if isinstance(body, dict) and "analyze" in body:
        analyze = as_bool(body["analyze"])
    return *read_json(body), analyze


@app.function(cpu=2, memory=4096, timeout=1800, max_containers=1)
@modal.concurrent(max_inputs=10)
@modal.fastapi_endpoint(method="POST")
async def extract_endpoint(request: Request):
    """One paper per request, as one of:
    JSON {"paragraphs": ["...", "..."], "analyze": true}
    JSON {"text": "...", "analyze": true}
    multipart form with 'file' (.txt, .md, .xml TEI, .json) or 'text'
    plain text body (Content-Type: text/plain)"""
    try:
        paragraphs, is_text, analyze = await read_request(request)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"invalid input: {e}") from e
    if not paragraphs:
        raise HTTPException(status_code=400, detail="no paragraph found in the input")

    t0 = time.time()
    try:
        result = await shared_service.extract.remote.aio(
            paragraphs, analyze, is_text
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
