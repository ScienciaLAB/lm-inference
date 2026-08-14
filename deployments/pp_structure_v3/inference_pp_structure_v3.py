"""
PP-StructureV3 (PaddleOCR) served on Modal with a GPU.

PP-StructureV3 is the *full* PaddleOCR document-parsing pipeline: layout
analysis + OCR + table recognition + formula recognition. It is NOT the
PP-DocLayout-* layout detector used by the hybrid GROBID pipeline; it is roughly
two orders of magnitude more expensive per page and is a peer of MinerU/Docling,
not of the detector.

Deployment shape mirrors deployments/minerU/inference_minerU.py so the cost
numbers are directly comparable:
  - same GPU class (A100-40GB),
  - one document per GPU (`max_inputs=1`, no per-doc GPU sharing),
  - the same `duration_seconds` x `cost_per_second` accounting in the response.

Self-contained: unlike the MinerU deployment this file does not import
`lm_inference_utils`, so it can be handed over and deployed on its own.

Deploy (from lm-inference/):
    modal deploy deployments/pp_structure_v3/inference_pp_structure_v3.py
"""

import json
import tempfile
import time
from pathlib import Path

import modal
from fastapi import HTTPException, Request

GPU_TYPE = "A100-40GB"
# Modal price for A100-40GB, USD/second. Matches GPU_COST_PER_SECOND in
# lm_inference_utils.py; update both together.
COST_PER_SECOND = 0.000583
NUM_GPUS_USED = 1

# PaddleX downloads its sub-models (detection, recognition, table, formula, ...)
# on first use. Persist them so only the very first container ever pays for it.
MODEL_CACHE = modal.Volume.from_name("ppstructurev3-paddlex-cache", create_if_missing=True)
MODEL_CACHE_PATH = "/root/.paddlex"

image = (
    # PaddlePaddle GPU wheels are built against a specific CUDA/cuDNN pair; the
    # cu126 wheel below needs a CUDA 12.6 runtime with cuDNN present.
    modal.Image.from_registry(
        "nvidia/cuda:12.6.3-cudnn-devel-ubuntu22.04", add_python="3.11"
    )
    .apt_install(
        "git",
        "wget",
        "libgl1",            # OpenCV runtime dep
        "libglib2.0-0",
        "libgomp1",
        "poppler-utils",     # pdf2image / PDF rasterization backend
        "fonts-dejavu-core", # PaddleX's bundled font download is unreliable
    )
    # Install paddlepaddle-gpu from PaddlePaddle's own index (it is NOT on PyPI
    # for GPU builds), then paddleocr from PyPI.
    .pip_install(
        "paddlepaddle-gpu==3.2.2",
        index_url="https://www.paddlepaddle.org.cn/packages/stable/cu126/",
        extra_options="--no-cache-dir",
    )
    .pip_install(
        "paddleocr[all]>=3.2.0",
        "pdf2image",
        "fastapi[standard]",
        extra_options="--no-cache-dir",
    )
    .env(
        {
            # PaddleX probes its model mirror at import time; the probe blocks
            # startup on hosts that cannot reach it.
            "PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK": "True",
            "PADDLE_PDX_CACHE_HOME": MODEL_CACHE_PATH,
        }
    )
    .run_commands(
        "python -c 'import paddle; print(\"paddle:\", paddle.__version__)'",
        "python -c 'from paddleocr import PPStructureV3; print(\"paddleocr OK\")'",
    )
)

app = modal.App("ppstructurev3-app", image=image)


@app.cls(
    gpu=GPU_TYPE,
    scaledown_window=300,
    max_containers=4,
    min_containers=1,
    timeout=1800,
    volumes={MODEL_CACHE_PATH: MODEL_CACHE},
)
# One document per container, matching the MinerU deployment: concurrent
# requests fan out across containers (one doc per GPU) rather than sharing a
# single GPU, so the per-document cost is not divided by a sharing factor.
@modal.concurrent(max_inputs=1)
class PPStructureV3Service:
    @modal.enter()
    def load_pipeline(self):
        from paddleocr import PPStructureV3

        print("Loading PP-StructureV3 pipeline...")
        t0 = time.time()

        self.pipeline = PPStructureV3(
            device="gpu:0",
            precision="fp32",
            # Preprocessing modules that cost time and add nothing for
            # born-digital scientific PDFs.
            use_doc_orientation_classify=False,
            use_doc_unwarping=False,
            use_textline_orientation=False,
            use_seal_recognition=False,
            # The parts that make this a full pipeline rather than a detector.
            use_table_recognition=True,
            use_formula_recognition=True,
            use_chart_recognition=False,  # chart-to-table VLM; very slow
        )
        print(f"Pipeline constructed in {time.time() - t0:.1f}s")

        self._warmup()
        # Persist any sub-models this container downloaded.
        MODEL_CACHE.commit()
        print("PP-StructureV3 ready.")

    def _warmup(self):
        """Force lazy sub-model init so the first real request is not charged for it."""
        from PIL import Image, ImageDraw

        img = Image.new("RGB", (1240, 1754), "white")
        ImageDraw.Draw(img).text((100, 100), "warmup 123 abc", fill="black")

        with tempfile.TemporaryDirectory() as td:
            p = Path(td) / "warmup.png"
            img.save(p)
            t0 = time.time()
            for _ in self.pipeline.predict(input=str(p)):
                pass
            print(f"Warmup pass in {time.time() - t0:.1f}s")

    @modal.method()
    def parse_document(self, file_bytes: bytes, original_filename: str):
        import shutil

        suffix = Path(original_filename).suffix.lower()
        if suffix not in {".pdf", ".jpg", ".jpeg", ".png"}:
            suffix = ".pdf"

        work_dir = tempfile.mkdtemp(prefix="ppsv3_")
        input_path = Path(work_dir) / f"input{suffix}"
        input_path.write_bytes(file_bytes)

        try:
            print(f"Processing {original_filename} ...")
            # predict() rasterizes the PDF internally, so this covers the whole
            # end-to-end cost, matching how the other systems are timed.
            results = list(self.pipeline.predict(input=str(input_path)))
            if not results:
                raise RuntimeError("PP-StructureV3 returned no pages")

            markdown_pages = [res.markdown for res in results]
            markdown = self.pipeline.concatenate_markdown_pages(markdown_pages)

            json_pages = []
            for res in results:
                # res.json is the structured layout+content representation.
                payload = getattr(res, "json", None)
                json_pages.append(
                    json.loads(payload) if isinstance(payload, str) else payload
                )

            print(f"Parsed {len(results)} pages, {len(markdown)} markdown chars")
            return {
                "markdown_content": markdown,
                "json_content": json_pages,
                "num_pages": len(results),
            }

        except Exception as e:
            print(f"PP-StructureV3 failed on {original_filename}: {e}")
            raise
        finally:
            shutil.rmtree(work_dir, ignore_errors=True)


shared_service = PPStructureV3Service()


@app.function(cpu=2, memory=4096, timeout=1800, max_containers=1)
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
        output_format = form.get("output_format", "markdown_content")

        start_time = time.perf_counter()

        # Async dispatch so concurrent requests are not serialized on the
        # endpoint's event loop; with max_inputs=1 they fan out one doc per GPU.
        outputs = await shared_service.parse_document.remote.aio(
            file_bytes=file_content, original_filename=original_filename
        )
        duration = time.perf_counter() - start_time

        results = outputs.get(output_format)
        num_pages = outputs.get("num_pages", 0)

        total_cost = duration * COST_PER_SECOND * NUM_GPUS_USED
        cost_result = {
            "duration_seconds": round(duration, 2),
            "cost_usd": round(total_cost, 6),
            "num_pages": num_pages,
            "sec_per_page": round(duration / num_pages, 3) if num_pages else 0.0,
        }

        print(
            f"[COST_LOG] File: {original_filename}, Pages: {num_pages}, "
            f"Duration: {cost_result['duration_seconds']}s, "
            f"Cost: ${cost_result['cost_usd']:.6f}"
        )

        return {"result": results, "cost_info": cost_result}

    except HTTPException:
        raise
    except Exception as e:
        print(f"Error in web endpoint: {e}")
        raise HTTPException(
            status_code=500, detail=f"Error processing document: {str(e)}"
        )
