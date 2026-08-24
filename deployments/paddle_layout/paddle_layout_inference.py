"""
PaddlePaddle Layout Detection
Deploys PaddleOCR LayoutDetection (PP-DocLayout-L) as
CPU endpoint on Modal.com.

Endpoints:
    POST /parse   — single PDF → JSON bounding boxes
    POST /batch   — multiple PDFs → list of JSON results
    GET  /health  — container health check

Deploy:
    modal deploy Deployments/paddle_layout/paddle_layout_inference.py
"""

import json
import os
import tempfile
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import modal
from fastapi import HTTPException, Request, UploadFile, File, Form
from fastapi.responses import JSONResponse

from lm_inference_utils import filter_and_aggregate

# Constants
MODEL_NAME = "PP-DocLayout-L"
DEFAULT_DPI = 72

# Configuration number of workers and number of models per worker (one worker = one container)
N_WORKERS = int(os.environ.get("N_WORKERS", "2"))
M_PARALLEL = int(os.environ.get("M_PARALLEL", "1"))
CPU_CORE_PER_CONTAINER = 8.0
RAM_PER_CONTAINER = max(16384, M_PARALLEL * 4096)

# Modal Image CPU-only PaddlePaddle
image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("poppler-utils", "libgomp1", "libgl1", "libglib2.0-0")
    .pip_install(
        "paddlepaddle==3.2.0",
        "paddleocr==3.4.0",
        "pdf2image==1.17.0",
        "ultralytics>=8.3.0",
        "fastapi",
        "python-multipart",
    )
    # Pre-download model weights at build time
    .run_commands(
        f'python -c "'
        f"from paddleocr import LayoutDetection; "
        f"LayoutDetection(model_name='{MODEL_NAME}')"
        f'"'
    )
    .env({"N_WORKERS": str(N_WORKERS), "M_PARALLEL": str(M_PARALLEL)})
    .add_local_file("lm_inference_utils.py", "/root/lm_inference_utils.py")
)

app = modal.App("paddle-layout-cpu", image=image)


FIGURE_TYPE_AGGREGATION = {
    "figure": ["figure", "image", "chart", "figure_text", "chart_text"],
    "table": ["table", "table_text"],
    "equation": ["equation", "formula", "equation_text"],
}

PARATEXT_TYPE_AGGREGATION = {
    "headnote": ["header"],
    "footer": ["footer"],
}

GROBID_TYPE_AGGREGATION = {
    **FIGURE_TYPE_AGGREGATION,
    **PARATEXT_TYPE_AGGREGATION,
}

FILTER_MAP = {
    "display": FIGURE_TYPE_AGGREGATION,
    "paratext": PARATEXT_TYPE_AGGREGATION,
    "grobid": GROBID_TYPE_AGGREGATION,
}


def _pdf_to_images(pdf_path: str, output_dir: Path, dpi: int) -> List[Path]:
    """Convert a PDF to per-page JPEG images using poppler."""
    from pdf2image import convert_from_path

    images = convert_from_path(pdf_path, dpi=dpi, thread_count=os.cpu_count())
    image_paths = []
    for i, img in enumerate(images, 1):
        p = output_dir / f"page_{i:04d}.jpg"
        img.save(p, "JPEG")
        image_paths.append(p)
    return image_paths


def _transform_results(raw_output, output_dir: Path) -> List[Dict[str, Any]]:
    """
    Transform PaddleOCR LayoutDetection output into standardised
    bounding-box dicts: {page, x, y, width, height, type}.

    Uses save_to_json → file read to match the proven local pipeline.
    """
    # Save each page result to JSON via the SDK
    for i, res in enumerate(raw_output):
        res.page_index = i + 1
        res.save_to_json(save_path=str(output_dir / f"res_{i}.json"))

    # Read back and transform — same logic as local/paddle_inference.py
    elements: List[Dict[str, Any]] = []
    for json_file in sorted(output_dir.glob("res_*.json")):
        page_number = int(json_file.stem.split("_")[1]) + 1
        with open(json_file, "r", encoding="utf-8") as f:
            data = json.load(f)

        for box in data.get("boxes", []):
            coords = box.get("coordinate", [0, 0, 0, 0])
            x1, y1, x2, y2 = coords
            elements.append(
                {
                    "page": page_number,
                    "x": int(x1),
                    "y": int(y1),
                    "width": int(x2 - x1),
                    "height": int(y2 - y1),
                    "type": box.get("label", "unknown"),
                }
            )
    return elements


@app.cls(
    cpu=CPU_CORE_PER_CONTAINER,
    memory=RAM_PER_CONTAINER,
    timeout=1800,
    scaledown_window=120,
    max_containers=N_WORKERS,
    min_containers=0,
)
@modal.concurrent(max_inputs=M_PARALLEL)
class PaddleLayoutService:
    """Stateful container that loads the model once and serves requests."""

    @modal.enter()
    def load_model(self):
        import queue
        from paddleocr import LayoutDetection

        print(f"Loading {M_PARALLEL} instance(s) of {MODEL_NAME}...")
        t0 = time.time()
        self.model_pool = queue.Queue()

        for i in range(M_PARALLEL):
            print(f"Loading model instance {i + 1}/{M_PARALLEL}...")
            self.model_pool.put(LayoutDetection(model_name=MODEL_NAME))

        print(f"All {M_PARALLEL} model(s) loaded in {time.time() - t0:.1f}s")

    @modal.method()
    def process_pdf(
        self,
        file_bytes: bytes,
        filename: str,
        dpi: int = DEFAULT_DPI,
        only: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Process a single PDF and return bounding boxes.

        Args:
            file_bytes: Raw PDF content.
            filename:   Original filename (used for logging).
            dpi:        Resolution for PDF → image conversion.
            only:       Optional filter — "display", "paratext", or "grobid".

        Returns:
            Dict with keys: filename, num_pages, bounding_boxes,
            processing_time_s, model.
        """
        # Block until a model is available in the pool
        model = self.model_pool.get()

        try:
            with tempfile.TemporaryDirectory(prefix="paddle_") as tmp_dir:
                tmp_path = Path(tmp_dir)

                # Write PDF to disk
                pdf_path = tmp_path / filename
                pdf_path.write_bytes(file_bytes)

                print(f"[START] Processing {filename}...")

                # PDF → images
                image_paths = _pdf_to_images(str(pdf_path), tmp_path, dpi)
                image_strs = [str(p) for p in image_paths]

                # Inference
                t0 = time.time()
                raw_output = model.predict(
                    image_strs, batch_size=os.cpu_count(), layout_nms=True
                )
                inference_time = time.time() - t0

                # Transform
                bounding_boxes = _transform_results(raw_output, tmp_path)

                # Optional filtering
                if only and only in FILTER_MAP:
                    bounding_boxes = filter_and_aggregate(
                        bounding_boxes, FILTER_MAP[only]
                    )

                print(f"[DONE] {filename} finished in {inference_time:.1f}s")

            return {
                "filename": filename,
                "num_pages": len(image_paths),
                "elements": bounding_boxes,
                "processing_time_s": round(inference_time, 3),
                "model": MODEL_NAME,
            }
        finally:
            self.model_pool.put(model)


# Shared reference used by the FastAPI endpoints
_service = PaddleLayoutService()


# Endpoints
@app.function(cpu=1, memory=2048, timeout=1800, max_containers=2)
@modal.concurrent(max_inputs=20)
@modal.fastapi_endpoint(method="GET")
async def health():
    """Simple health / readiness probe."""
    return {"status": "ok", "model": MODEL_NAME}


@app.function(cpu=1, memory=2048, timeout=1800, max_containers=2)
@modal.concurrent(max_inputs=20)
@modal.fastapi_endpoint(method="POST")
async def parse(request: Request):
    """
    Process a single PDF document.

    Form fields:
        file        — PDF file (multipart upload, required)
        dpi         — int, default 72
        only        — "display" | "paratext" | "grobid" (optional)
    """
    from starlette.datastructures import UploadFile as StarletteUploadFile

    try:
        form = await request.form()

        # Validate file
        if "file" not in form or not isinstance(form["file"], StarletteUploadFile):
            raise HTTPException(
                status_code=400,
                detail="Missing or invalid 'file' in multipart form.",
            )

        upload: StarletteUploadFile = form["file"]
        file_bytes = await upload.read()
        filename = upload.filename or "document.pdf"

        if not filename.lower().endswith(".pdf"):
            raise HTTPException(status_code=400, detail="Only PDF files are supported.")

        dpi = int(form.get("dpi", DEFAULT_DPI))
        only = form.get("only", None)

        if only and only not in FILTER_MAP:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid 'only' value. Choose from: {list(FILTER_MAP.keys())}",
            )

        # Dispatch to the CPU service container
        start = time.perf_counter()
        result = await _service.process_pdf.remote.aio(
            file_bytes=file_bytes,
            filename=filename,
            dpi=dpi,
            only=only,
        )
        wall_time = time.perf_counter() - start

        timing = {
            "wall_time_s": round(wall_time, 2),
            "inference_time_s": result["processing_time_s"],
        }

        print(
            f"[TIMING] File={filename} "
            f"Wall={timing['wall_time_s']}s "
            f"Inference={timing['inference_time_s']}s"
        )

        return JSONResponse(
            content={
                "result": result,
                "timing": timing,
            }
        )

    except HTTPException:
        raise
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        print(f"[ERROR] parse endpoint: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Error processing document: {str(e)}",
        )


@app.function(cpu=1, memory=2048, timeout=3600, max_containers=2)
@modal.concurrent(max_inputs=10)
@modal.fastapi_endpoint(method="POST")
async def batch(
    files: List[UploadFile] = File(...),
    dpi: int = Form(DEFAULT_DPI),
    only: Optional[str] = Form(None),
):
    import asyncio

    try:
        if not files:
            raise HTTPException(
                status_code=400,
                detail="No PDF files found in the request.",
            )

        # Filter to only .pdf files
        pdf_files = [f for f in files if (f.filename or "").lower().endswith(".pdf")]

        if not pdf_files:
            raise HTTPException(
                status_code=400,
                detail="No valid .pdf files found in the 'files' field.",
            )

        if only and only not in FILTER_MAP:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid 'only' value. Choose from: {list(FILTER_MAP.keys())}",
            )

        # Read all file bytes upfront
        pdf_items = []
        for upload in pdf_files:
            file_bytes = await upload.read()
            filename = upload.filename or "document.pdf"
            pdf_items.append((file_bytes, filename))

        # spread across containers in parallel (max_inputs=1 per container).
        start = time.perf_counter()

        async def _process_one(file_bytes: bytes, filename: str):
            try:
                result = await _service.process_pdf.remote.aio(
                    file_bytes=file_bytes,
                    filename=filename,
                    dpi=dpi,
                    only=only,
                )
                return {"ok": True, "result": result, "filename": filename}
            except Exception as exc:
                return {"ok": False, "error": str(exc), "filename": filename}

        outcomes = await asyncio.gather(*[_process_one(fb, fn) for fb, fn in pdf_items])

        wall_time = time.perf_counter() - start

        batch_results = []
        errors = []
        for outcome in outcomes:
            if outcome["ok"]:
                batch_results.append(outcome["result"])
            else:
                errors.append(
                    {"filename": outcome["filename"], "error": outcome["error"]}
                )

        summary = {
            "total_files": len(files),
            "successful": len(batch_results),
            "failed": len(errors),
            "wall_time_s": round(wall_time, 2),
        }

        print(
            f"[TIMING] Batch: {summary['total_files']} files, "
            f"Wall={summary['wall_time_s']}s"
        )

        return JSONResponse(
            content={
                "results": batch_results,
                "errors": errors,
                "summary": summary,
            }
        )

    except HTTPException:
        raise
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        import traceback

        print(f"[ERROR] batch endpoint exception: {type(e).__name__} - {e}")
        traceback.print_exc()
        raise HTTPException(
            status_code=500,
            detail=f"Error processing batch: {str(e)}",
        )
