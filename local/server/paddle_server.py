"""
PaddlePaddle Layout Detection Server

A FastAPI service that wraps the PaddleDocumentProcessor to expose
PaddlePaddle layout detection as an HTTP API. Clients POST a PDF and
receive typed-area bounding boxes (figures, tables, equations, etc.) as JSON.

"""

import argparse
import asyncio
import logging
import os
import shutil
import tempfile
import time

try:
    import psutil
except ImportError:
    psutil = None
from contextlib import asynccontextmanager
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, File, HTTPException, Query, UploadFile
from pydantic import BaseModel

from local.paddle_options import (
    LAYOUT_MERGE_BBOXES_MODES,
    LAYOUT_MODEL_CHOICES,
    parse_threshold,
)

# Logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("paddle_server")

# Global state
_processor = None
_model_ready = False
_server_config: Dict[str, Any] = {}
_inference_lock = asyncio.Lock()


#  Pydantic models
class FilterMode(str, Enum):
    """Available filter modes for bounding-box output."""

    display = "display"
    paratext = "paratext"
    grobid = "grobid"


class HealthResponse(BaseModel):
    status: str
    uptime_seconds: float


class ReadyResponse(BaseModel):
    ready: bool
    model_name: str
    detail: str


class ProcessResponse(BaseModel):
    """Wrapper for the /process endpoint response"""

    filename: str
    num_pages: int
    num_elements: int
    processing_time_seconds: float
    throughput_pages_per_second: Optional[float] = None
    system_ram_mb: Optional[float] = None
    system_cpu_percent: Optional[float] = None
    elements: List[Dict[str, Any]]


class ErrorResponse(BaseModel):
    detail: str


# Type-aggregation maps
FIGURE_TYPE_AGGREGATION = {
    "figure": ["figure", "image", "chart", "figure_text", "chart_text"],
    "table": ["table", "table_text"],
    "equation": ["equation", "formula", "equation_text", "display_formula"],
}

PARATEXT_TYPE_AGGREGATION = {
    "headnote": ["header"],
    "footer": ["footer"],
}

GROBID_TYPE_AGGREGATION = {
    **FIGURE_TYPE_AGGREGATION,
    **PARATEXT_TYPE_AGGREGATION,
}

# Lifespan: model loaded once at startup, released on shutdown
_start_time: float = 0.0


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load the PaddlePaddle model once at startup."""
    global _processor, _model_ready, _start_time

    _start_time = time.time()
    model_name = _server_config.get("model_name", "PP-DocLayout-L")
    dpi = _server_config.get("dpi", 72)

    logger.info("Loading PaddlePaddle model '%s' (dpi=%d) …", model_name, dpi)
    load_start = time.time()

    from local.paddle_inference import PaddleDocumentProcessor

    _processor = PaddleDocumentProcessor(
        model_name=model_name,
        dpi=dpi,
        preload_model=True,
        threshold=_server_config.get("threshold"),
        layout_merge_bboxes_mode=_server_config.get("layout_merge_bboxes_mode"),
    )
    _model_ready = True
    logger.info("Model loaded in %.1f s", time.time() - load_start)

    yield

    # Cleanup on shutdown
    _model_ready = False
    _processor = None
    logger.info("Server shutting down, model released.")


# FastAPI app
app = FastAPI(
    title="PaddlePaddle Layout Detection Server",
    description=(
        "Upload a PDF document and receive typed-area bounding boxes "
        "(figures, tables, equations, headers, footers, …) detected by "
        "PaddlePaddle layout analysis."
    ),
    version="1.0.0",
    lifespan=lifespan,
)


# Endpoints
@app.get(
    "/health",
    response_model=HealthResponse,
    summary="Liveness check",
    tags=["Status"],
)
async def health():
    """Returns 200 as long as the server process is alive."""
    return HealthResponse(
        status="ok",
        uptime_seconds=round(time.time() - _start_time, 2),
    )


@app.get(
    "/ready",
    response_model=ReadyResponse,
    summary="Readiness check",
    tags=["Status"],
)
async def ready():
    """Returns 200 when the model is loaded and ready"""
    model_name = _server_config.get("model_name", "PP-DocLayout-L")
    if not _model_ready:
        raise HTTPException(
            status_code=503,
            detail="Model is not loaded yet. The server is still starting.",
        )
    return ReadyResponse(
        ready=True,
        model_name=model_name,
        detail="Model is loaded and ready.",
    )


@app.post(
    "/process",
    response_model=ProcessResponse,
    responses={
        400: {"model": ErrorResponse, "description": "Invalid input"},
        503: {"model": ErrorResponse, "description": "Model not ready"},
    },
    summary="Process a PDF and return typed-area bounding boxes",
    tags=["Inference"],
)
async def process_pdf(
    file: UploadFile = File(..., description="PDF document to process"),
    filter: Optional[FilterMode] = Query(
        default=None,
        description=(
            "Filter bounding boxes to a specific category. "
            "'display' = figures/tables/equations, "
            "'paratext' = headers/footers, "
            "'grobid' = both. "
            "Leave empty for all raw boxes."
        ),
    ),
    merge_captions: bool = Query(
        default=True,
        description=(
            "When true, captions are merged with their parent "
            "figures/tables and output is cleaned for Grobid consumption. "
            "When false, raw PaddlePaddle boxes are returned."
        ),
    ),
):
    """
    Upload a PDF receive typed-area bounding boxes as JSON.

    The server runs PaddlePaddle layout detection, optionally merges
    caption bounding boxes with their parent figures/tables, and
    optionally filters to specific element categories.
    """
    if not _model_ready:
        raise HTTPException(status_code=503, detail="Model is not loaded yet.")

    # Validate file type
    if not file.filename or not file.filename.lower().endswith(".pdf"):
        raise HTTPException(
            status_code=400,
            detail=f"Expected a PDF file, got: '{file.filename}'",
        )

    # Create a temporary directory for this request
    tmp_dir = Path(tempfile.mkdtemp(prefix="paddle_server_"))
    try:
        # Save uploaded PDF to disk
        pdf_path = tmp_dir / file.filename
        content = await file.read()
        pdf_path.write_bytes(content)
        logger.info("Received '%s' (%d bytes)", file.filename, len(content))

        # Run inference in a thread to avoid blocking the event loop
        # We use a lock because the PaddlePaddle model instance is NOT thread-safe
        async with _inference_lock:
            result, elements = await asyncio.to_thread(
                _run_inference, str(pdf_path), str(tmp_dir), filter, merge_captions
            )

        if not result.get("success", False):
            raise HTTPException(
                status_code=500,
                detail=f"Processing failed: {result.get('error', 'Unknown error')}",
            )

        # Get RAM and CPU usage
        sys_ram = None
        sys_cpu = None
        if psutil:
            process = psutil.Process()
            sys_ram = round(process.memory_info().rss / (1024 * 1024), 2)
            sys_cpu = psutil.cpu_percent()

        # Calculate throughput
        num_pages = result.get("num_pages", 0)
        proc_time = round(result.get("processing_time", 0), 3)
        throughput = round(num_pages / proc_time, 2) if proc_time > 0 else 0.0

        return ProcessResponse(
            filename=file.filename,
            num_pages=num_pages,
            num_elements=len(elements),
            processing_time_seconds=proc_time,
            throughput_pages_per_second=throughput,
            system_ram_mb=sys_ram,
            system_cpu_percent=sys_cpu,
            elements=elements,
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.exception("Unexpected error processing '%s'", file.filename)
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        # Always clean up temp files
        shutil.rmtree(tmp_dir, ignore_errors=True)


# Inference helper (runs in a thread)


def _run_inference(
    pdf_path: str,
    tmp_dir: str,
    filter_mode: Optional[FilterMode],
    merge_captions: bool,
) -> tuple:

    from local.paddle_inference import load_transform_elements
    from lm_inference_utils import filter_and_aggregate

    output_dir = os.path.join(tmp_dir, "output")
    os.makedirs(output_dir, exist_ok=True)

    # Run PaddlePaddle inference
    result = _processor.process_document(pdf_path, output_dir)

    if not result.get("success", False):
        return result, []

    # Load and normalize bounding boxes
    elements = load_transform_elements(result["output_dir"])

    # Optionally merge captions with parent figures/tables
    if merge_captions:
        try:
            from local.caption_merging import (
                DistanceMerger,
                link_captions_and_merge,
                _to_output_item,
            )

            figures, tables, paratext_areas = link_captions_and_merge(
                elements, DistanceMerger()
            )

            # Rebuild the elements list from merged results
            merged_elements = []
            for item in figures:
                merged_elements.append(_to_output_item(item, "figure"))
            for item in tables:
                merged_elements.append(_to_output_item(item, "table"))
            for item in paratext_areas:
                merged_elements.append(_to_output_item(item, "ignore"))

            merged_elements.sort(key=lambda x: (x["page"], x["y"]))
            elements = merged_elements
        except ImportError:
            logger.warning("caption_merging module not available, skipping merge.")

    # Apply filter if requested
    if filter_mode and not merge_captions:
        # Only apply raw filter when merge_captions is off.
        # When merge_captions is on, the caption_merging pipeline already
        # produces the cleaned output (figure/table/paratext only).
        if filter_mode == FilterMode.display:
            elements = filter_and_aggregate(elements, FIGURE_TYPE_AGGREGATION)
        elif filter_mode == FilterMode.paratext:
            elements = filter_and_aggregate(elements, PARATEXT_TYPE_AGGREGATION)
        elif filter_mode == FilterMode.grobid:
            elements = filter_and_aggregate(elements, GROBID_TYPE_AGGREGATION)

    return result, elements


# CLI entry point


def main():
    parser = argparse.ArgumentParser(
        description="Start the PaddlePaddle layout detection server",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""\
Examples:
  python -m local.server.paddle_server
  python -m local.server.paddle_server --model-name PP-DocLayout-L --port 8080
  python -m local.server.paddle_server --host 0.0.0.0 --port 8000 --workers 2
        """,
    )
    parser.add_argument(
        "--host",
        default="0.0.0.0",
        help="Bind address (default: 0.0.0.0)",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8080,
        help="Port to listen on (default: 8080)",
    )
    parser.add_argument(
        "--model-name",
        choices=LAYOUT_MODEL_CHOICES,
        default="PP-DocLayout-L",
        help="PaddlePaddle model for layout detection (default: PP-DocLayout-L)",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=72,
        help="DPI for PDF to image conversion (default: 72)",
    )
    parser.add_argument(
        "--threshold",
        type=parse_threshold,
        help="Detection score threshold: a single value (0.4) or per-class "
        "values as class:score pairs, keyed by class index or label "
        "(e.g. inline_formula:0.2 to keep low-confidence embedded math)",
    )
    parser.add_argument(
        "--layout-merge-bboxes-mode",
        choices=LAYOUT_MERGE_BBOXES_MODES,
        help="How to merge overlapping boxes: 'union' keeps the envelope, "
        "'large' keeps the outer box, 'small' keeps the inner one",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help="Number of uvicorn workers (default: 1). "
        "Use >1 only if you have enough GPU/CPU memory for multiple model copies.",
    )
    parser.add_argument(
        "--reload",
        action="store_true",
        help="Enable auto-reload for development",
    )

    args = parser.parse_args()

    # Store config so the lifespan function can read it
    _server_config["model_name"] = args.model_name
    _server_config["dpi"] = args.dpi
    _server_config["threshold"] = args.threshold
    _server_config["layout_merge_bboxes_mode"] = args.layout_merge_bboxes_mode

    import uvicorn

    logger.info(
        "Starting server on %s:%d (model=%s, dpi=%d, workers=%d)",
        args.host,
        args.port,
        args.model_name,
        args.dpi,
        args.workers,
    )

    uvicorn.run(
        "local.server.paddle_server:app",
        host=args.host,
        port=args.port,
        workers=args.workers,
        reload=args.reload,
        log_level="info",
    )


if __name__ == "__main__":
    main()
