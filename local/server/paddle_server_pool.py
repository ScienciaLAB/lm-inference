"""
Each worker loads M model instances
into a pool, gated by an asyncio.Semaphore.  This allows a single worker
to process M PDFs in parallel instead of queuing them behind a single lock.

Usage:
    python -m local.server.paddle_server_pool \
        --workers 2 --models-per-worker 4 --port 8080

    Total model instances = workers × models-per-worker
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
logger = logging.getLogger("paddle_server_pool")

_model_pool: List[Any] = []  # Pool of PaddleDocumentProcessor instances
_model_semaphore: asyncio.Semaphore = None  # Semaphore to limit concurrent access
_model_ready = False
_server_config: Dict[str, Any] = {}


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
    models_per_worker: int
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


#  Type-aggregation maps
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

# Lifespan: load M models into pool at startup
_start_time: float = 0.0


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load M PaddlePaddle model instances into a pool at startup."""
    global _model_pool, _model_semaphore, _model_ready, _start_time

    _start_time = time.time()
    model_name = os.environ.get("PADDLE_MODEL_NAME", "PP-DocLayout-L")
    dpi = int(os.environ.get("PADDLE_DPI", "72"))
    models_per_worker = int(os.environ.get("PADDLE_MODELS_PER_WORKER", "1"))
    # Options are handed over as environment variables because uvicorn workers
    # are separate processes that re-import this module.
    raw_threshold = os.environ.get("PADDLE_THRESHOLD")
    threshold = parse_threshold(raw_threshold) if raw_threshold else None
    layout_merge_bboxes_mode = os.environ.get("PADDLE_LAYOUT_MERGE_BBOXES_MODE") or None

    logger.info(
        "Loading %d PaddlePaddle model(s) '%s' (dpi=%d) …",
        models_per_worker,
        model_name,
        dpi,
    )
    load_start = time.time()

    from local.paddle_inference import PaddleDocumentProcessor

    _model_pool = []
    for i in range(models_per_worker):
        logger.info("Loading model instance %d/%d …", i + 1, models_per_worker)
        processor = PaddleDocumentProcessor(
            model_name=model_name,
            dpi=dpi,
            preload_model=True,
            threshold=threshold,
            layout_merge_bboxes_mode=layout_merge_bboxes_mode,
        )
        _model_pool.append(processor)

    # Semaphore limits concurrent inference to the number of models available
    _model_semaphore = asyncio.Semaphore(models_per_worker)
    _model_ready = True
    logger.info(
        "%d model(s) loaded in %.1f s",
        models_per_worker,
        time.time() - load_start,
    )

    yield

    # Cleanup on shutdown
    _model_ready = False
    _model_pool = []
    _model_semaphore = None
    logger.info("Server shutting down, %d model(s) released.", models_per_worker)


#  FastAPI app
app = FastAPI(
    title="PaddlePaddle Layout Detection Server (Pool)",
    description=(
        "Upload a PDF document and receive typed-area bounding boxes "
        "(figures, tables, equations, headers, footers, …) detected by "
        "PaddlePaddle layout analysis.  This variant uses a model pool "
        "with an asyncio.Semaphore for intra-worker parallelism."
    ),
    version="2.0.0",
    lifespan=lifespan,
)


#  Endpoints
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
    """Returns 200 when all models are loaded and ready."""
    model_name = os.environ.get("PADDLE_MODEL_NAME", "PP-DocLayout-L")
    models_per_worker = int(os.environ.get("PADDLE_MODELS_PER_WORKER", "1"))
    if not _model_ready:
        raise HTTPException(
            status_code=503,
            detail="Models are not loaded yet. The server is still starting.",
        )
    return ReadyResponse(
        ready=True,
        model_name=model_name,
        models_per_worker=models_per_worker,
        detail=f"{models_per_worker} model(s) loaded and ready.",
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
        raise HTTPException(status_code=503, detail="Models are not loaded yet.")

    # Validate file type
    if not file.filename or not file.filename.lower().endswith(".pdf"):
        raise HTTPException(
            status_code=400,
            detail=f"Expected a PDF file, got: '{file.filename}'",
        )

    tmp_dir = Path(tempfile.mkdtemp(prefix="paddle_server_"))
    try:
        # Save uploaded PDF to disk
        pdf_path = tmp_dir / file.filename
        content = await file.read()
        pdf_path.write_bytes(content)
        logger.info("Received '%s' (%d bytes)", file.filename, len(content))

        # Acquire the semaphore blocks if all M models are busy
        async with _model_semaphore:
            # Grab an available model from the pool
            model = _model_pool.pop()
            try:
                result, elements = await asyncio.to_thread(
                    _run_inference,
                    model,
                    str(pdf_path),
                    str(tmp_dir),
                    filter,
                    merge_captions,
                )
            finally:
                _model_pool.append(model)

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
        shutil.rmtree(tmp_dir, ignore_errors=True)


def _run_inference(
    processor,
    pdf_path: str,
    tmp_dir: str,
    filter_mode: Optional[FilterMode],
    merge_captions: bool,
) -> tuple:

    from local.paddle_inference import load_transform_elements
    from lm_inference_utils import filter_and_aggregate

    output_dir = os.path.join(tmp_dir, "output")
    os.makedirs(output_dir, exist_ok=True)

    result = processor.process_document(pdf_path, output_dir)

    if not result.get("success", False):
        return result, []

    elements = load_transform_elements(result["output_dir"])

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
        if filter_mode == FilterMode.display:
            elements = filter_and_aggregate(elements, FIGURE_TYPE_AGGREGATION)
        elif filter_mode == FilterMode.paratext:
            elements = filter_and_aggregate(elements, PARATEXT_TYPE_AGGREGATION)
        elif filter_mode == FilterMode.grobid:
            elements = filter_and_aggregate(elements, GROBID_TYPE_AGGREGATION)

    return result, elements


def main():
    parser = argparse.ArgumentParser(
        description="Start the PaddlePaddle layout detection server (Pool variant)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""\
Examples:
  python -m local.server.paddle_server_pool --workers 1 --models-per-worker 2
  python -m local.server.paddle_server_pool --workers 2 --models-per-worker 4 --port 8080
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
        help="Number of uvicorn worker processes (default: 1). "
        "Each worker loads its own set of models.",
    )
    parser.add_argument(
        "--models-per-worker",
        type=int,
        default=1,
        help="Number of model instances per worker (default: 1). "
        "Higher values allow more parallel PDF processing per worker. "
        "Total models = workers × models-per-worker.",
    )
    parser.add_argument(
        "--reload",
        action="store_true",
        help="Enable auto-reload for development",
    )

    args = parser.parse_args()

    os.environ["PADDLE_MODEL_NAME"] = args.model_name
    os.environ["PADDLE_DPI"] = str(args.dpi)
    if args.threshold:
        # Validated here so a malformed value fails before the workers start
        parse_threshold(args.threshold)
        os.environ["PADDLE_THRESHOLD"] = args.threshold
    if args.layout_merge_bboxes_mode:
        os.environ["PADDLE_LAYOUT_MERGE_BBOXES_MODE"] = args.layout_merge_bboxes_mode
    os.environ["PADDLE_MODELS_PER_WORKER"] = str(args.models_per_worker)

    total_models = args.workers * args.models_per_worker
    logger.info(
        "Total model instances: %d (workers=%d × models_per_worker=%d)",
        total_models,
        args.workers,
        args.models_per_worker,
    )

    import uvicorn

    logger.info(
        "Starting server on %s:%d (model=%s, dpi=%d, workers=%d, models_per_worker=%d)",
        args.host,
        args.port,
        args.model_name,
        args.dpi,
        args.workers,
        args.models_per_worker,
    )

    uvicorn.run(
        "local.server.paddle_server_pool:app",
        host=args.host,
        port=args.port,
        workers=args.workers,
        reload=args.reload,
        log_level="info",
    )


if __name__ == "__main__":
    main()
