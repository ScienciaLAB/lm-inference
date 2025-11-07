import modal
import subprocess
import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(message)s")
logger = logging.getLogger(__name__)

app = modal.App(name="docling-serve")

# Configuration constants
MINUTES = 60 # seconds
# CU126 GPU image from github container registry
DOCLING_IMAGE = "ghcr.io/docling-project/docling-serve-cu126:latest"
PORT = 5001
ARTIFACTS_PATH = "/artifacts"
STARTUP_TIMEOUT = 10 * MINUTES  # 30 minutes - increased timeout for model downloads and startup

image = (
    modal.Image.from_registry(DOCLING_IMAGE)
    .run_commands(
        f"docling-tools models download granitedocling layout tableformer picture_classifier easyocr --output-dir {ARTIFACTS_PATH}",
    )
    .env({
        "DOCLING_SERVE_MAX_SYNC_WAIT": f"{STARTUP_TIMEOUT}",
        "DOCLING_SERVE_ENG_LOC_ARTIFACTS_PATH": ARTIFACTS_PATH,
        "DOCLING_SERVE_ENG_LOC_SHARE_MODELS": "true",
        "DOCLING_NUM_THREADS": "8"
    })
)

@app.function(
    image=image,
    timeout=STARTUP_TIMEOUT,
    # GPU A100 for performance
    gpu="A10",
    scaledown_window=10 * MINUTES,  # 30 minutes - keep containers warm longer
    memory=16384,  # 16GB memory for document processing
    max_containers=2  # Limit concurrent instances
)
@modal.web_server(PORT, startup_timeout=STARTUP_TIMEOUT, label="docling")
def start_docling_server():
    logger.info(f"Starting Docling server on port {PORT}")
    
    # Command to run the server
    cmd = f"docling-serve -vv run --host 0.0.0.0 --port {PORT} --artifacts-path {ARTIFACTS_PATH}"
    try:
        subprocess.Popen(cmd, shell=True)
        logger.info("Docling server process started")
        return "Starting Docling server..."
    except Exception as e:
        logger.error("Failed to start server: %s", str(e))
        raise

