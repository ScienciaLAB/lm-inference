import os

import modal
import subprocess
import logging
from dotenv import load_dotenv
load_dotenv()

# Basic logging for better debugging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(message)s")
logger = logging.getLogger(__name__)

# Modal app configuration
app = modal.App(name="docling-serve-gpu")

# Configuration constants
# CU126 GPU image from github container registry
DOCLING_IMAGE = "ghcr.io/docling-project/docling-serve-cu126:latest"
PORT = 5001
ARTIFACTS_PATH = "/artifacts"
STARTUP_TIMEOUT = 600

# Persistent volume for model artifacts to cache downloads across deployments
artifacts_vol = modal.Volume.from_name("docling-artifacts", create_if_missing=True)

image = (
    modal.Image.from_registry(DOCLING_IMAGE).run_commands(
        # Pre-download Granite-Docling
        "python -c 'from huggingface_hub import snapshot_download; snapshot_download(repo_id=\"ibm-granite/granite-docling-258M\", local_dir=\"/artifacts/models/ibm-granite/granite-docling-258M\")'"
    )
    .env({
        "DOCLING_ARTIFACTS_PATH": ARTIFACTS_PATH,
        #"DOCLING_MODEL_PROVIDER": "huggingface",
        #"DOCLING_MODEL_NAME": "ibm-granite/granite-3.1-vlm",
        #"HUGGINGFACE_TOKEN": os.getenv("HUGG_TOKEN"),
    })
)

@app.function(
    image=image,
    volumes={ARTIFACTS_PATH: artifacts_vol},
    timeout=STARTUP_TIMEOUT,
    # GPU A10 for performance
    gpu="A10",
    scaledown_window=600)
@modal.web_server(PORT, startup_timeout=STARTUP_TIMEOUT, label="docling-gpu")
def start_docling_server():
    logger.info("Starting Docling server on port %d", PORT)
    
    # Command to run the server
    cmd = f"docling-serve run --host 0.0.0.0 --port {PORT}"

    if os.environ.get("DOCLING_SERVE_ENABLE_UI", "0") == "1":
        cmd += " --enable-ui"
        logger.info("Gradio UI enabled")

    try:
        subprocess.Popen(cmd, shell=True)
        logger.info("Docling server process started")
        return "Starting Docling server..."
    except Exception as e:
        logger.error("Failed to start server: %s", str(e))
        raise

