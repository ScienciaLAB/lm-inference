import modal
import subprocess
import logging

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
    modal.Image.from_registry(DOCLING_IMAGE)
    .env({
        "DOCLING_SERVE_ENABLE_UI": "0",  
        "DOCLING_ARTIFACTS_PATH": ARTIFACTS_PATH,  # Path for caching artifacts
    })
)

@app.function(
    image=image,
    volumes={ARTIFACTS_PATH: artifacts_vol},
    timeout=STARTUP_TIMEOUT,
    # GPU A100 for performance
    gpu="A10",
scaledown_window=600)
@modal.web_server(PORT, startup_timeout=STARTUP_TIMEOUT, label="docling-gpu")
def start_docling_server():
    logger.info("Starting Docling server on port %d", PORT)
    
    # Command to run the server
    cmd = f"docling-serve run --host 0.0.0.0 --port {PORT}" 
    try:
        subprocess.Popen(cmd, shell=True)
        logger.info("Docling server process started")
        return "Starting Docling server..."
    except Exception as e:
        logger.error("Failed to start server: %s", str(e))
        raise

