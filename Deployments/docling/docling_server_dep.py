import modal
import os
import subprocess
import logging

# Basic logging for better debugging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(message)s")
logger = logging.getLogger(__name__)

# Modal app
app = modal.App(name="docling-serve")

# Configuration constants
#docling image cpu lighter than GPU
DOCING_IMAGE = "ghcr.io/docling-project/docling-serve-cpu:latest"
PORT = 5001
ARTIFACTS_PATH = "/artifacts"
STARTUP_TIMEOUT = 300  # Seconds for model initialization

# Persistent volume for model artifacts 
artifacts_vol = modal.Volume.from_name("docling-artifacts", create_if_missing=True)

# Build Modal image with environment variables
image = (
    modal.Image.from_registry(DOCING_IMAGE)
    .env({
        "DOCLING_SERVE_ENABLE_UI": "1",  # Enable Gradio UI
        "DOCLING_ARTIFACTS_PATH": ARTIFACTS_PATH,  
        "OMP_NUM_THREADS": "4",  
    })
)

@app.function(
    image=image,
    volumes={ARTIFACTS_PATH: artifacts_vol},
    timeout=STARTUP_TIMEOUT
)
@modal.web_server(PORT, startup_timeout=STARTUP_TIMEOUT, label="docling")
def start_docling_server():
    """Start the Docling server with minimal error handling."""
    logger.info("Starting Docling server on port %d", PORT)
    
    cmd = f"docling-serve run --host 0.0.0.0 --port {PORT} --enable-ui"
    try:
        subprocess.Popen(cmd, shell=True)
        logger.info("Docling server process started")
        return "Starting Docling server..."
    except Exception as e:
        logger.error("Failed to start server: %s", str(e))
        raise

if __name__ == "__main__":
    logger.info("Deploying to Modal")
    app.deploy()