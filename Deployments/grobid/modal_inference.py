import modal
import requests
import subprocess
import time
import os
import logging
from fastapi import UploadFile

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(message)s")
logger = logging.getLogger(__name__)

# Constants
MINUTES = 60
HOURS = 60 * MINUTES
GROBID_PORT = 8070
GROBID_TIMEOUT = 120
STARTUP_TIMEOUT = 600  # 10 minutes for Grobid startup

app = modal.App("grobid-service")

# GROBID image with necessary dependencies
grobid_image = (
    modal.Image.from_registry("grobid/grobid:0.8.2-full")
    .pip_install(
        "requests",
        "fastapi[standard]",
        "uvicorn[standard]"
    )
    .env({
        "JAVA_OPTS": "-Xmx4g -XX:+UseG1GC",  # Optimize Java memory usage
        "GROBID_HOME": "/opt/grobid"
    })
)

@app.cls(
    image=grobid_image,
    gpu="T4",                    # GPU-enabled for DL models
    memory=8192,                 # 8 GB RAM
    timeout=STARTUP_TIMEOUT,     # Container timeout
    container_idle_timeout=15 * MINUTES,  # Keep warm for 15 minutes
    max_containers=2,            # Limit concurrent instances
    concurrency_limit=4          # Limit concurrent requests per container
)
class GrobidService:
    """GROBID service for academic PDF processing."""

    def __init__(self):
        self.base_url = f"http://localhost:{GROBID_PORT}/api"
        self.proc = None

    def __enter__(self):
        """Initialize Grobid service when container starts."""
        logger.info("Starting GROBID service...")

        try:
            # Start Grobid service
            self.proc = subprocess.Popen(
                [
                    "java",
                    "-Xmx4g",
                    "-XX:+UseG1GC",
                    "-jar",
                    "/opt/grobid/grobid-service/build/libs/grobid-service-0.8.2-onejar.jar"
                ],
                cwd="/opt/grobid",
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                env=os.environ.copy()
            )

            # Wait for Grobid to start with health checks
            if not self._wait_for_startup():
                raise RuntimeError("Failed to start Grobid service within timeout")

            logger.info("GROBID service started successfully")
            return self

        except Exception as e:
            if self.proc:
                self.proc.terminate()
                self.proc.wait()
            raise RuntimeError(f"Failed to initialize GROBID: {str(e)}")

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Cleanup when container shuts down."""
        if self.proc:
            logger.info("Shutting down GROBID service...")
            self.proc.terminate()
            try:
                self.proc.wait(timeout=30)
            except subprocess.TimeoutExpired:
                self.proc.kill()
                self.proc.wait()

    def _wait_for_startup(self, max_attempts: int = 60) -> bool:
        """Wait for GROBID service to be ready."""
        for attempt in range(max_attempts):
            try:
                response = requests.get(
                    f"{self.base_url}/isalive",
                    timeout=5
                )
                if response.ok:
                    logger.info(f"GROBID is ready after {attempt + 1} attempts")
                    return True
            except requests.RequestException:
                pass

            if attempt < max_attempts - 1:
                time.sleep(2)  # Wait 2 seconds between attempts

        return False

    def _make_request(self, endpoint: str, file: UploadFile, additional_params: dict = None):
        """Make a request to GROBID API."""
        try:
            files = {"input": (file.filename, file.file, file.content_type)}
            params = additional_params or {}

            response = requests.post(
                f"{self.base_url}/{endpoint}",
                files=files,
                data=params,
                timeout=GROBID_TIMEOUT
            )

            if response.ok:
                return {
                    "status": "success",
                    "content_type": response.headers.get("content-type", ""),
                    "result": response.text
                }
            else:
                logger.error(f"GROBID API error: {response.status_code} - {response.text}")
                return {
                    "status": "error",
                    "code": response.status_code,
                    "message": response.text
                }

        except requests.RequestException as e:
            logger.error(f"Request failed: {str(e)}")
            return {
                "status": "error",
                "code": 500,
                "message": f"Request failed: {str(e)}"
            }

    @modal.web_endpoint(method="POST", docs=True)
    def parse_pdf(self, file: UploadFile, service: str = "processFulltextDocument"):
        """
        Full-text PDF parsing via GROBID service.

        Args:
            file: PDF file to process
            service: GROBID service to use (processFulltextDocument, processHeaderDocument, etc.)
        """
        return self._make_request(service, file)

    @modal.web_endpoint(method="POST", docs=True)
    def parse_citations(self, file: UploadFile):
        """Extract citations from PDF via GROBID service."""
        return self._make_request("processCitationList", file)

    @modal.web_endpoint(method="POST", docs=True)
    def parse_references(self, file: UploadFile):
        """Extract references from PDF via GROBID service."""
        return self._make_request("processReferences", file)

    @modal.web_endpoint(method="POST", docs=True)
    def parse_header(self, file: UploadFile):
        """Extract header information from PDF via GROBID service."""
        return self._make_request("processHeaderDocument", file)

    @modal.web_endpoint(method="POST", docs=True)
    def parse_raw_text(self, file: UploadFile):
        """Extract raw text from PDF via GROBID service."""
        return self._make_request("processRawtext", file)

    @modal.web_endpoint(method="GET", docs=True)
    def health_check(self):
        """Check if GROBID service is healthy."""
        try:
            response = requests.get(f"{self.base_url}/isalive", timeout=5)
            if response.ok:
                return {
                    "status": "healthy",
                    "service": "grobid",
                    "version": "0.8.2",
                    "port": GROBID_PORT
                }
            else:
                return {
                    "status": "unhealthy",
                    "code": response.status_code,
                    "service": "grobid"
                }
        except Exception as e:
            return {
                "status": "error",
                "message": str(e),
                "service": "grobid"
            }

    @modal.web_endpoint(method="GET", docs=True)
    def get_statistics(self):
        """Get GROBID service statistics."""
        try:
            response = requests.get(f"{self.base_url}/health", timeout=5)
            if response.ok:
                return {
                    "status": "success",
                    "data": response.json()
                }
            else:
                return {
                    "status": "error",
                    "message": "Statistics endpoint not available"
                }
        except Exception as e:
            return {
                "status": "error",
                "message": f"Failed to get statistics: {str(e)}"
            }


# Local testing entrypoint
@app.local_entrypoint()
def main(pdf_path: str, service: str = "processFulltextDocument"):
    """Local testing function for GROBID service."""
    if not os.path.exists(pdf_path):
        logger.error(f"Error: File {pdf_path} does not exist")
        return 1

    try:
        with open(pdf_path, "rb") as f:
            # Create a mock UploadFile for testing
            from fastapi import UploadFile
            from io import BytesIO

            pdf_data = BytesIO(f.read())
            upload_file = UploadFile(
                filename=os.path.basename(pdf_path),
                file=pdf_data,
                content_type="application/pdf"
            )

        logger.info(f"Processing {pdf_path} with service: {service}")

        # Create a temporary service instance for testing
        with GrobidService() as grobid:
            result = grobid._make_request(service, upload_file)

            if result["status"] == "success":
                logger.info("Processing successful!")
                logger.info(f"Content type: {result['content_type']}")
                print("\nResult:")
                print(result["result"])
                return 0
            else:
                logger.error(f"Error processing file: {result.get('message', 'Unknown error')}")
                return 1

    except Exception as e:
        logger.error(f"An error occurred: {str(e)}")
        return 1


# Additional utility functions
def create_sample_pdf():
    """Create a sample PDF for testing (placeholder)."""
    logger.info("Sample PDF creation not implemented - use real PDFs for testing")


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        exit(main(sys.argv[1], sys.argv[2] if len(sys.argv) > 2 else "processFulltextDocument"))
    else:
        logger.error("Usage: python modal_inference.py <pdf_path> [service]")
        exit(1)