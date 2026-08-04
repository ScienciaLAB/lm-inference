# PaddlePaddle Inference Server

A FastAPI service that wraps the `PaddleDocumentProcessor` as an HTTP API. Clients POST a PDF and receive typed-area bounding boxes as JSON.

## Installation

Install the server-specific dependencies (in addition to the PaddleOCR requirements in the root readme):

```shell
pip install -r requirements.txt
```

## Starting the Server


```shell
# Default (PP-DocLayout-L, port 8080)
python -m local.server.paddle_server

# Custom model and port
python -m local.server.paddle_server --model-name PP-DocLayout-L --port 8000

# Multiple workers (requires enough memory for multiple model copies)
python -m local.server.paddle_server --workers 4 --port 8080
```

> **Worker Recommendation:** For optimal throughput, we recommend setting the number of server workers (`--workers`) to match the number of client threads (`--n` in grobid-client) 

## API Endpoints

### `GET /health`
Returns `200 OK` as long as the server process is alive

### `GET /ready`
Returns `200 OK` when the model is loaded. Returns `503` if the model is still loading.

### `POST /process`
Upload a PDF file and receive typed-area bounding boxes as JSON.

**Query Parameters:**
- `filter` (string): Filter mode (`display`, `paratext`, `grobid`). Leave empty for all raw boxes.
- `merge_captions` (boolean): Default `true`. Merge caption boxes with parent figures/tables.

**Example Usage:**
```shell
curl -X POST http://localhost:8080/process \
  -F "file=@document.pdf" \
  -F "filter=grobid"
```

Once running, visit `http://localhost:8080/docs` for the interactive Swagger UI.
