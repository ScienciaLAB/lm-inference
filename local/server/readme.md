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

```


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

## Tracking & Resources
The `/process` endpoint returns resource tracking metrics in the JSON response:
- `processing_time_seconds`: Time taken by the model to process the PDF.
- `throughput_pages_per_second`: The calculated throughput (pages processed per second).
- `system_ram_mb`: Current memory usage (RSS) of the worker handling the request.
- `system_cpu_percent`: CPU utilization of the worker.

## Hardware Specifications (Local Machine)
- **Machine**: MacBook Pro (MacBookPro17,1)
- **Chip**: Apple M1 (ARM)
- **CPU Cores**: 8 (4 Performance and 4 Efficiency)
- **Memory (RAM)**: 16 GB

## Experiment 1: Benchmarking Server Workers
To find the optimum performance on the M1 machine, we track the processing time and RAM usage across different numbers of workers.
When running with `--workers N`, the server will load `N` separate model instances in memory, allowing `N` PDFs to be processed in parallel.

### 1. Start the API Server

```shell
# 1 Worker (1 Model, 1 Parallel PDF)
python -m local.server.paddle_server --workers 1 --port 8080

# 2 Workers (2 Models, 2 Parallel PDFs)
python -m local.server.paddle_server --workers 2 --port 8080

# 4 Workers (4 Models, 4 Parallel PDFs)
python -m local.server.paddle_server --workers 4 --port 8080
```

### 2. Client Command (Process Folder in Parallel)
To test the server, we can send parallel requests from the `pdfs/` folder. The following command uses `xargs` to send 4 requests at a time to the server:

```shell
time find pdfs -name "*.pdf" | xargs -P 4 -I {} curl -s -X POST http://localhost:8080/process -F "file=@{}" -o {}.json
```
*(Adjust `-P 4` to match the number of workers you are testing)*

### 3. Results (Dataset: 100 PDFs, 1003 Pages)

| Workers | Total Real-World Time | Average Throughput | Total Peak RAM (All Workers) | Total Peak CPU (All Workers) |
|---------|-----------------------|--------------------|------------------------------|------------------------------|
| **1 Worker**  | 857.39s (14:17.39) | 1.17 pages/sec | ~2.1 GB | ~42% (out of 800%) |
| **2 Workers** | 540.51s (9:00.51) | 1.86 pages/sec | ~3.7 GB | ~117% (out of 800%) |
| **4 Workers** | 456.97s (7:36.97) | 2.19 pages/sec | ~5.8 GB | ~294% (out of 800%) |
| **6 Workers** | 480.31s (8:00.31) | 2.09 pages/sec | ~8.1 GB | ~470% (out of 800%) |

Throughput is calculated by dividing 1003 total pages by the Total Real-World Time. Total RAM and CPU are calculated by taking the absolute maximum recorded in the output JSON files and multiplying by the number of workers.

---

## Experiment 2: Model Pool (Multiple Models per Worker)

In Experiment 1, each worker process loads exactly **1 model** and processes **1 PDF at a time** (sequential).

In this experiment, we use a different architecture: each worker process loads **M model instances** into a pool. This allows a single worker to process **M PDFs in parallel** without waiting.

**Key difference:**
- **Approach A** (`paddle_server.py`): `N` workers × 1 model each = `N` total models
- **Approach B** (`paddle_server_pool.py`): `N` workers × `M` models each = `N × M` total models

### 1. Start the API Server (Pool variant)

```shell
# 2 Workers, 2 Models each (2 processes, 4 PDFs in parallel)
python -m local.server.paddle_server_pool --workers 2 --models-per-worker 2 --port 8080
```

### 2. Client Command

```shell
time find pdfs -name "*.pdf" | xargs -P <TOTAL_MODELS> -I {} sh -c 'curl -s -X POST http://localhost:8080/process -F "file=@$1" -o "output_json/$(basename "$1").json"' _ {}
```
*(Set `-P` to `workers × models-per-worker` to fully saturate the server)*

### 3. Results (Dataset: 100 PDFs, 1003 Pages)

| Workers | Models | Total Models | Total Real-World Time | Average Throughput | Total Peak RAM | Total Peak CPU |
|---------|---------------|--------------|----------------------|--------------------|-|-|
| **1** | **2** | 2 | 572.79s (9:32.79) | 1.75 pages/sec | ~3.0 GB | ~63% |
| **1** | **4** | 4 | 633.20s (10:33.20) | 1.58 pages/sec | ~3.9 GB | ~100% |
| **2** | **2** | 4 | 543.06s (9:03.06) | 1.85 pages/sec | ~5.2 GB | ~168% |
| **4** | **2** | 8 | 433.07s (7:13.07) | 2.32 pages/sec | ~7.9 GB | ~394% |

Throughput is calculated by dividing 1003 total pages by the Total Real-World Time. Total RAM and CPU are calculated by taking the absolute maximum recorded in the output JSON files and multiplying by the total number of models.
