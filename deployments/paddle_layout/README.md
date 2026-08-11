# PaddlePaddle Layout Detection Modal Deployment

Deployment of PaddleOCR `PP-DocLayout-L` on Modal.com using CPU-only containers.

## Quick Start

```bash
# Deploy (persistent)
modal deploy Deployments/paddle_layout/paddle_layout_inference.py
```

## Endpoints

### `GET /health`

Health check.

```bash
curl https://sciencialab--paddle-layout-cpu-health.modal.run
```

### `POST /parse` — Single PDF

Process one PDF and get bounding boxes.

```bash
curl -X POST https://sciencialab--paddle-layout-cpu-parse.modal.run \
  -F "file=@paper.pdf"
```

**Optional parameters:**

| Field | Type   | Default | Description                                      |
|-------|--------|---------|--------------------------------------------------|
| `dpi` | int    | 72      | Resolution for PDF → image conversion            |
| `only`| string | —       | Filter: `display`, `paratext`, or `grobid`       |

**With filtering:**

```bash
curl -X POST https://sciencialab--paddle-layout-cpu-parse.modal.run \
  -F "file=@paper.pdf" \
  -F "only=grobid" \
  -F "dpi=150"
```


### `POST /batch` — Multiple PDFs

Process multiple PDFs in one request. Files are fanned out to separate containers for parallel processing.

```bash
curl -X POST https://sciencialab--paddle-layout-cpu-batch.modal.run \
  -F "files=@paper1.pdf" \
  -F "files=@paper2.pdf" \
  -F "files=@paper3.pdf"
```


## Container Configuration

| Setting            | Value   | Rationale                                   |
|--------------------|---------|---------------------------------------------|
| CPU                | 8 cores | Layout detection is CPU-bound               |
| Memory             | 16 GB   | Model + PDF images in memory                |
| Max containers     | 10      | Parallel scaling for batch workloads        |
| Concurrent inputs  | 4       | Per-container concurrency                   |

## Filter Types

| Value      | Includes                                                  |
|------------|-----------------------------------------------------------|
| `display`  | figures, images, charts, tables, equations                |
| `paratext` | headers, footers                                          |
| `grobid`   | All of the above (display + paratext)                     |

## Billing Actual Costs

You can view the actual costs for your tests using Modal's built-in CLI:

```bash
modal billing report --for today --show-resources
```

## Performance Benchmarking

We are testing two specific configurations to find the optimal parallel throughput:
1. **Config A:** 4 workers, 1 model per worker (4 models total)
2. **Config B:** 4 workers, 2 models per worker (8 models total)

**Hardware:** 8 CPU cores per container (fixed)
**Dataset:** 100 PDFs folder

### Results (Dataset: 100 PDFs, 1003 Pages, ~10.03 avg pages/doc)

| Config | n_workers | m_parallel | Total Time (`wall_time_s`) | Total Cost |
|--------|-----------|------------|----------------------------|------------|
| A | 4 | 1 | 	279.1s (~4.65 min)	 | $0.21 |
| B | 4 | 2 | 307.4s (~5.12 min) | $0.21 |

### How to run the tests
**Test 1: Config A (4 workers, 1 model each)**
```bash
# 1. Deploy the configuration
export N_WORKERS=4 M_PARALLEL=1
modal deploy Deployments/paddle_layout/paddle_layout_inference.py

# 2. Run the 100 PDFs using the batching python script
time python run_inference.py

# 3. Check the cost
modal billing report --for today
```

**Test 2: Config B (4 workers, 2 models each)**
```bash
# 1. Deploy the configuration
export N_WORKERS=4 M_PARALLEL=2
modal deploy Deployments/paddle_layout/paddle_layout_inference.py

# 2. Run the 100 PDFs using the batching python script

# 3. Check the cost
modal billing report --for today
```

---

## GROBID Evaluation (With and Without PaddlePaddle)

This experiment evaluates the end-to-end processing time of using the `grobid-client-python` library with and without  PaddlePaddle for layout detection (typed areas).

### Configurations Used

- **GROBID Client**: `--n 4`. We use 4 concurrent threads.
- **PaddlePaddle Server (Modal)**: We use the standard server architecture with 4 workers to provide exactly 4 models in memory, perfectly matching the 4 concurrent requests coming from the GROBID client.

### 1. GROBID Baseline (No PaddlePaddle)
```shell
time grobid_client processFulltextDocument \
  --input ./pdfs \
  --output ./test_output_baseline \
  --server https://**********-grobidcrf.hf.space \
  --n 4 \
  --verbose
```

### 2. Benchmark B: GROBID + PaddlePaddle (Modal Endpoint)
Make sure your Modal endpoint is deployed with `export N_WORKERS=4 M_PARALLEL=1` and `modal deploy`.

**Run the client:**
```shell
time grobid_client processFulltextDocument \
  --input ./pdfs \
  --output ./test_output_typed \
  --server https://**********-grobidcrf.hf.space \
  --typed_area \
  --typed_area_server https://sk-datascience--paddle-layout-cpu-parse.modal.run \
  --n 4 \
  --verbose
```

### 3. Results (Dataset: 100 PDFs, 1003 Pages, ~10.03 avg pages/doc)

| Setup | GROBID Client Threads (`--n`) | Paddle Workers (`N_WORKERS`) | Total Time | Speed | Throughput | Cost (Modal) |
|-------|-------------------------------|------------------------------|-----------------------|-------|------------|--------------|
| **GROBID Only** | 4 | N/A | 2m 42.98s (162.02s) | 0.62 docs/sec | 1.62 sec/doc | $0.00 |
| **GROBID + Paddle (Modal)** | 4 | 4 | 6m 23.58s (383.58s) | 0.26 docs/sec | 3.84 sec/doc | 0.18 |
