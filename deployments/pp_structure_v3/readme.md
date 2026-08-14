# PP-StructureV3 — GPU inference (Modal + local)

PP-StructureV3 is PaddleOCR's **full document-parsing pipeline**: layout analysis
+ OCR + table recognition + formula recognition. It is **not** the
`PP-DocLayout-*` layout *detector* used by the hybrid GROBID pipeline in this
repo — it is a peer of MinerU and Docling, and costs roughly two orders of
magnitude more per page.

Measured reference point: **~147 s/page on CPU** (Intel Xeon E5-2650 v4, 12
cores), vs **0.41 s/page** for PP-DocLayout-L on the same machine. CPU is not a
viable path at corpus scale; use a GPU.

## Files

| file | what it does |
|:--|:--|
| `inference_pp_structure_v3.py` | Modal deployment (A100-40GB, one doc per GPU) |
| `run_ppstructurev3_inference.py` | Client that drives the Modal endpoint over a PDF folder |
| `../../local/ppstructurev3_inference_batch.py` | Local runner — multi-GPU or CPU, no Modal |

All three are **standalone** (no imports from this repo), so they can be handed
over on their own.

---

## Option A — Modal

### Deploy

From the `lm-inference/` directory:

```shell
modal deploy deployments/pp_structure_v3/inference_pp_structure_v3.py
```

First deploy builds the image (~10-15 min: the CUDA base plus
`paddlepaddle-gpu` and `paddleocr[all]` are large). The first container start
also downloads the PaddleX sub-models into the `ppstructurev3-paddlex-cache`
volume; later containers reuse it.

### Smoke test

```shell
curl -X POST https://<workspace>--ppstructurev3-app-parse-document-endpoint.modal.run \
  -F 'file=@document.pdf' \
  -F 'output_format=markdown_content'
```

Response: `{"result": ..., "cost_info": {"duration_seconds", "cost_usd", "num_pages", "sec_per_page"}}`.
Output formats: `markdown_content` (default), `json_content`.

### Batch run

```shell
python deployments/pp_structure_v3/run_ppstructurev3_inference.py \
  --pdf_folder  ../data/sources/PMC_1943_lifescience/pdf \
  --output_dir  ./ppstructurev3_out \
  --csv_output  ./ppstructurev3_results.csv \
  --endpoint    https://<workspace>--ppstructurev3-app-parse-document-endpoint.modal.run \
  --threads     4
```

Resumable — re-running skips documents already marked `success=True` in the CSV.

Keep `--threads` ≤ `max_containers` (default 4). The deployment sets
`@modal.concurrent(max_inputs=1)`, so **one document occupies one A100**; extra
threads just queue.

### Cost accounting

The endpoint reports `duration_seconds × $0.000583/s` (A100-40GB), i.e.
**in-request GPU time only**. The Modal dashboard total for the run will be
**higher**, because it also bills container lifetime between requests and during
`scaledown_window`. For a figure to publish, use the **dashboard total ÷ number
of documents** — that is what was actually charged. The client prints both
framings and a reminder.

Set `min_containers=0` if you want to avoid paying for an idle warm container
between runs; the tradeoff is a cold start (model load) on the next request.

---

## Option B — local GPU machine

### Install

```shell
# 1. PaddlePaddle GPU build — NOT on PyPI, use PaddlePaddle's own index.
#    Pick the index matching the host's CUDA version:
#      CUDA 12.6 -> .../stable/cu126/    CUDA 12.9 -> .../stable/cu129/
#      CUDA 11.8 -> .../stable/cu118/
python -m pip install paddlepaddle-gpu==3.2.2 \
    -i https://www.paddlepaddle.org.cn/packages/stable/cu126/

# 2. PaddleOCR with the table/formula extras
python -m pip install "paddleocr[all]>=3.2.0" pdf2image

# 3. System deps
sudo apt-get install -y poppler-utils libgl1 libglib2.0-0
```

Verify:

```shell
python -c "import paddle; paddle.utils.run_check()"
python -c "import paddle; print('GPUs:', paddle.device.cuda.device_count())"
```

`run_check()` must report GPU, not CPU. If it says CPU, the wheel does not match
the host CUDA version — reinstall from the correct index above.

### Run

```shell
# all visible GPUs, one worker per GPU
python local/ppstructurev3_inference_batch.py /path/to/pdfs \
    -o ./ppstructurev3_out --devices auto --gpu-type A100_40GB

# specific GPUs
python local/ppstructurev3_inference_batch.py /path/to/pdfs \
    -o ./out --devices gpu:0,gpu:2

# quick timing sample before committing to a full corpus
python local/ppstructurev3_inference_batch.py /path/to/pdfs \
    -o ./out --devices gpu:0 --limit 50
```

Useful flags:

| flag | effect |
|:--|:--|
| `--workers-per-device N` | N processes per GPU. Each needs ~8-10 GB VRAM — 1 is safe on a 40 GB A100 with room to spare, 2 is usually fine, check `nvidia-smi` |
| `--precision fp16` | Faster; verify output quality before trusting it for a metrics run |
| `--no-formula` / `--no-table` | Disable those sub-pipelines. Big speedup, but then it is no longer a fair MinerU/Docling peer |
| `--save-json` | Per-page structured JSON alongside the markdown |
| `--gpu-type A100_40GB` | Adds a `cost_usd` column at Modal's rate, for comparability |
| `--limit N`, `--force` | Sample run / ignore the resume cache |

Resumable and crash-safe: every finished document is flushed to the CSV
immediately, and re-running the same command continues from there. Ctrl-C is
safe.

### Output

- `<output>/<doc>.md` — concatenated markdown, one file per PDF
- `<output>/ppstructurev3_results.csv` — `document, pages, runtime_sec, sec_per_page, cost_usd, device, success, error`

The summary block at the end prints mean s/doc, mean s/page, wall-clock
throughput, and — with `--gpu-type` — a per-1M-document extrapolation.

---

## Sizing the run before starting it

Do a `--limit 50` run first and read `sec_per_page`, then:

```
wall_hours = total_pages x sec_per_page / (3600 x num_gpus)
```

The evaluation corpora in this repo are **4,538 documents / 64,051 pages**
(life-sci 1,943 docs / 22,409 pp; chem-phy 2,595 docs / 41,642 pp). At a
hypothetical 1 s/page on one GPU that is ~18 hours; at 3 s/page, ~53 hours.
Plan GPU count accordingly.

## Known gotchas

- **`fonts` error — "locations (loca) table missing".** PaddleX's font download
  can land corrupt. Fix: overwrite `~/.paddlex/fonts/PingFang-SC-Regular.ttf`
  with any valid TTF, e.g.
  `cp /usr/share/fonts/truetype/dejavu/DejaVuSans.ttf ~/.paddlex/fonts/PingFang-SC-Regular.ttf`.
  Only affects visualization output, but it raises and fails the document.
- **Multiprocessing start method.** Must be `spawn`; `fork` corrupts the CUDA
  context in children. The local script sets this already.
- **Model source check hangs at startup.** `PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK=True`
  is set by both scripts.
- **`concatenate_markdown_pages` missing.** Requires `paddleocr >= 3.1`. Upgrade.
- **Chart recognition** is off in both scripts (`use_chart_recognition=False`).
  It invokes a chart-to-table VLM and is disproportionately slow. Turn it on
  only if chart content is being evaluated.

## Sources

- [PP-StructureV3 pipeline docs](http://www.paddleocr.ai/main/en/version3.x/pipeline_usage/PP-StructureV3.html)
- [PaddlePaddle Linux pip install](https://www.paddlepaddle.org.cn/documentation/docs/en/install/pip/linux-pip_en.html)
- [PaddleOCR installation guide](https://github.com/PaddlePaddle/PaddleOCR/blob/main/docs/version3.x/installation.en.md)
