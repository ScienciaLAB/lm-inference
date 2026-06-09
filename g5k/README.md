# Running `paddle_inference_batch` on Grid5000 (Nancy)

`run_paddle_g5k.sh` is an **OAR batch job** that runs the existing pipeline
`python -m local.paddle_inference_batch` over a folder of PDFs on a Grid5000
(Nancy) GPU node, using the `PP-DocLayout-L` model by default.

Grid5000 schedules with **OAR** (not SLURM). The script only *runs* the job — the
Python environment (a conda env named `paddle`) and poppler are assumed to be set up
already; the script does not provision them.

## Prerequisites

- A conda env **`paddle`** with `paddlepaddle-gpu`, `paddleocr`, `pdf2image`, and the
  `poppler` binaries (`pdftoppm`) on PATH. (See `local/readme.md` → *Installation → PaddleOCR (GPU)*.)
- This repo checked out on the site's shared `~/home`, e.g. `~/lm-inference`.
- Input PDFs in a folder (default `<repo>/pdfs`).

## Submit

`-S` makes `oarsub` read the `#OAR` directives from the script; **positional
arguments go after the script path, inside the quotes**:

```bash
# all defaults  (pdfs/ -> output/, PP-DocLayout-L, 8 workers)
oarsub -S ./g5k/run_paddle_g5k.sh

# explicit args
oarsub -S "./g5k/run_paddle_g5k.sh ./pdfs ./output PP-DocLayout-L 8"
```

### Positional parameters

| Pos | Name | Meaning | Default |
|-----|------|---------|---------|
| `$1` | `PDF_DIR` | input folder of PDFs | `<repo>/pdfs` |
| `$2` | `OUT_DIR` | output folder | `<repo>/output` |
| `$3` | `MODEL_NAME` | layout model (see below) | `PP-DocLayout-L` |
| `$4` | `WORKERS` | parallel worker processes | `8` |

They are **positional**, so to set `$4` you must also pass `$1`–`$3`.

#### Model choices (`$3`)

| Model | Notes |
|-------|-------|
| `PP-DocLayout-S` | small — fastest |
| `PP-DocLayout-M` | medium — balanced |
| `PP-DocLayout-L` | large — most accurate (**default**) |
| `PP-DocLayoutV2` | v2 model |
| `PP-DocBlockLayout` | block-level layout |

### OAR directives (edit at the top of the script)

| Directive | Purpose |
|-----------|---------|
| `#OAR -n paddle-ocr` | job name shown in `oarstat` |
| `#OAR -q production` | Nancy's queue for GPU clusters |
| `#OAR -p gpu_count>0 AND gpu_compute_capability_major>=5` | only nodes with a usable GPU |
| `#OAR -l host=1,walltime=10:00:00` | reserve a whole node; max run time |
| `#OAR -O paddle.log` / `-E paddle.log` | stdout / stderr log files |

## Behaviour

- **One worker per GPU (automatic):** when GPUs are present the pipeline ignores
  `WORKERS` and runs **one worker (one model) per GPU** — 1 worker for a single GPU,
  N workers for N GPUs, allocating the models equally. `WORKERS` only applies on
  CPU-only runs (capped at 4). No flag needed.
- **Verbose:** the job passes `--verbose`, so `paddle.log` shows each PDF as a worker
  starts it (in addition to the per-file completion lines).
- **Resumable:** a PDF is skipped when `<OUT_DIR>/<name>.json` already exists, so
  resubmitting after a walltime kill continues where it left off. To reprocess
  everything, add `--force` to the `python -m local.paddle_inference_batch` line.

## Monitor

```bash
oarstat -u                  # your jobs and their state
tail -f paddle.log          # live progress (job context, nvidia-smi, per-file output)
```

## Results

Output lands in `OUT_DIR` (`<repo>/output` by default):

- one aggregated `<document>.json` per PDF (the "done" marker used for resume),
- per page, `res_N.json` (boxes) and `res_N.jpg` (annotated visualization).

## Further options

`run_paddle_g5k.sh` exposes the four most common knobs. The underlying pipeline has
more (`--dpi`, `--only display|paratext|grobid`, `--cleanup-images`, `--force`); add
them to the `python -m local.paddle_inference_batch` line in the script if you need
them. Run `python -m local.paddle_inference_batch --help` for the full list, and see
`local/readme.md` for the output format and filtering details.
