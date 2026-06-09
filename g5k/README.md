# Running PaddleOCR `PP-DocLayout-L` on Grid5000 (Nancy)

Wrappers to run the existing batch pipeline
(`python -m local.paddle_inference_batch`) unattended on a Grid5000 GPU node,
using the **large** layout model `PP-DocLayout-L`.

Grid5000 schedules with **OAR** (not SLURM): you reserve a GPU with `oarsub` and
Nancy exposes GPU nodes through the `production` queue.

## Files

| File | Purpose |
|------|---------|
| `setup_env_g5k.sh` | One-time: create a venv + install dependencies. |
| `run_paddle_g5k.sh` | OAR passive batch job (has `#OAR` directives). |

## 1. Connect and get the code

```bash
ssh access.grid5000.fr        # from your laptop
ssh nancy                     # hop to the Nancy frontend

git clone <this-repo-url> ~/lm-inference
cd ~/lm-inference
mkdir -p pdfs && cp /path/to/your/*.pdf pdfs/   # input PDFs
```

`~/home` is shared (NFS) across the Nancy frontend and its compute nodes, so the
venv and data you create here are visible to the job.

## 2. One-time environment setup

```bash
bash g5k/setup_env_g5k.sh                 # CPU paddlepaddle wheel (repo default)
# or, for real GPU acceleration (CUDA wheel):
GPU_WHEEL=1 bash g5k/setup_env_g5k.sh
```

Creates a venv at `~/venvs/paddle-g5k` and installs
`local/requirement.paddle.ladas.txt`.

> **poppler:** `pdf2image` needs the `pdftoppm` binary. The setup script checks for
> it and, if missing, prints how to get it without root (`module load poppler`,
> `conda install -c conda-forge poppler`, `guix install poppler`, or a `sudo-g5k`
> deploy job). Fix that before submitting, or conversion will fail.

## 3. Submit the job

```bash
oarsub -S ./g5k/run_paddle_g5k.sh
```

The `#OAR` directives inside the script request **1 host / 1 GPU for 2h** on the
`production` queue and write logs to `paddle.<jobid>.stdout` / `.stderr`. Edit the
`walltime` / `gpu` line in the script to change resources.

## 4. Monitor

```bash
oarstat -u                       # your jobs and their state
tail -f paddle.<jobid>.stdout    # live progress (nvidia-smi + per-file output)
```

## 5. Results

Output lands in `~/lm-inference/results/`:
- one aggregated `<document>.json` per PDF,
- per page, `res_N.json` (boxes) and `res_N.jpg` (annotated visualization).

## Equivalent bare command

The job runs, parameterized, exactly:

```bash
python -m local.paddle_inference_batch ./pdfs -o ./results \
    --model-name PP-DocLayout-L --workers 4
```

## Overriding defaults

`oarsub` does **not** forward your shell environment to the job, so the cleanest way
to change `PDF_DIR`, `OUT_DIR`, `MODEL_NAME`, or `WORKERS` is to edit the defaults at
the top of `run_paddle_g5k.sh`. For a quick one-off without editing, run an
interactive job and call the module directly:

```bash
oarsub -I -q production -l host=1/gpu=1,walltime=0:30:00
source ~/venvs/paddle-g5k/bin/activate
cd ~/lm-inference
python -m local.paddle_inference_batch ./pdfs -o ./results \
    --model-name PP-DocLayout-M --workers 2
```
