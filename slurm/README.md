# SLURM (DFKI) runners

Array-sharded sbatch wrappers around the local batch runners, one GPU per array
task. Configuration is by environment variable so the same script serves every
corpus:

| Variable | Meaning |
|---|---|
| `MODEL` | `glm-ocr` or `firered-ocr` (run_vlm_ocr.sbatch only) |
| `INPUT` | directory of PDFs |
| `OUTPUT` | output root (markdown + per-shard timing CSVs + model caches) |
| `REPO` | lm-inference checkout (defaults to the script's parent) |
| `VENV` | virtualenv path (created on first run) |

```bash
cd lm-inference && mkdir -p logs
MODEL=glm-ocr INPUT=$DATA/PMC_2595_chem_phy/pdf OUTPUT=$SCRATCH/glmocr \
  sbatch --array=0-3 slurm/run_vlm_ocr.sbatch
INPUT=$DATA/PMC_1943_lifescience/pdf OUTPUT=$SCRATCH/ppsv3 \
  sbatch --array=0-3 slurm/run_ppstructurev3.sbatch
```

Site specifics to adjust before the first submission:

- `--partition` (and `--account` if enforced) in each script's `#SBATCH` header.
- If the cluster mandates enroot/pyxis containers instead of venvs, replace the
  venv block with `srun --container-image=...` and bake the pip installs into
  the image.
- `HF_HOME` / `PADDLE_PDX_CACHE_HOME` default into `$OUTPUT` so model weights
  are shared across array tasks; point them at a persistent scratch path.

Timing caveat for the paper: shard CSVs record per-document wall-times on
whatever GPU the partition provides — costs are NOT comparable to the Modal
A100-40GB billed basis used in the cost table. Use SLURM runs for *quality*
outputs; keep cost measurements on Modal. Merge shard CSVs with a plain
`head -1` + `tail -n +2` concatenation.

Both VLM presets carry `VERIFY` markers (model id + prompt) in
`local/vlm_ocr_batch.py` and the two Modal deployments — check them against the
official model cards before burning GPU hours.
