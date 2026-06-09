# Running `paddle_inference_batch` on Grid5000 (Nancy)

OAR batch job that runs the existing pipeline
`python -m local.paddle_inference_batch` over a folder of PDFs on a Grid5000
(Nancy) node, using the `PP-DocLayout-L` model by default.

Grid5000 schedules with **OAR** (not SLURM). The environment (the `paddle` conda env
and poppler) is assumed to be set up already — this script does not provision it.

## Submit

The script takes positional arguments (all optional, with defaults):

| Pos | Meaning | Default |
|-----|---------|---------|
| `$1` | input PDF dir | `<repo>/pdfs` |
| `$2` | output dir | `<repo>/output` |
| `$3` | model name | `PP-DocLayout-L` |
| `$4` | workers | `8` |

`-S` reads the `#OAR` directives from the script (queue, resources, logs); arguments
go after the script path, inside the quotes:

```bash
# all defaults
oarsub -S ./g5k/run_paddle_g5k.sh

# explicit args
oarsub -S "./g5k/run_paddle_g5k.sh ./pdfs ./output PP-DocLayout-L 8"
```

Resources, queue, and log files are set by the `#OAR` directives at the top of
`run_paddle_g5k.sh` (`-q production`, `-l host=1,walltime=10:00:00`, logs to
`paddle.log`). Edit those lines to change walltime or request a GPU
(`-l host=1/gpu=1,...`).

## Monitor

```bash
oarstat -u                  # your jobs and their state
tail -f paddle.log          # live progress (job context, nvidia-smi, per-file output)
```

## Results

Output lands in the chosen output dir (`<repo>/output` by default):
- one aggregated `<document>.json` per PDF,
- per page, `res_N.json` (boxes) and `res_N.jpg` (annotated visualization).

Already-processed PDFs (where `<name>.json` exists) are skipped automatically; pass
`--force` by editing the `python -m local.paddle_inference_batch` line if you need to
reprocess.
