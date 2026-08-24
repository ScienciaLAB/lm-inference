"""
Batch runner for PaddleOCR PP-StructureV3 (full OCR + table + formula pipeline).

Standalone by design: this file has NO imports from the rest of this repository,
so it can be handed to another person / machine on its own. It only needs
`paddleocr` and a `paddlepaddle` (CPU) or `paddlepaddle-gpu` (GPU) build.

Unlike `paddle_inference_batch.py` (which runs the PP-DocLayout-* *detector*),
PP-StructureV3 is the full recognition pipeline: layout + OCR + table + formula.
It is roughly two orders of magnitude more expensive per page and is intended to
run on GPU. See readme_ppstructurev3.md for setup and measured throughput.

Parallelism model
-----------------
One worker process per GPU (each worker loads its own copy of the pipeline and
is pinned to a single device via `device="gpu:N"`). PDFs are dispatched to
whichever worker is free, so uneven page counts self-balance. The parent process
writes the CSV incrementally, so a crash or Ctrl-C loses at most the in-flight
documents and `--resume` (default) picks up where it stopped.

Usage
-----
    # single GPU
    python ppstructurev3_inference_batch.py /path/to/pdfs -o ./out --devices gpu:0

    # all visible GPUs, one worker each
    python ppstructurev3_inference_batch.py /path/to/pdfs -o ./out --devices auto

    # two workers per GPU (only if VRAM allows; ~8-10 GB per worker)
    python ppstructurev3_inference_batch.py /path/to/pdfs -o ./out \
        --devices auto --workers-per-device 2

    # CPU (very slow: ~147 s/page on a Xeon E5-2650 v4 - for reference only)
    python ppstructurev3_inference_batch.py /path/to/pdfs -o ./out --devices cpu
"""

import argparse
import csv
import os
import queue
import sys
import time
import traceback
from multiprocessing import Manager, set_start_method
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

# Silence PaddleX's model-source reachability probe, which blocks startup on
# hosts without access to the default model mirror.
os.environ.setdefault("PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK", "True")

# Modal GPU pricing, USD per second. Kept in sync with lm_inference_utils.py so
# the cost column is directly comparable to the other systems in the benchmark.
GPU_COST_PER_SECOND = {
    "A10G": 0.000264,
    "A10": 0.000306,
    "L4": 0.000222,
    "L40S": 0.000542,
    "A100_40GB": 0.000583,
    "A100_80GB": 0.000694,
    "H100": 0.001097,
    "H200": 0.001261,
    "B200": 0.001736,
    "T4": 0.000164,
    "none": 0.0,
}

CSV_FIELDS = [
    "document",
    "pages",
    "runtime_sec",
    "sec_per_page",
    "cost_usd",
    "device",
    "success",
    "error",
]

# Per-process globals, set once by _init_worker.
_pipeline = None
_device = None
_opts = None


def _init_worker(device_queue, opts):
    """Runs once per worker process: claim a device, then load the pipeline."""
    global _pipeline, _device, _opts

    _opts = opts
    try:
        _device = device_queue.get_nowait()
    except queue.Empty:
        # More workers than queued devices should be impossible, but fall back
        # to CPU rather than crashing the pool.
        _device = "cpu"

    from paddleocr import PPStructureV3

    t0 = time.time()
    print(f"[worker-{os.getpid()}] loading PP-StructureV3 on {_device} ...", flush=True)

    kwargs = {
        "device": _device,
        # Preprocessing modules that cost time and are not needed for
        # born-digital scientific PDFs.
        "use_doc_orientation_classify": False,
        "use_doc_unwarping": False,
        "use_textline_orientation": False,
        "use_seal_recognition": False,
        "use_table_recognition": not opts["no_table"],
        "use_formula_recognition": not opts["no_formula"],
        "use_chart_recognition": not opts["no_chart"],
    }
    if _device.startswith("cpu"):
        # MKL-DNN + explicit thread count matter a lot on CPU; ignored on GPU.
        kwargs["enable_mkldnn"] = True
        kwargs["cpu_threads"] = opts["cpu_threads"]
    else:
        kwargs["precision"] = opts["precision"]

    _pipeline = PPStructureV3(**kwargs)
    print(
        f"[worker-{os.getpid()}] loaded on {_device} in {time.time() - t0:.1f}s",
        flush=True,
    )

    if opts["warmup"]:
        # First real predict() otherwise pays lazy sub-model init, which would
        # be charged to whichever document happened to land first.
        _warmup()


def _warmup():
    """Run one synthetic page so lazy sub-model init is not billed to a document."""
    import tempfile

    try:
        from PIL import Image, ImageDraw

        img = Image.new("RGB", (1240, 1754), "white")
        draw = ImageDraw.Draw(img)
        draw.text((100, 100), "warmup 123 abc", fill="black")
        with tempfile.TemporaryDirectory() as td:
            p = os.path.join(td, "warmup.png")
            img.save(p)
            t0 = time.time()
            for _ in _pipeline.predict(p):
                pass
        print(f"[worker-{os.getpid()}] warmup {time.time() - t0:.1f}s", flush=True)
    except Exception as e:  # warmup is best-effort
        print(f"[worker-{os.getpid()}] warmup skipped: {e}", flush=True)


def _process_one(pdf_path_str):
    """Process a single PDF end to end. Runs in a worker process."""
    pdf_path = Path(pdf_path_str)
    out_dir = Path(_opts["output_dir"])
    md_path = out_dir / f"{pdf_path.stem}.md"

    t0 = time.time()
    try:
        # PP-StructureV3 rasterizes the PDF internally, so this timing is the
        # true end-to-end per-document cost (rasterization + full recognition).
        results = list(_pipeline.predict(input=str(pdf_path)))

        markdown_pages = []
        images = {}
        for res in results:
            md_info = res.markdown
            markdown_pages.append(md_info)
            if _opts["save_images"]:
                images.update(md_info.get("markdown_images", {}) or {})

        markdown = _pipeline.concatenate_markdown_pages(markdown_pages)
        runtime = time.time() - t0

        md_path.write_text(markdown, encoding="utf-8")

        if _opts["save_images"] and images:
            for rel_path, image in images.items():
                dest = out_dir / pdf_path.stem / rel_path
                dest.parent.mkdir(parents=True, exist_ok=True)
                image.save(dest)

        if _opts["save_json"]:
            json_dir = out_dir / "json" / pdf_path.stem
            json_dir.mkdir(parents=True, exist_ok=True)
            for i, res in enumerate(results, 1):
                res.save_to_json(save_path=str(json_dir / f"page_{i:04d}.json"))

        return {
            "document": pdf_path.name,
            "pages": len(results),
            "runtime_sec": round(runtime, 2),
            "sec_per_page": round(runtime / len(results), 3) if results else 0.0,
            "device": _device,
            "success": True,
            "error": "",
        }

    except Exception as e:
        return {
            "document": pdf_path.name,
            "pages": 0,
            "runtime_sec": round(time.time() - t0, 2),
            "sec_per_page": 0.0,
            "device": _device,
            "success": False,
            "error": f"{type(e).__name__}: {e}",
        }


def resolve_devices(spec, workers_per_device):
    """Expand --devices into one device string per worker process."""
    if spec == "auto":
        try:
            import paddle

            n = paddle.device.cuda.device_count()
        except Exception as e:
            print(f"Could not query GPU count ({e}); falling back to CPU.")
            n = 0
        if n == 0:
            print("No GPU detected - running on CPU. This will be very slow.")
            devices = ["cpu"]
        else:
            devices = [f"gpu:{i}" for i in range(n)]
    else:
        devices = [d.strip() for d in spec.split(",") if d.strip()]

    if "cpu" in devices:
        # Multiple CPU workers oversubscribe the same cores and thrash memory.
        workers_per_device = 1

    return [d for d in devices for _ in range(workers_per_device)]


def load_done(csv_path):
    """Documents already recorded as successful, for --resume."""
    done = set()
    if not Path(csv_path).exists():
        return done
    with open(csv_path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if str(row.get("success", "")).lower() == "true":
                done.add(row["document"])
    return done


def main():
    parser = argparse.ArgumentParser(
        description="Batch PP-StructureV3 inference over a directory of PDFs.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("input_dir", help="Directory containing PDF files")
    parser.add_argument(
        "--output", "-o", default="ppstructurev3_output", help="Output directory"
    )
    parser.add_argument(
        "--csv",
        default=None,
        help="Timing CSV path (default: <output>/ppstructurev3_results.csv)",
    )
    parser.add_argument(
        "--devices",
        default="auto",
        help='Comma-separated devices, e.g. "gpu:0,gpu:1", or "auto" / "cpu" '
        "(default: auto)",
    )
    parser.add_argument(
        "--workers-per-device",
        type=int,
        default=1,
        help="Worker processes per device. >1 needs ~8-10 GB VRAM each (default: 1)",
    )
    parser.add_argument(
        "--precision",
        choices=["fp32", "fp16"],
        default="fp32",
        help="GPU compute precision (default: fp32)",
    )
    parser.add_argument(
        "--cpu-threads",
        type=int,
        default=os.cpu_count(),
        help="Intra-op threads when running on CPU (default: all cores)",
    )
    parser.add_argument(
        "--gpu-type",
        default="none",
        choices=sorted(GPU_COST_PER_SECOND),
        help="Price the run at this GPU's Modal rate (default: none = no cost column)",
    )
    parser.add_argument(
        "--no-table", action="store_true", help="Disable table recognition"
    )
    parser.add_argument(
        "--no-formula", action="store_true", help="Disable formula recognition"
    )
    parser.add_argument(
        "--no-chart",
        action="store_true",
        help="Disable chart recognition (off by default in PP-StructureV3; "
        "this flag keeps it off explicitly)",
    )
    parser.add_argument(
        "--save-images", action="store_true", help="Save figures cropped from the PDF"
    )
    parser.add_argument(
        "--save-json", action="store_true", help="Save per-page structured JSON"
    )
    parser.add_argument(
        "--no-warmup",
        action="store_true",
        help="Skip the synthetic warmup page (warmup keeps lazy model init out "
        "of the first document's timing)",
    )
    parser.add_argument(
        "--limit", type=int, default=None, help="Process at most N documents"
    )
    parser.add_argument(
        "--force", action="store_true", help="Reprocess documents already in the CSV"
    )
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    if not input_dir.is_dir():
        sys.exit(f"Input directory does not exist: {input_dir}")

    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = args.csv or str(out_dir / "ppstructurev3_results.csv")

    pdfs = sorted(input_dir.glob("*.pdf"))
    if not pdfs:
        sys.exit(f"No PDF files found in {input_dir}")

    if not args.force:
        done = load_done(csv_path)
        skipped = len(pdfs)
        pdfs = [p for p in pdfs if p.name not in done]
        skipped -= len(pdfs)
        if skipped:
            print(f"Resuming: skipping {skipped} already-processed documents.")

    if args.limit:
        pdfs = pdfs[: args.limit]

    if not pdfs:
        print("Nothing to process.")
        return 0

    devices = resolve_devices(args.devices, args.workers_per_device)
    rate = GPU_COST_PER_SECOND[args.gpu_type]

    opts = {
        "output_dir": str(out_dir),
        "precision": args.precision,
        "cpu_threads": args.cpu_threads,
        "no_table": args.no_table,
        "no_formula": args.no_formula,
        "no_chart": args.no_chart,
        "save_images": args.save_images,
        "save_json": args.save_json,
        "warmup": not args.no_warmup,
    }

    print(f"Documents to process : {len(pdfs)}")
    print(f"Workers / devices    : {len(devices)} -> {devices}")
    print(f"Output               : {out_dir}")
    print(f"Timing CSV           : {csv_path}")
    if rate:
        print(f"Pricing at           : {args.gpu_type} (${rate}/s per device)")
    print("-" * 70, flush=True)

    manager = Manager()
    device_queue = manager.Queue()
    for d in devices:
        device_queue.put(d)

    write_header = not Path(csv_path).exists()
    totals = {"ok": 0, "fail": 0, "pages": 0, "runtime": 0.0}
    wall_start = time.time()

    with open(csv_path, "a", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=CSV_FIELDS)
        if write_header:
            writer.writeheader()
            csv_file.flush()

        with ProcessPoolExecutor(
            max_workers=len(devices),
            initializer=_init_worker,
            initargs=(device_queue, opts),
        ) as executor:
            futures = {executor.submit(_process_one, str(p)): p for p in pdfs}

            for n, future in enumerate(as_completed(futures), 1):
                pdf = futures[future]
                try:
                    row = future.result()
                except Exception as e:
                    row = {
                        "document": pdf.name,
                        "pages": 0,
                        "runtime_sec": 0.0,
                        "sec_per_page": 0.0,
                        "device": "?",
                        "success": False,
                        "error": f"worker died: {type(e).__name__}: {e}",
                    }

                row["cost_usd"] = round(row["runtime_sec"] * rate, 6) if rate else ""
                writer.writerow(row)
                csv_file.flush()  # crash-safe: every finished doc is on disk

                if row["success"]:
                    totals["ok"] += 1
                    totals["pages"] += row["pages"]
                    totals["runtime"] += row["runtime_sec"]
                    print(
                        f"({n}/{len(pdfs)}) OK  {row['document']} "
                        f"- {row['pages']}p, {row['runtime_sec']}s "
                        f"({row['sec_per_page']} s/page) [{row['device']}]",
                        flush=True,
                    )
                else:
                    totals["fail"] += 1
                    print(
                        f"({n}/{len(pdfs)}) ERR {row['document']} - {row['error']}",
                        flush=True,
                    )

    wall = time.time() - wall_start
    print("-" * 70)
    print("PP-STRUCTUREV3 BATCH SUMMARY")
    print("-" * 70)
    print(f"Succeeded            : {totals['ok']}")
    print(f"Failed               : {totals['fail']}")
    print(f"Total pages          : {totals['pages']}")
    print(f"Wall-clock           : {wall:.1f}s")
    if totals["ok"]:
        print(f"Mean s/doc (serial)  : {totals['runtime'] / totals['ok']:.2f}")
    if totals["pages"]:
        print(f"Mean s/page (serial) : {totals['runtime'] / totals['pages']:.2f}")
        print(f"Throughput           : {totals['pages'] / wall:.2f} pages/s wall-clock")
    if rate:
        # Cost accrues on every provisioned device for the whole wall-clock run,
        # not just while a document is in flight.
        billed = wall * rate * len(devices)
        print(f"Billed (wall x rate x devices) : ${billed:.4f}")
        if totals["ok"]:
            print(f"Per document         : ${billed / totals['ok']:.6f}")
            print(f"Per 1M documents     : ${billed / totals['ok'] * 1e6:,.0f}")
    print(f"CSV                  : {csv_path}")

    return 0 if totals["fail"] == 0 else 1


if __name__ == "__main__":
    try:
        # PaddlePaddle requires spawn; fork corrupts CUDA context in children.
        set_start_method("spawn", force=True)
    except RuntimeError:
        pass
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        print("\nInterrupted. Re-run the same command to resume from the CSV.")
        sys.exit(130)
    except Exception:
        traceback.print_exc()
        sys.exit(1)
