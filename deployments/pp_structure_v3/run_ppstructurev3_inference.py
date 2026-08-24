"""
Client for the PP-StructureV3 Modal endpoint.

Standalone: no repo imports, so it can be handed over alongside the deployment.
Equivalent to evaluation/pmc/inference/run_mineru_inference.py but without the
`grobid_alignment_utils` dependency, and it also records page counts.

Usage:
    python run_ppstructurev3_inference.py \
        --pdf_folder /path/to/pdfs \
        --output_dir ./ppstructurev3_out \
        --csv_output ./ppstructurev3_results.csv \
        --endpoint https://<workspace>--ppstructurev3-app-parse-document-endpoint.modal.run \
        --threads 4

`--threads` should not exceed `max_containers` in the deployment (default 4):
each in-flight request occupies one GPU container.
"""

import argparse
import csv
import json
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from threading import Lock

import requests

parser = argparse.ArgumentParser(description="Resumable batch PP-StructureV3 client")
parser.add_argument("--pdf_folder", required=True, help="Folder with PDF files")
parser.add_argument("--output_dir", required=True, help="Directory to save outputs")
parser.add_argument("--endpoint", required=True, help="Modal endpoint URL")
parser.add_argument("--csv_output", required=True, help="Output CSV path")
parser.add_argument(
    "--output_format",
    default="markdown_content",
    choices=["markdown_content", "json_content"],
    help="markdown_content (.md) or json_content (.json)",
)
parser.add_argument(
    "--threads",
    type=int,
    default=4,
    help="Concurrent in-flight requests; keep <= deployment max_containers",
)
parser.add_argument("--timeout", type=int, default=1800, help="Per-request timeout (s)")
args = parser.parse_args()

os.makedirs(args.output_dir, exist_ok=True)
EXT = ".md" if args.output_format == "markdown_content" else ".json"
FIELDS = ["document", "pages", "runtime_sec", "sec_per_page", "cost_usd", "success"]

csv_lock = Lock()


def load_done(csv_path):
    done = set()
    if Path(csv_path).exists():
        with open(csv_path, newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                if str(row.get("success", "")).lower() == "true":
                    done.add(row["document"])
    return done


def append_row(csv_path, row):
    with csv_lock:
        write_header = not Path(csv_path).exists()
        with open(csv_path, "a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=FIELDS)
            if write_header:
                writer.writeheader()
            writer.writerow(row)


def run_one(pdf_path):
    out_file = Path(args.output_dir) / f"{pdf_path.stem}{EXT}"
    try:
        with open(pdf_path, "rb") as f:
            response = requests.post(
                args.endpoint,
                files={"file": (pdf_path.name, f, "application/pdf")},
                data={"output_format": args.output_format},
                timeout=args.timeout,
            )

        if response.status_code != 200:
            print(
                f"[FAIL] {pdf_path.name}: HTTP {response.status_code} {response.text[:200]}"
            )
            return {
                "document": pdf_path.name,
                "pages": 0,
                "runtime_sec": 0.0,
                "sec_per_page": 0.0,
                "cost_usd": 0.0,
                "success": False,
            }

        payload = response.json()
        cost = payload.get("cost_info", {})
        result = payload.get("result")

        if args.output_format == "markdown_content":
            out_file.write_text(result or "", encoding="utf-8")
        else:
            out_file.write_text(json.dumps(result, indent=2), encoding="utf-8")

        row = {
            "document": pdf_path.name,
            "pages": cost.get("num_pages", 0),
            "runtime_sec": round(float(cost.get("duration_seconds", 0.0)), 2),
            "sec_per_page": cost.get("sec_per_page", 0.0),
            "cost_usd": round(float(cost.get("cost_usd", 0.0)), 6),
            "success": True,
        }
        print(
            f"[OK] {pdf_path.name} - {row['pages']}p, {row['runtime_sec']}s, "
            f"${row['cost_usd']:.6f}"
        )
        return row

    except Exception as e:
        print(f"[ERROR] {pdf_path.name}: {e}")
        return {
            "document": pdf_path.name,
            "pages": 0,
            "runtime_sec": 0.0,
            "sec_per_page": 0.0,
            "cost_usd": 0.0,
            "success": False,
        }


pdf_files = sorted(Path(args.pdf_folder).glob("*.pdf"))
done = load_done(args.csv_output)
to_process = [p for p in pdf_files if p.name not in done]

if done:
    print(f"Resuming: {len(done)} already done, {len(to_process)} remaining.")
print(f"\n>>> Processing {len(to_process)} documents with {args.threads} threads...\n")

totals = {"runtime": 0.0, "cost": 0.0, "pages": 0, "ok": 0}

with ThreadPoolExecutor(max_workers=args.threads) as pool:
    futures = {pool.submit(run_one, pdf): pdf for pdf in to_process}
    for future in as_completed(futures):
        row = future.result()
        # Record failures too, so the CSV is a complete run log; --resume only
        # skips rows marked success=True.
        append_row(args.csv_output, row)
        if row["success"]:
            totals["ok"] += 1
            totals["runtime"] += row["runtime_sec"]
            totals["cost"] += row["cost_usd"]
            totals["pages"] += row["pages"]

n = len(to_process)
print(f"\nDone. CSV: {args.csv_output}")
print(f"Processed {n}: {totals['ok']} succeeded, {n - totals['ok']} failed")
if totals["ok"]:
    print(f"Mean runtime : {totals['runtime'] / totals['ok']:.2f} s/doc")
if totals["pages"]:
    print(f"Mean per page: {totals['runtime'] / totals['pages']:.2f} s/page")
print(f"Total cost   : ${totals['cost']:.4f}")
if totals["ok"]:
    per_doc = totals["cost"] / totals["ok"]
    print(f"Per document : ${per_doc:.6f}")
    print(f"Per 1M docs  : ${per_doc * 1e6:,.0f}")
print(
    "\nNote: this sums per-request billed time. Compare against the Modal "
    "dashboard total for the run - the dashboard also charges container "
    "lifetime between requests.\n"
)
