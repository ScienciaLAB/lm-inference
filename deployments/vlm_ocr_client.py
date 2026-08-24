"""
Client for the GLM-OCR and FireRed-OCR Modal endpoints (they share one contract:
POST multipart 'file' -> {markdown_content, num_pages, duration_seconds, cost_usd}).

Standalone: no repo imports. Same resumable CSV schema as
deployments/pp_structure_v3/run_ppstructurev3_inference.py, so the results drop
straight into the existing cost/quality tooling.

Usage:
    python deployments/vlm_ocr_client.py \
        --pdf_folder /path/to/pdfs \
        --output_dir ./glmocr_out \
        --csv_output ./glmocr_results.csv \
        --endpoint https://<workspace>--glm-ocr-app-parse-document-endpoint.modal.run \
        --threads 4

`--threads` should not exceed the deployment's `max_containers` (default 4):
each in-flight request occupies one GPU container (max_inputs=1).
"""

import argparse
import csv
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from threading import Lock

import requests

parser = argparse.ArgumentParser(description="Resumable batch VLM-OCR client")
parser.add_argument("--pdf_folder", required=True)
parser.add_argument("--output_dir", required=True)
parser.add_argument("--endpoint", required=True, help="Modal endpoint URL")
parser.add_argument("--csv_output", required=True)
parser.add_argument("--threads", type=int, default=4,
                    help="In-flight requests; keep <= deployment max_containers")
parser.add_argument("--timeout", type=int, default=1800)
args = parser.parse_args()

os.makedirs(args.output_dir, exist_ok=True)
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


def append_row(row):
    with csv_lock:
        write_header = not Path(args.csv_output).exists()
        with open(args.csv_output, "a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=FIELDS)
            if write_header:
                writer.writeheader()
            writer.writerow(row)


def run_one(pdf_path: Path):
    out_file = Path(args.output_dir) / f"{pdf_path.stem}.md"
    try:
        with open(pdf_path, "rb") as f:
            resp = requests.post(
                args.endpoint,
                files={"file": (pdf_path.name, f, "application/pdf")},
                timeout=args.timeout,
            )
        resp.raise_for_status()
        data = resp.json()
        out_file.write_text(data["markdown_content"], encoding="utf-8")
        pages = int(data.get("num_pages") or 0)
        dur = float(data.get("duration_seconds") or 0)
        append_row({
            "document": pdf_path.name, "pages": pages,
            "runtime_sec": round(dur, 2),
            "sec_per_page": round(dur / max(pages, 1), 2),
            "cost_usd": data.get("cost_usd"), "success": True,
        })
        return f"{pdf_path.name}: {pages} pp in {dur:.1f}s"
    except Exception as e:  # noqa: BLE001
        append_row({
            "document": pdf_path.name, "pages": 0, "runtime_sec": 0,
            "sec_per_page": 0, "cost_usd": 0, "success": False,
        })
        return f"{pdf_path.name}: FAILED ({e})"


def main():
    done = load_done(args.csv_output)
    pdfs = [p for p in sorted(Path(args.pdf_folder).glob("*.pdf")) if p.name not in done]
    print(f"{len(pdfs)} PDFs to process ({len(done)} already done)")
    with ThreadPoolExecutor(max_workers=args.threads) as ex:
        futures = {ex.submit(run_one, p): p for p in pdfs}
        for i, fut in enumerate(as_completed(futures), 1):
            print(f"[{i}/{len(pdfs)}] {fut.result()}", flush=True)


if __name__ == "__main__":
    main()
