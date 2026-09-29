"""
Client for the Baguette-Software-Dataset Modal endpoint.

Standalone: no repo imports. The client sends the input as it is; the server
reads it, splits it into paragraphs and splits the long paragraphs.

One input is one paper:
    --text          a text, given on the command line
    --input_file    one or more files
    --input_folder  all the files of a folder, with a resumable CSV, as in
                    deployments/vlm_ocr_client.py

File types, by extension:
    .txt, .md  text; the paragraphs are the blocks separated by blank lines
    .xml       TEI (GROBID): the <p> elements of the abstract, body and back
    .json      a list of strings, or an object with 'paragraphs' or 'text'

Usage:
    python deployments/baguette/baguette_client.py --endpoint <url> \
        --text "The data was acquired using the ResearchIR MAX 4.0 software."

    python deployments/baguette/baguette_client.py --endpoint <url> \
        --input_file paper.txt

    python deployments/baguette/baguette_client.py --endpoint <url> \
        --input_folder /path/to/papers \
        --output_dir ./baguette_out \
        --csv_output ./baguette_results.csv \
        --threads 4

<url> is https://<workspace>--baguette-software-dataset-app-extract-endpoint.modal.run

Without --output_dir the result is printed. `--threads` should not exceed the
deployment's `max_containers` (default 4): each in-flight request occupies one
GPU container (max_inputs=1).
"""

import argparse
import csv
import json
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from threading import Lock

import requests

EXTENSIONS = (".txt", ".md", ".xml", ".json")
FIELDS = [
    "document", "paragraphs", "failed_paragraphs", "datasets", "software",
    "runtime_sec", "cost_usd", "success",
]  # fmt: skip
csv_lock = Lock()


def parse_args():
    parser = argparse.ArgumentParser(description="Baguette client")
    parser.add_argument("--endpoint", required=True, help="Modal endpoint URL")
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--text", help="Text of one paper")
    source.add_argument("--input_file", nargs="+", help="One file per paper")
    source.add_argument("--input_folder", help="Folder with one file per paper")
    parser.add_argument("--output_dir", help="One JSON file per paper; default: print")
    parser.add_argument("--csv_output", help="CSV summary; makes the run resumable")
    parser.add_argument("--threads", type=int, default=4,
                        help="In-flight requests; keep <= deployment max_containers")  # fmt: skip
    parser.add_argument("--timeout", type=int, default=1800)
    parser.add_argument("--retries", type=int, default=3,
                        help="Retries of a paper after a server or network error")  # fmt: skip
    parser.add_argument("--retry_wait", type=int, default=20,
                        help="Seconds before the first retry; doubled at each retry")  # fmt: skip
    parser.add_argument("--no_analyze", action="store_true",
                        help="Step 1 only: skip the article-level record")  # fmt: skip
    args = parser.parse_args()
    if args.input_folder and not args.output_dir:
        parser.error("--input_folder requires --output_dir")
    return args


def load_done(csv_path):
    done = set()
    if csv_path and Path(csv_path).exists():
        with open(csv_path, newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                if str(row.get("success", "")).lower() == "true":
                    done.add(row["document"])
    return done


def append_row(csv_path, row):
    if not csv_path:
        return
    with csv_lock:
        write_header = not Path(csv_path).exists()
        with open(csv_path, "a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=FIELDS)
            if write_header:
                writer.writeheader()
            writer.writerow(row)


def send(args, path: Path | None):
    """Send one paper: the file at `path`, or the text of the command line."""
    analyze = "false" if args.no_analyze else "true"
    if path is None:
        return requests.post(
            args.endpoint,
            json={"text": args.text, "analyze": not args.no_analyze},
            timeout=args.timeout,
        )
    with open(path, "rb") as f:
        return requests.post(
            args.endpoint,
            files={"file": (path.name, f)},
            data={"analyze": analyze},
            timeout=args.timeout,
        )


def send_with_retries(args, path: Path | None):
    """Send one paper, again after a server error (HTTP 5xx) or a network
    error: a cold start, or containers of the previous version still serving
    during a redeploy. An input error (HTTP 4xx) is not retried."""
    for attempt in range(args.retries + 1):
        try:
            resp = send(args, path)
            error = f"HTTP {resp.status_code}: {resp.text[:300]}"
            if resp.status_code < 500:
                return resp
        except requests.RequestException as e:
            error = str(e)
        if attempt < args.retries:
            wait = args.retry_wait * 2**attempt
            name = path.name if path else "text"
            print(f"{name}: {error}; retry in {wait}s", file=sys.stderr, flush=True)
            time.sleep(wait)
    raise RuntimeError(error)


def run_one(args, path: Path | None) -> tuple[bool, str]:
    name = path.name if path else "text"
    row = {
        "document": name, "paragraphs": 0, "failed_paragraphs": 0,
        "datasets": 0, "software": 0, "runtime_sec": 0, "cost_usd": 0,
        "success": False,
    }  # fmt: skip
    try:
        resp = send_with_retries(args, path)
        if resp.status_code != 200:
            raise RuntimeError(f"HTTP {resp.status_code}: {resp.text[:300]}")
        data = resp.json()
        output = json.dumps(data, ensure_ascii=False, indent=1)
        if args.output_dir:
            out_file = Path(args.output_dir) / f"{name}.json"
            out_file.write_text(output, encoding="utf-8")
        else:
            print(output)
        items = data.get("paragraphs") or []
        errors = [item["error"] for item in items if "error" in item]
        failed = len(errors)
        mentions = data.get("mentions") or {}
        dur = float(data.get("duration_seconds") or 0)
        row.update({
            "paragraphs": len(items), "failed_paragraphs": failed,
            "datasets": len(mentions.get("datasets") or []),
            "software": len(mentions.get("software") or []),
            "runtime_sec": round(dur, 2), "cost_usd": data.get("cost_usd"),
            "success": True,
        })  # fmt: skip
        append_row(args.csv_output, row)
        message = (
            f"{name}: {len(items)} paragraphs, {row['datasets']} datasets, "
            f"{row['software']} software in {dur:.1f}s"
        )
        if failed:
            message += f" ({failed} paragraphs FAILED: {errors[0][:100]})"
        return True, message
    except Exception as e:  # noqa: BLE001
        append_row(args.csv_output, row)
        return False, f"{name}: FAILED ({e})"


def main():
    args = parse_args()
    if args.output_dir:
        os.makedirs(args.output_dir, exist_ok=True)

    if args.text is not None:
        papers = [None]
    elif args.input_file:
        papers = [Path(p) for p in args.input_file]
    else:
        papers = [
            p
            for p in sorted(Path(args.input_folder).iterdir())
            if p.is_file() and p.suffix.lower() in EXTENSIONS
        ]
    done = load_done(args.csv_output)
    papers = [p for p in papers if p is None or p.name not in done]

    # Progress goes to stderr, so that a printed result can be piped.
    print(f"{len(papers)} documents to process ({len(done)} already done)", file=sys.stderr)
    failures = 0
    with ThreadPoolExecutor(max_workers=args.threads) as ex:
        futures = {ex.submit(run_one, args, p): p for p in papers}
        for i, fut in enumerate(as_completed(futures), 1):
            success, message = fut.result()
            failures += not success
            print(f"[{i}/{len(papers)}] {message}", file=sys.stderr, flush=True)
    sys.exit(1 if failures else 0)


if __name__ == "__main__":
    main()
