"""
Client for the Baguette-Software-Dataset Modal endpoint
(POST {"paragraphs": [...], "analyze": bool} -> {paragraphs, mentions, record,
duration_seconds, cost_usd}).

Standalone: no repo imports. Resumable CSV, as in deployments/vlm_ocr_client.py.

One input file is one paper. The paragraphs are read according to the extension:
    .xml   TEI (GROBID): the <p> elements of the abstract, body and back
    .json  a list of strings, or an object with a "paragraphs" list
    .txt, .md  paragraphs separated by blank lines

Usage:
    python deployments/baguette/baguette_client.py \
        --input_folder /path/to/tei \
        --output_dir ./baguette_out \
        --csv_output ./baguette_results.csv \
        --endpoint https://<workspace>--baguette-software-dataset-app-extract-endpoint.modal.run \
        --threads 4

`--threads` should not exceed the deployment's `max_containers` (default 4):
each in-flight request occupies one GPU container (max_inputs=1).
"""

import argparse
import csv
import json
import os
import xml.etree.ElementTree as ET
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from threading import Lock

import requests

TEI = "{http://www.tei-c.org/ns/1.0}"
EXTENSIONS = (".xml", ".json", ".txt", ".md")
FIELDS = [
    "document", "paragraphs", "failed_paragraphs", "datasets", "software",
    "runtime_sec", "cost_usd", "success",
]  # fmt: skip
csv_lock = Lock()


def parse_args():
    parser = argparse.ArgumentParser(description="Resumable batch Baguette client")
    parser.add_argument("--input_folder", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--endpoint", required=True, help="Modal endpoint URL")
    parser.add_argument("--csv_output", required=True)
    parser.add_argument("--threads", type=int, default=4,
                        help="In-flight requests; keep <= deployment max_containers")  # fmt: skip
    parser.add_argument("--timeout", type=int, default=1800)
    parser.add_argument("--no_analyze", action="store_true",
                        help="Step 1 only: skip the article-level record")  # fmt: skip
    return parser.parse_args()


def clean(text: str) -> str:
    return " ".join(text.split())


def read_tei(path: Path) -> list[str]:
    root = ET.parse(path).getroot()
    paragraphs = []
    for section in (f".//{TEI}abstract", f"./{TEI}text"):
        for parent in root.iterfind(section):
            for p in parent.iter(f"{TEI}p"):
                paragraphs.append(clean("".join(p.itertext())))
    return paragraphs


def read_json(path: Path) -> list[str]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(data, dict):
        data = data.get("paragraphs")
    if not isinstance(data, list) or not all(isinstance(p, str) for p in data):
        raise ValueError("expected a list of strings or a 'paragraphs' list")
    return [clean(p) for p in data]


def read_text(path: Path) -> list[str]:
    blocks, current = [], []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            current.append(line)
        elif current:
            blocks.append(clean(" ".join(current)))
            current = []
    if current:
        blocks.append(clean(" ".join(current)))
    return blocks


def read_paragraphs(path: Path) -> list[str]:
    suffix = path.suffix.lower()
    if suffix == ".xml":
        paragraphs = read_tei(path)
    elif suffix == ".json":
        paragraphs = read_json(path)
    else:
        paragraphs = read_text(path)
    return [p for p in paragraphs if p]


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


def run_one(args, path: Path):
    out_file = Path(args.output_dir) / f"{path.name}.json"
    row = {
        "document": path.name, "paragraphs": 0, "failed_paragraphs": 0,
        "datasets": 0, "software": 0, "runtime_sec": 0, "cost_usd": 0,
        "success": False,
    }  # fmt: skip
    try:
        paragraphs = read_paragraphs(path)
        if not paragraphs:
            raise ValueError("no paragraph found")
        resp = requests.post(
            args.endpoint,
            json={"paragraphs": paragraphs, "analyze": not args.no_analyze},
            timeout=args.timeout,
        )
        resp.raise_for_status()
        data = resp.json()
        # Keep the text next to its mentions: the endpoint returns indexes only.
        for item in data.get("paragraphs") or []:
            item["text"] = paragraphs[item["index"]]
        out_file.write_text(
            json.dumps(data, ensure_ascii=False, indent=1), encoding="utf-8"
        )
        failed = sum(1 for item in data.get("paragraphs") or [] if "error" in item)
        mentions = data.get("mentions") or {}
        dur = float(data.get("duration_seconds") or 0)
        row.update({
            "paragraphs": len(paragraphs), "failed_paragraphs": failed,
            "datasets": len(mentions.get("datasets") or []),
            "software": len(mentions.get("software") or []),
            "runtime_sec": round(dur, 2), "cost_usd": data.get("cost_usd"),
            "success": True,
        })  # fmt: skip
        append_row(args.csv_output, row)
        message = (
            f"{path.name}: {len(paragraphs)} paragraphs, {row['datasets']} datasets, "
            f"{row['software']} software in {dur:.1f}s"
        )
        if failed:
            message += f" ({failed} paragraphs FAILED)"
        return message
    except Exception as e:  # noqa: BLE001
        append_row(args.csv_output, row)
        return f"{path.name}: FAILED ({e})"


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    done = load_done(args.csv_output)
    files = [
        p
        for p in sorted(Path(args.input_folder).iterdir())
        if p.is_file() and p.suffix.lower() in EXTENSIONS and p.name not in done
    ]
    print(f"{len(files)} documents to process ({len(done)} already done)")
    with ThreadPoolExecutor(max_workers=args.threads) as ex:
        futures = {ex.submit(run_one, args, p): p for p in files}
        for i, fut in enumerate(as_completed(futures), 1):
            print(f"[{i}/{len(files)}] {fut.result()}", flush=True)


if __name__ == "__main__":
    main()
