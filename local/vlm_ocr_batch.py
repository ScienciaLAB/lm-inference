"""
Local batch runner for full-page VLM OCR models: GLM-OCR and FireRed-OCR.

Standalone by design (no imports from the rest of this repository): needs only
`vllm` and `pymupdf`. One vLLM engine is loaded once; each PDF is rasterized
page by page and all pages of a document are submitted to the engine in one
batch, so vLLM's internal scheduler keeps the GPU busy. Output is one Markdown
file per PDF plus a resumable timing CSV with the same schema as the other
runners in this benchmark (document, pages, runtime_sec, sec_per_page, success).

!!! VERIFY BEFORE A REAL RUN !!!
The MODEL_ID and PROMPT of each preset must be checked against the official
model card — OCR VLMs are prompt-sensitive and both models postdate this
script. GLM-OCR in particular ships its own region-routed pipeline
(PP-DocLayout-V3 + parallel region recognition); running it full-page through
vLLM measures the bare model, not that pipeline. If the authors publish an
inference package, prefer it and keep only the CSV/resume conventions here.

Usage:
    python local/vlm_ocr_batch.py /path/to/pdfs -o ./out --model glm-ocr
    python local/vlm_ocr_batch.py /path/to/pdfs -o ./out --model firered-ocr \
        --csv ./firered_results.csv --dpi 200

    # SLURM array sharding: process every Nth document (see slurm/run_vlm_ocr.sbatch)
    python local/vlm_ocr_batch.py /pdfs -o ./out --model glm-ocr --shard 2/4
"""

import argparse
import base64
import csv
import io
import time
from pathlib import Path

PRESETS = {
    "glm-ocr": {
        # VERIFY: HF id from the GLM-OCR technical report (arXiv 2603.10910).
        "model_id": "zai-org/GLM-OCR",
        # VERIFY: replace with the official document-parsing prompt.
        "prompt": "Convert this document page to Markdown. Preserve the reading "
        "order, headings, paragraphs, tables (as Markdown tables) and formulas "
        "(as LaTeX). Output only the Markdown content.",
        "max_tokens": 8192,
    },
    "firered-ocr": {
        # VERIFY: FireRedTeam/FireRed-OCR (arXiv 2603.01840), Qwen3-VL-based, 2B.
        "model_id": "FireRedTeam/FireRed-OCR",
        # VERIFY: replace with the official prompt from the model card.
        "prompt": "Convert this document page to Markdown. Preserve the reading "
        "order, headings, paragraphs, tables (as Markdown tables) and formulas "
        "(as LaTeX). Output only the Markdown content.",
        "max_tokens": 8192,
    },
}

FIELDS = ["document", "pages", "runtime_sec", "sec_per_page", "success"]


def render_pages(pdf_path: Path, dpi: int):
    """Rasterize a PDF to a list of base64-encoded PNG pages."""
    import fitz  # pymupdf

    pages = []
    with fitz.open(pdf_path) as doc:
        for page in doc:
            pix = page.get_pixmap(dpi=dpi)
            buf = io.BytesIO(pix.tobytes("png"))
            pages.append(base64.b64encode(buf.getvalue()).decode("ascii"))
    return pages


def page_messages(b64_png: str, prompt: str):
    return [
        {
            "role": "user",
            "content": [
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/png;base64,{b64_png}"},
                },
                {"type": "text", "text": prompt},
            ],
        }
    ]


def load_done(csv_path: Path):
    done = set()
    if csv_path.exists():
        with open(csv_path, newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                if str(row.get("success", "")).lower() == "true":
                    done.add(row["document"])
    return done


def append_row(csv_path: Path, row: dict):
    write_header = not csv_path.exists()
    with open(csv_path, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS)
        if write_header:
            writer.writeheader()
        writer.writerow(row)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("input_dir", help="Directory containing PDF files")
    ap.add_argument("-o", "--output_dir", required=True, help="Directory for .md outputs")
    ap.add_argument("--model", required=True, choices=sorted(PRESETS))
    ap.add_argument("--csv", default=None, help="Timing CSV (default: <output_dir>/<model>_results.csv)")
    ap.add_argument("--dpi", type=int, default=200)
    ap.add_argument("--shard", default=None, help="K/N: process PDFs with index %% N == K (for SLURM arrays)")
    ap.add_argument("--tensor-parallel-size", type=int, default=1)
    ap.add_argument("--gpu-memory-utilization", type=float, default=0.90)
    ap.add_argument("--max-model-len", type=int, default=16384)
    args = ap.parse_args()

    preset = PRESETS[args.model]
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = Path(args.csv) if args.csv else out_dir / f"{args.model}_results.csv"

    pdfs = sorted(Path(args.input_dir).glob("*.pdf"))
    if args.shard:
        k, n = (int(x) for x in args.shard.split("/"))
        pdfs = [p for i, p in enumerate(pdfs) if i % n == k]
    done = load_done(csv_path)
    pdfs = [p for p in pdfs if p.name not in done and not (out_dir / f"{p.stem}.md").exists()]
    print(f"{len(pdfs)} PDFs to process with {preset['model_id']} -> {out_dir}", flush=True)
    if not pdfs:
        return

    from vllm import LLM, SamplingParams

    llm = LLM(
        model=preset["model_id"],
        trust_remote_code=True,
        tensor_parallel_size=args.tensor_parallel_size,
        gpu_memory_utilization=args.gpu_memory_utilization,
        max_model_len=args.max_model_len,
        limit_mm_per_prompt={"image": 1},
    )
    sampling = SamplingParams(temperature=0.0, max_tokens=preset["max_tokens"])

    for i, pdf in enumerate(pdfs, 1):
        t0 = time.perf_counter()
        try:
            pages = render_pages(pdf, args.dpi)
            batch = [page_messages(b64, preset["prompt"]) for b64 in pages]
            outputs = llm.chat(batch, sampling_params=sampling, use_tqdm=False)
            md = "\n\n".join(o.outputs[0].text.strip() for o in outputs)
            (out_dir / f"{pdf.stem}.md").write_text(md, encoding="utf-8")
            dt = time.perf_counter() - t0
            append_row(csv_path, {
                "document": pdf.name, "pages": len(pages),
                "runtime_sec": round(dt, 2),
                "sec_per_page": round(dt / max(len(pages), 1), 2),
                "success": True,
            })
            print(f"[{i}/{len(pdfs)}] {pdf.name}: {len(pages)} pp in {dt:.1f}s", flush=True)
        except Exception as e:  # noqa: BLE001 — keep the batch going, record the failure
            dt = time.perf_counter() - t0
            append_row(csv_path, {
                "document": pdf.name, "pages": 0,
                "runtime_sec": round(dt, 2), "sec_per_page": 0, "success": False,
            })
            print(f"[{i}/{len(pdfs)}] {pdf.name}: FAILED after {dt:.1f}s: {e}", flush=True)


if __name__ == "__main__":
    main()
