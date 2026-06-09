# Local Inference Scripts

Scripts for running layout detection inference locally using PaddleOCR and LADaS (YOLO-based) models, plus caption-to-figure/table linking and evaluation.

All scripts are run as modules from the `lm-inference/` root directory using `python -m local.<script>`.

## Table of Contents

- [Installation](#installation)
  - [PaddleOCR (CPU)](#paddleocr-cpu)
  - [PaddleOCR (GPU)](#paddleocr-gpu)
  - [LADaS (YOLO)](#ladas-yolo)
- [Paddle Inference](#paddle-inference)
  - [Single Document Processing](#single-document-processing)
  - [Batch Processing (PDFs)](#batch-processing-pdfs)
  - [Batch Processing (Images)](#batch-processing-images)
- [LADaS Inference](#ladas-inference)
  - [Single Document Processing](#single-document-processing-1)
  - [Batch Processing (Images)](#batch-processing-images-1)
- [Caption Merging](#caption-merging)
  - [Strategies](#strategies)
  - [Evaluation](#evaluation)
- [Output Format](#output-format)
- [Filtering Options](#filtering-options)

---

## Installation

### PaddleOCR (CPU)

Reference: <https://www.paddlepaddle.org.cn/documentation/docs/en/install/pip/macos-pip_en.html>

```shell
pip install paddlepaddle==3.2.0 -i https://www.paddlepaddle.org.cn/packages/stable/cpu/
pip install -U "paddleocr[doc-parser]"
pip install pdf2image
```

### PaddleOCR (GPU)

```shell
pip install paddlepaddle-gpu==3.2.0 -i https://www.paddlepaddle.org.cn/packages/stable/cu126/
pip install -U "paddleocr[doc-parser]"
pip install pdf2image
```

> **Note:** You also need to install `poppler` for PDF to image conversion:
>
> - macOS: `brew install poppler`
> - Ubuntu: `apt-get install poppler-utils`
> - Windows: Download from [poppler releases](https://github.com/oschwartz10612/poppler-windows/releases)

### LADaS (YOLO)

LADaS uses Ultralytics YOLO for layout detection. Install the required dependencies:

```shell
pip install ultralytics
pip install pdf2image
```

---

## Paddle Inference

### Single Document Processing

Use `paddle_inference` to process a single PDF document with PaddleOCR layout detection.

```shell
python -m local.paddle_inference <input_pdf> [options]
```

#### Arguments

| Argument | Description | Default |
|----------|-------------|---------|
| `input` | Input PDF document path (required) | - |
| `--output`, `-o` | Output directory for processed images and results | `output` |
| `--model-name` | Model name for layout detection | `PP-DocLayout-S` |
| `--dpi` | DPI for PDF to image conversion | `72` |
| `--temp-dir` | Temporary directory for processing | auto-generated |
| `--only` | Filter output to specific element types | - |
| `--cleanup-images` | Clean up intermediate files, keep only aggregated JSON | `False` |

#### Available Models

- `PP-DocLayout-S` - Small model (fastest, default)
- `PP-DocLayout-M` - Medium model (balanced)
- `PP-DocLayout-L` - Large model (most accurate)
- `PP-DocLayoutV2` - Version 2 model
- `PP-DocBlockLayout` - Block-level layout model

#### Examples

```shell
python -m local.paddle_inference document.pdf
python -m local.paddle_inference document.pdf --output ./results --model-name PP-DocLayout-L
python -m local.paddle_inference document.pdf -o ./results --only display --cleanup-images
python -m local.paddle_inference document.pdf --dpi 150 --model-name PP-DocLayout-M
```

---

### Batch Processing (PDFs)

Use `paddle_inference_batch` to process multiple PDF documents in a directory with parallel workers. Already-processed files (where `<name>.json` exists in the output directory) are skipped automatically.

```shell
python -m local.paddle_inference_batch <input_dir> [options]
```

#### Arguments

| Argument | Description | Default |
|----------|-------------|---------|
| `input_dir` | Input directory containing PDF files (required) | - |
| `--output`, `-o` | Output directory for results | `output` |
| `--model-name` | Model name for layout detection | `PP-DocLayout-S` |
| `--dpi` | DPI for PDF to image conversion | `72` |
| `--workers`, `-w` | Number of parallel workers | CPU count (max 4) |
| `--temp-dir` | Temporary directory for processing | auto-generated |
| `--only` | Filter output to specific element types | - |
| `--cleanup-images` | Clean up intermediate files | `False` |
| `--force` | Reprocess all files even if output already exists | `False` |

#### Examples

```shell
python -m local.paddle_inference_batch ./input_pdfs --output ./results
python -m local.paddle_inference_batch ./input_pdfs -o ./results --workers 4
python -m local.paddle_inference_batch ./input_pdfs -o ./results --model-name PP-DocLayout-L --dpi 150 --only grobid --cleanup-images
python -m local.paddle_inference_batch ./input_pdfs -o ./results --force
```

---

### Batch Processing (Images)

Use `paddle_img_inference_batch` to process multiple PNG/JPG files with parallel workers. Already-processed files are skipped automatically.

```shell
python -m local.paddle_img_inference_batch <input_dir> [options]
```

#### Arguments

| Argument | Description | Default |
|----------|-------------|---------|
| `input_dir` | Input directory containing PNG/JPG files (required) | - |
| `--output`, `-o` | Output directory for results | `output` |
| `--model-name` | Model name for layout detection | `PP-DocLayout-S` |
| `--workers`, `-w` | Number of parallel workers | CPU count |
| `--force` | Reprocess all files even if output already exists | `False` |

#### Examples

```shell
python -m local.paddle_img_inference_batch ./images -o ./results --model-name PP-DocLayout-M --workers 4
python -m local.paddle_img_inference_batch ./images -o ./results --force
```

---

## LADaS Inference

### Single Document Processing

Use `ladas_inference` to process PDF documents using the LADaS YOLO-based layout detection model.

```shell
python -m local.ladas_inference <input_pdf> --model-file <model_path> [options]
```

#### Arguments

| Argument | Description | Default |
|----------|-------------|---------|
| `input` | Input PDF document path (required) | - |
| `--model-file` | Path to YOLO model file (required) | - |
| `--output`, `-o` | Output directory for results | `output` |
| `--dpi` | DPI for PDF to image conversion | `72` |
| `--temp-dir` | Temporary directory for processing | auto-generated |
| `--only` | Filter output to specific element types | - |
| `--cleanup-images` | Clean up intermediate files | `False` |

#### Examples

```shell
python -m local.ladas_inference document.pdf --model-file ./LADaS/model-train-test.pt
python -m local.ladas_inference document.pdf --model-file ./model.pt -o ./results --only display
python -m local.ladas_inference document.pdf --model-file ./model.pt -o ./results --dpi 100 --only grobid --cleanup-images
```

---

### Batch Processing (Images)

Use `ladas_img_inference_batch` to process multiple PNG/JPG files with parallel workers. Already-processed files are skipped automatically.

```shell
python -m local.ladas_img_inference_batch <input_dir> --model-file <model_path> [options]
```

#### Arguments

| Argument | Description | Default |
|----------|-------------|---------|
| `input` | Folder containing PNG/JPG images (required) | - |
| `--model-file` | Path to .pt model file (required) | - |
| `--output`, `-o` | Output directory | `output_ladas` |
| `--workers`, `-w` | Number of threads | CPU count - 1 |
| `--force` | Reprocess all files even if output already exists | `False` |

#### Examples

```shell
python -m local.ladas_img_inference_batch ./images --output ./results/LADaS --model-file ./LADaS/model-train-test.pt --workers 4
python -m local.ladas_img_inference_batch ./images -o ./results --model-file ./model.pt --force
```

---

## Caption Merging

`caption_merging` filters raw PaddleOCR output to retain only **figures**, **tables**, and **paratext** (headers, footers, page numbers), discarding all other element types (text, titles, etc.). It then links detected captions to their parent figures/tables and merges their bounding boxes into a single region. In the output JSON, paratext elements are assigned type `"paratext"` for downstream use with GROBID's `typedAreas` API.

It provides a `CaptionMerger` base class (ABC) with two strategies.

### Strategies

| Strategy | Class | Description |
|----------|-------|-------------|
| `threshold` | `ThresholdMerger` | Requires horizontal overlap (same column) and a max distance threshold (default: 50). Only links above/below. |
| `distance` | `DistanceMerger` | Considers all 4 directions (above, below, left, right). Ranks all valid candidates by distance and picks the closest. No hard threshold. |

```shell
# Process JSON files using the distance strategy (default)
python -m local.caption_merging input_dir/ -o output_dir/

# Use the threshold strategy with custom max distance
python -m local.caption_merging input_dir/ -o output_dir/ --strategy threshold --max-distance 80

# Output only figures and tables (no paratext)
python -m local.caption_merging input_dir/ -o output_dir/ --only figure,table

# Output only paratext
python -m local.caption_merging input_dir/ -o output_dir/ --only paratext
```

### Evaluation

`caption_merging_eval` benchmarks merging strategies against 150 annotated ground truth files in `caption_merging_evaluation/`. Each ground truth file contains figures, captions, and pre-computed merged boxes (`figure_all`/`table_all`) linked by a `group` field.

The evaluation randomly shuffles input order across multiple trials to test robustness against the greedy processing order, then compares merged boxes to ground truth using IoU.

```shell
# Evaluate all strategies (default)
python -m local.caption_merging_eval caption_merging_evaluation/ -o results.json

# Evaluate a single strategy
python -m local.caption_merging_eval caption_merging_evaluation/ -o results.json --strategy distance

# Custom parameters
python -m local.caption_merging_eval caption_merging_evaluation/ -o results.json --strategy all --trials 10 --iou-threshold 0.9 --seed 42
```

#### Arguments

| Argument | Description | Default |
|----------|-------------|---------|
| `eval_dir` | Directory with ground truth JSON files | `./caption_merging_evaluation` |
| `--output`, `-o` | Output JSON file for results (required) | - |
| `--strategy`, `-s` | Strategy to evaluate: `threshold`, `distance`, or `all` | `all` |
| `--max-distance`, `-d` | Max distance for ThresholdMerger | `50` |
| `--trials`, `-t` | Number of random shuffle trials per file | `10` |
| `--iou-threshold` | IoU threshold for counting a match as correct | `0.5` |
| `--seed` | Random seed for reproducibility | `42` |

#### Ground Truth Format

Each JSON file in `caption_merging_evaluation/` is an array of elements with `page`, `x`, `y`, `width`, `height`, `type`, and `group` fields. Types include `figure`, `figure_caption`, `figure_all`, `table`, `table_caption`, `table_all`. The `group` field links a figure/table to its caption and the expected merged box.

---

## Output Format

Both Paddle and LADaS inference scripts produce a standardized JSON output format with detected layout elements:

```json
[
  {
    "page": 1,
    "x": 100,
    "y": 200,
    "width": 400,
    "height": 300,
    "type": "figure",
    "confidence": 0.95
  }
]
```

| Field | Description |
|-------|-------------|
| `page` | Page number (1-indexed) |
| `x` | X coordinate of top-left corner |
| `y` | Y coordinate of top-left corner |
| `width` | Width of bounding box |
| `height` | Height of bounding box |
| `type` | Element type (figure, table, equation, header, footer, etc.) |
| `confidence` | Detection confidence score (LADaS only) |

---

## Filtering Options

Use the `--only` flag to filter and aggregate detected elements into specific categories:

| Option | Description | Included Types |
|--------|-------------|----------------|
| `display` | Display elements only | figure, image, chart, table, equation, formula |
| `paratext` | Paratext elements only | header, footer |
| `grobid` | Both display and paratext | All of the above |

### Type Aggregation

When using filters, similar element types are aggregated:

**Display elements:**

- `figure` <- figure, image, chart, figure_text, chart_text
- `table` <- table, table_text
- `equation` <- equation, formula, equation_text

**Paratext elements:**

- `headnote` <- header
- `footer` <- footer

---

## Programmatic Usage

```python
from local.paddle_inference import PaddleDocumentProcessor, load_transform_elements, filter_and_aggregate

processor = PaddleDocumentProcessor(
    model_name="PP-DocLayout-S",
    dpi=72,
    preload_model=True
)

result = processor.process_document("document.pdf", "output")

if result["success"]:
    elements = load_transform_elements(result["output_dir"])
    figure_types = {
        "figure": ["figure", "image", "chart"],
        "table": ["table"],
        "equation": ["equation", "formula"]
    }
    filtered = filter_and_aggregate(elements, figure_types)
```

```python
from local.ladas_inference import LADaSDocumentProcessor, load_transform_elements

processor = LADaSDocumentProcessor(
    model_name="path/to/model.pt",
    dpi=72,
    preload_model=True
)

result = processor.process_document("document.pdf", "output")

if result["success"]:
    elements = load_transform_elements(result["output_dir"])
    print(f"Detected {len(elements)} elements")
```

---

## Output Directory Structure

After processing, the output directory will contain:

```
output/
├── document/           # Per-document subdirectory
│   ├── page_0001.jpg   # Converted page images
│   ├── page_0002.jpg
│   ├── res_0.jpg       # Annotated result images
│   ├── res_0.json      # Per-page detection results
│   ├── res_1.jpg
│   └── res_1.json
└── document.json       # Aggregated results (with --cleanup-images)
```

When using `--cleanup-images`, only the aggregated JSON file remains:

```
output/
└── document.json       # Aggregated results only
```
