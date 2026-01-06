# Local Inference Scripts

This directory contains scripts for running layout detection inference locally using PaddleOCR and LADaS (YOLO-based) models.

## Table of Contents

- [Installation](#installation)
  - [PaddleOCR (CPU)](#paddleocr-cpu)
  - [PaddleOCR (GPU)](#paddleocr-gpu)
  - [LADaS (YOLO)](#ladas-yolo)
- [Paddle Inference](#paddle-inference)
  - [Single Document Processing](#single-document-processing)
  - [Batch Processing](#batch-processing)
- [LADaS Inference](#ladas-inference)
- [Output Format](#output-format)
- [Filtering Options](#filtering-options)

---

## Installation

### PaddleOCR (CPU)

Reference: https://www.paddlepaddle.org.cn/documentation/docs/en/install/pip/macos-pip_en.html 

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

Use `paddle_inference.py` to process a single PDF document with PaddleOCR layout detection.

#### Usage

```shell
python paddle_inference.py <input_pdf> [options]
```

#### Arguments

| Argument | Description | Default |
|----------|-------------|---------|
| `input` | Input PDF document path (required) | - |
| `--output`, `-o` | Output directory for processed images and results | `output` |
| `--model-name` | Model name for layout detection | `PP-DocLayout-S` |
| `--dpi` | DPI for PDF to image conversion | `70` |
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

Basic usage:
```shell
python paddle_inference.py document.pdf
```

Specify output directory and model:
```shell
python paddle_inference.py document.pdf --output ./results --model-name PP-DocLayout-L
```

Extract only display elements (figures, tables, equations) with cleanup:
```shell
python paddle_inference.py document.pdf -o ./results --only display --cleanup-images
```

High-resolution processing:
```shell
python paddle_inference.py document.pdf --dpi 150 --model-name PP-DocLayout-M
```

---

### Batch Processing

Use `paddle_inference_batch.py` to process multiple PDF documents in a directory with parallel workers.

#### Usage

```shell
python paddle_inference_batch.py <input_dir> [options]
```

#### Arguments

| Argument | Description | Default |
|----------|-------------|---------|
| `input_dir` | Input directory containing PDF files (required) | - |
| `--output`, `-o` | Output directory for results | `output` |
| `--model-name` | Model name for layout detection | `PP-DocLayout-S` |
| `--dpi` | DPI for PDF to image conversion | `70` |
| `--workers`, `-w` | Number of parallel workers | CPU count |
| `--temp-dir` | Temporary directory for processing | auto-generated |
| `--only` | Filter output to specific element types | - |
| `--cleanup-images` | Clean up intermediate files | `False` |

#### Examples

Process all PDFs in a directory:
```shell
python paddle_inference_batch.py ./input_pdfs --output ./results
```
Process all images in a directory:

```shell
python paddle_img_inference_batch.py /Users/mandamac1/Downloads/grobid-alignment/data/paddle/Doclaynet/PNG_val -o ./results/PP-DocLayout-M --model-name PP-DocLayout-M --workers 4
```

Parallel processing with 4 workers:
```shell
python paddle_inference_batch.py ./input_pdfs -o ./results --workers 4
```
High-quality batch processing with filtering:
```shell
python paddle_inference_batch.py ./input_pdfs -o ./results --model-name PP-DocLayout-L --dpi 150 --only grobid --cleanup-images
```

---

## LADaS Inference

Use `ladas_inference.py` to process PDF documents using the LADaS YOLO-based layout detection model.

## LADaS image Inference
```shell
python ladas_img_inference_batch.py /Users/mandamac1/Downloads/grobid-alignment/data/paddle/Doclaynet/PNG_val --output ./results/LADaS --model ./LADaS/model-train-test.pt
```

#### Usage
```shell
python ladas_inference.py <input_pdf> --model-file <model_path> [options]
```



#### Arguments

| Argument | Description | Default |
|----------|-------------|---------|
| `input` | Input PDF document path (required) | - |
| `--model-file` | Path to YOLO model file (required) | - |
| `--output`, `-o` | Output directory for results | `output` |
| `--dpi` | DPI for PDF to image conversion | `70` |
| `--temp-dir` | Temporary directory for processing | auto-generated |
| `--only` | Filter output to specific element types | - |
| `--cleanup-images` | Clean up intermediate files | `False` |

#### Examples

Basic usage with model file:
```shell
python ladas_inference.py document.pdf --model-file ./LADaS/model-train-test.pt
```

With output directory and filtering:
```shell
python ladas_inference.py document.pdf --model-file ./model.pt -o ./results --only display
```

Full processing with cleanup:
```shell
python ladas_inference.py document.pdf --model-file ./model.pt -o ./results --dpi 100 --only grobid --cleanup-images
```

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
  },
  {
    "page": 1,
    "x": 50,
    "y": 600,
    "width": 500,
    "height": 150,
    "type": "table"
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
| `paratext` | Paratext elements only | header → headnote, footer |
| `grobid` | Both display and paratext | All of the above |

### Type Aggregation

When using filters, similar element types are aggregated:

**Display elements:**
- `figure` ← figure, image, chart, figure_text, chart_text
- `table` ← table, table_text
- `equation` ← equation, formula, equation_text

**Paratext elements:**
- `headnote` ← header
- `footer` ← footer

---

## Programmatic Usage

You can also use the processors programmatically in your Python code:

### PaddleDocumentProcessor

```python
from paddle_inference import PaddleDocumentProcessor, load_transform_elements, filter_and_aggregate

# Initialize processor
processor = PaddleDocumentProcessor(
    model_name="PP-DocLayout-S",
    dpi=70,
    preload_model=True
)

# Process a document
result = processor.process_document("document.pdf", "output")

if result["success"]:
    # Load and transform elements to standard format
    elements = load_transform_elements(result["output_dir"])
    
    # Optionally filter elements
    figure_types = {
        "figure": ["figure", "image", "chart"],
        "table": ["table"],
        "equation": ["equation", "formula"]
    }
    filtered = filter_and_aggregate(elements, figure_types)
```

### LADaSDocumentProcessor

```python
from ladas_inference import LADaSDocumentProcessor, load_transform_elements

# Initialize processor with model path
processor = LADaSDocumentProcessor(
    model_name="path/to/model.pt",
    dpi=70,
    preload_model=True
)

# Process a document
result = processor.process_document("document.pdf", "output")

if result["success"]:
    elements = load_transform_elements(result["output_dir"])
    print(f"Detected {len(elements)} elements")
```

---

## Directory Structure

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