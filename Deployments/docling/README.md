# Docling Deployment

This directory contains the Modal deployment configuration for the Docling document processing service.

## API Usage

The deployment provides REST endpoints for document conversion. Below are common usage examples.

### Base URL
```
https://sana-khamaassi--docling-gpu.modal.run
```

### 1. Convert Document from URL (JSON Output)

```bash
curl -X POST "https://sana-khamaassi--docling-gpu.modal.run/v1/convert/source" \
  -H "Accept: application/json" \
  -H "Content-Type: application/json" \
  -d '{
    "options": {
      "to_formats": ["json"]
    },
    "sources": [
      {
        "kind": "http",
        "url": "https://arxiv.org/pdf/2309.10923"
      }
    ]
  }'
```

### 2. Convert Document from URL (Markdown Output)

```bash
curl -X POST "https://sana-khamaassi--docling-gpu.modal.run/v1/convert/source" \
  -H "Accept: application/json" \
  -H "Content-Type: application/json" \
  -d '{
    "options": {
      "to_formats": ["md"]
    },
    "sources": [
      {
        "kind": "http",
        "url": "https://arxiv.org/pdf/2309.10923"
      }
    ]
  }'
```

### 3. Convert Local File (JSON Output)

```bash
curl -X POST "https://sana-khamaassi--docling-gpu.modal.run/v1/convert/file" \
  -H "accept: application/json" \
  -F "files=@/path/to/your/document.pdf" \
  -F "to_formats=json" \
  -o "output.json"
```

### 4. Convert Local File (Markdown Output)

```bash
curl -X POST "https://sana-khamaassi--docling-gpu.modal.run/v1/convert/file" \
  -H "accept: application/json" \
  -F "files=@/path/to/your/document.pdf" \
  -F "to_formats=md" \
  -o "output.md"
```

## Advanced Configuration

### Full Options Example

```bash
curl -X 'POST' \
  'https://sana-khamaassi--docling-gpu.modal.run/v1/convert/source' \
  -H 'accept: application/json' \
  -H 'Content-Type: application/json' \
  -d '{
    "options": {
      "from_formats": [
        "docx", "pptx", "html", "image", "pdf", "asciidoc", "md", "xlsx"
      ],
      "to_formats": ["md", "json", "html", "text", "doctags"],
      "image_export_mode": "placeholder",
      "do_ocr": true,
      "force_ocr": false,
      "ocr_engine": "easyocr",
      "ocr_lang": ["fr", "de", "es", "en"],
      "pdf_backend": "dlparse_v2",
      "table_mode": "fast",
      "abort_on_error": false,
      "do_table_structure": true,
      "include_images": true,
      "images_scale": 2
    },
    "http_sources": [{"url": "https://arxiv.org/pdf/2206.01062"}]
  }'
```

### Granite Docling Pipeline

The deployment is configured to use the Granite Docling model for enhanced document understanding:

```bash
curl -X 'POST' \
  'https://sana-khamaassi--docling-gpu.modal.run/v1/convert/source' \
  -H 'accept: application/json' \
  -H 'Content-Type: application/json' \
  -d '{
    "sources": [{"kind": "http", "url": "https://arxiv.org/pdf/2501.17887"}],
    "options": {
      "do_picture_description": true,
      "do_table_structure": true,
      "pipeline": "granitedocling"
    }
  }'
```

## Deployment Details

### Configuration
- **GPU**: A10 with 16GB memory
- **Timeout**: 10 minutes startup, 5 minutes scale-down
- **Max containers**: 2
- **Concurrency**: 4 max inputs
- **Models**: Pre-downloaded for fast startup

### Supported Formats
- **Input**: PDF, DOCX, PPTX, HTML, Images, AsciiDoc, Markdown, XLSX
- **Output**: JSON, Markdown, HTML, Text, DocTags
- **OCR**: EasyOCR with multi-language support (EN, FR, DE, ES)

### Features
- **OCR**: Text extraction from images and scanned PDFs
- **Table Extraction**: Automatic table detection and structure analysis
- **Image Analysis**: Picture description and classification
- **Layout Analysis**: Document structure understanding
- **Granite Model**: Enhanced AI-powered document processing

## Development

### Local Development

For local development, you can run Docling directly:

```bash
# Using CLI
pip install docling
docling convert document.pdf --output result.md

# Using Python
from docling.document_converter import DocumentConverter
converter = DocumentConverter()
result = converter.convert("document.pdf")
```

**Source:** [Docling Project GitHub](https://github.com/docling-project/docling)

### Deployment

To deploy or update the service:

```bash
modal deploy docling_server_inference.py
```
