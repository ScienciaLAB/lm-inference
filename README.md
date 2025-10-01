# modal-inference

# Collaborative Modal Deployments

This repository manages deployments of multiple services (e.g., Docling, vLLM models) on Modal.com.

## Repository Structure

- **deployments/**: Contains subfolders for each deployment (e.g., `docling/` with scripts like `docling_serve.py`).

## Deployment

The Docling server is deployed on [Modal](https://modal.com) and accessible here: [Docling on Modal](https://sana-khamaassi--docling-gpu.modal.run/ui/)

# Docling Serve – Testing with `curl`

This section shows how to test the deployed Docling Serve instance using `curl`.  
We focus on converting **documents from a URL** or a **local file**, and requesting results in **JSON** or **Markdown** format.
The deployment URL on Modal :

```
https://sana-khamaassi--docling-gpu.modal.run
```

---

The usage of the API is avalaible here
(Source: [Docling Serve GitHub usage.md](https://github.com/docling-project/docling-serve/blob/main/docs/usage.md))

## Full CURL request

curl -X 'POST' \
 'https://sana-khamaassi--docling-gpu.modal.run/v1/convert/source' \
 -H 'accept: application/json' \
 -H 'Content-Type: application/json' \
 -d '{
"options": {
"from_formats": [
"docx",
"pptx",
"html",
"image",
"pdf",
"asciidoc",
"md",
"xlsx"
],
"to_formats": ["md", "json", "html", "text", "doctags"],
"image_export_mode": "placeholder",
"do_ocr": true,
"force_ocr": false,
"ocr_engine": "easyocr",
"ocr_lang": [
"fr",
"de",
"es",
"en"
],
"pdf_backend": "dlparse_v2",
"table_mode": "fast",
"abort_on_error": false,
"do_table_structure": true,
"include_images": true,
"images_scale": 2
},
"http_sources": [{"url": "https://arxiv.org/pdf/2206.01062"}]
}'

## 1. Convert a Document from a URL – JSON Output

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

**Explanation:**

- `Accept: application/json` ensures the response is wrapped in JSON, with Markdown content in the `"md_content"` field

**Expected Output (shape):**

```json
{
  "document": {
    "filename": "2501.17887v1.pdf",
    "md_content": "## Docling: An Efficient Open-Source Toolkit..."
  },
  "status": "success"
}
```

**Documentation Reference:**  
From Docling Serve README:  
(Source: [Docling Serve GitHub](https://github.com/docling-project/docling-serve))

---

## 2. Convert a Document from a URL – Markdown Output

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

**Explanation:**

- Same request as above, but `Accept: text/markdown` tells the server to return only the Markdown text instead of JSON.

**Expected Output (shape):**

````markdown
## Docling: An Efficient Open-Source Toolkit for AI-driven Document Conversion

\n\nNikolaos Livathinos _ , Christoph Auer _ , Maksym Lysak, Ahmed Nassar, Michele Dolfi, Panagiotis Vagenas, Cesar Berrospi, Matteo Omenetti, Kasper Dinkla, Yusik Kim, Shubham Gupta, Rafael Teixeira de Lima, Valery Weber, Lucas Morin, Ingmar Meijer,```
````

📖 **Documentation Reference:**  
Docling Serve supports format negotiation through the `Accept` header. Setting `text/markdown` instructs the API to return Markdown instead of JSON.  
(Source: [Docling Serve GitHub usage.md](https://github.com/docling-project/docling-serve/blob/main/docs/usage.md))

---

## 3. Convert a Local File – JSON Output

```bash
curl -X POST "https://sana-khamaassi--docling-gpu.modal.run/v1/convert/file_name"      -H "Accept: application/json"      -F "file=@./mydocument.pdf"
```

**Explanation:**

- Uploads a local file (`mydocument.pdf`)
- The API extracts its content and responds in JSON format
- The Markdown content will be available in the `"md_content"` field of the JSON

**Expected Output (shape):**

```json
{
  "document": {
    "filename": "mydocument.pdf",
    "md_content": "## Extracted Title\n\nThis is the document body..."
  },
  "status": "success"
}
```

📖 **Documentation Reference:**  
Docling Serve file upload endpoint `/v1/convert/file` accepts `multipart/form-data` with a `file` field.  
(Source: [Docling Serve GitHub](https://github.com/docling-project/docling-serve))

---

## 4. Convert a Local File – Markdown Output

```bash
curl -X POST "https://sana-khamaassi--docling-gpu.modal.run/v1/convert/file"      -H "Accept: text/markdown"      -F "file=@./mydocument.pdf"
```

**Explanation:**

- Same as above, but response is returned directly in Markdown.

**Expected Output (shape):**

```markdown
## Extracted Title

This is the document body in Markdown format...
```

**Documentation Reference:**  
Docling Serve supports Markdown output via the `Accept: text/markdown` header, even when uploading local files.  
(Source: [Docling Serve GitHub](https://github.com/docling-project/docling-serve))

---
