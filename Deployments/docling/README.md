# Docline deployment

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

---

## 3. Convert a Local File – JSON Output

curl -X POST "https://sana-khamaassi--docling-gpu.modal.run/v1/convert/file" \
  -H "accept: application/json" \
  -F "files=@/file_path" \
  -F "to_formats=json"
  -o "output_path/output.json"



---

## 4. Convert a Local File – Markdown Output

curl -X POST "https://sana-khamaassi--docling-gpu.modal.run/v1/convert/file" \
  -H "accept: application/json" \
  -F "files=@file_path" \
  -F "to_formats=md" \
  -o "output_path/output.md"


---

***********Notes*****

We can run docling locally through docling cli or as a python package  https://github.com/docling-project/docling 
