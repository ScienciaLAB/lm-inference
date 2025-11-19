# cURL commands to query the deployed Modal service

The script allow to deploy: 
- Qwen-VLMM-3B: `qwen-3` deployment
- Qwen-VLMM-25B: `qwen-25` deployment

## Single image analysis endpoint (VLMMModel.generate)

```shell
curl -X POST "https://{your-app-name}--{qwen-3|qwen-25}-generate.modal.run" \
  -H "Content-Type: multipart/form-data" \
  -F "question=Extract the text of this formula in Latex format" \
  -F "image=@/path/to/your/image.png"
```

## Multi-page document structure extraction endpoint
```shell 
curl -X POST "https://{your-app-name}--{qwen-3|qwen-25}-extract-pdf.modal.run" \
  -H "Content-Type: application/json" \
  -d '{
    "pages": [
      {
        "page_number": 1,
        "image_base64": "iVBORw0KGgoAAAANSUhEUgAA...",
        "width": 1200,
        "height": 1600,
        "dpi": 150
      },
      {
        "page_number": 2,
        "image_base64": "iVBORw0KGgoAAAANSUhEUgAA...",
        "width": 1200,
        "height": 1600,
        "dpi": 150
      }
    ]
  }'
```

## Main PDF document processing endpoint (process_pdf_document_structure)

```shell
curl -X POST "https://{your-app-name}--{qwen-25|qwen-3}-extract-pdf.modal.run" \
  -H "Content-Type: multipart/form-data" \
  -F "pdf=@/path/to/your/document.pdf" \
  -F "dpi=150"
```