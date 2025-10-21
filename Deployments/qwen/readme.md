# curl commands to query the deployed Modal service

# 1. Single image analysis endpoint (VLMMModel.generate)
curl -X POST "https://your-app-name--vlmmmodel-generate.modal.run" \
  -H "Content-Type: multipart/form-data" \
  -F "question=Extract the text of this formula in Latex format" \
  -F "image=@/path/to/your/image.png"

# 2. Multi-page document structure extraction endpoint (VLMMModel.extract_document_structure)
curl -X POST "https://your-app-name--vlmmmodel-extract-document-structure.modal.run" \
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

# 3. Main PDF document processing endpoint (process_pdf_document_structure)
curl -X POST "https://your-app-name--process-pdf-document-structure.modal.run" \
  -H "Content-Type: multipart/form-data" \
  -F "pdf=@/path/to/your/document.pdf" \
  -F "dpi=150"

# Example usage notes:
# - Replace "your-app-name" with the actual Modal app name from the app definition
# - Replace "/path/to/your/image.png" with actual image file path
# - Replace "/path/to/your/document.pdf" with actual PDF file path
# - For the JSON endpoint, replace the base64 strings with actual base64-encoded images
# - You can find the exact URL by running: modal app list