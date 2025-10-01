# modal-inference

# Collaborative Modal Deployments

This repository manages deployments of multiple services (e.g., Docling, vLLM models) on Modal.com.

## Repository Structure

- **deployments/**: Contains subfolders for each deployment (e.g., `docling/` with scripts like `docling_serve.py`).

## Deployment

The Docling server is deployed on [Modal](https://modal.com) and accessible here: [Docling on Modal](https://sana-khamaassi--docling.modal.run/ui/)
**The CURL cmd for processing document through url :**

```bash
curl -X POST https://sana-khamaassi--docling.modal.run/v1/convert/source \
  -H 'accept: application/json' \
  -H 'Content-Type: application/json' \
  -d '{
    "sources": [{"kind": "http", "url": "https://arxiv.org/pdf/2501.17887"}]
  }'
```
