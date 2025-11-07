# Sample code for running docling-serve on modal

## Setup steps

1. Run `uv sync`
2. Setup a secret called `DOCLING_SERVE_API_KEY` with an environment variable called `DOCLING_SERVE_API_KEY` using a randomly generated key. Docs: https://modal.com/docs/guide/secrets
3. Deploy it run `uv run modal deploy docling_server_inference.py`

## Sample curl
First Step 1 : set the DOCLING_SERVE_API_KEY in the terminal with : export DOCLING_SERVE_API_KEY="sedrfctvgjybhkjgvcfdxswqzwefrdgtfyguhjnbhgvcfdxswedrfuj"

```bash
curl -X "POST" \
  "https://sana-khamaassi--docling-serve-modal-docling-serve-fastap-f4af93.modal.run/v1/convert/file" \
  -H "accept: application/json" \
  -H "Content-Type: multipart/form-data" \
  -H "X-Api-Key: $DOCLING_SERVE_API_KEY" \
  -F "image_export_mode=placeholder" \
  -F "files=@/Users/mandamac1/Downloads/scie1.pdf;type=application/pdf" \
  -F "to_formats=json" \
  -F "do_ocr=true" \
  -F "vlm_pipeline_model=granite_docling" \
  -vv \
  -L \
  --http1.1 \
  -o output.json
```



## Notes

- L40S are selected for speed but it seems that even T4 would have worked.
- Async api endpoints from docling-serve are not used since modal has builtin handling for request timeout, and serverless container may terminate before the job is finished, see https://modal.com/docs/guide/webhook-timeouts
- Using Modal memory snapshots actually seems to make it slower.


