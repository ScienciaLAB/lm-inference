# MinerU 2.5 (vLLM) - Inference API

Deploys [MinerU 2.5](https://github.com/opendatalab/MinerU)
(`opendatalab/MinerU2.5-2509-1.2B`) on Modal. A vLLM server is warmed once per
container (`mineru-vllm-server`) and the `mineru` client talks to it over HTTP
(`vlm-http-client` backend) for each request.

## Deploy

Run from the repository root so `lm_inference_utils.py` is importable:

```shell
modal deploy deployments/minerU/inference_minerU.py
```

## CURL command

```shell
curl -X POST https://modal_generated_url \
     -F 'file=@/path/to/document.pdf' \
     -F 'output_format=markdown_content' \
     -F 'lang=ch'
```

Output formats: `['markdown_content', 'json_content', 'middle_json']` (default
`markdown_content`). `lang` is passed to MinerU for OCR hinting (default `ch`,
which also handles English).

**example**

```shell
curl -X POST https://your-app-name--mineru-vllm-official-app-parse-document-endpoint.modal.run \
  -F 'file=@../../scie1.pdf' \
  -F 'output_format=markdown_content' \
  -o output.md
```

The response is `{"result": ..., "cost_info": {"duration_seconds": ..., "cost_usd": ...}}`.
