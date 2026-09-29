# Baguette-Software-Dataset - Inference API

Extraction of dataset and software mentions from the paragraphs of a paper, with the
[Baguette-Software-Dataset](https://huggingface.co/buckets/dataesr/Baguette-Software-Dataset) model (608M parameters)
served by vLLM on one L4 GPU.

The model works in two steps:

1. **Extraction** (validated by the authors): each paragraph gives its dataset and software mentions.
2. **Analysis** (indicative only): the mentions of all paragraphs, deduplicated, give one article-level record with a
   PLOS-OSI summary.

The weights are downloaded from the public bucket into the Modal volume `baguette-model` when the first container
starts.

## Deploy

From the repository root (`lm-inference/`):

```shell
modal deploy deployments/baguette/inference_baguette.py
```

Smoke test on the paragraphs of the authors' example, without deploying:

```shell
modal run deployments/baguette/inference_baguette.py
```

## CURL command

```shell
curl -X POST https://<workspace>--baguette-software-dataset-app-extract-endpoint.modal.run/ \
  -H 'Content-Type: application/json' \
  -d '{"paragraphs": ["The data was acquired using the ResearchIR MAX 4.0 software."], "analyze": true}'
```

## Batch client

`baguette_client.py` sends a folder of papers, one request per paper, and writes one JSON file per paper plus a
CSV summary. A second run skips the papers that already succeeded.

```shell
python deployments/baguette/baguette_client.py \
  --input_folder /path/to/tei \
  --output_dir ./baguette_out \
  --csv_output ./baguette_results.csv \
  --endpoint https://<workspace>--baguette-software-dataset-app-extract-endpoint.modal.run \
  --threads 4
```

| Input | Paragraphs |
|---|---|
| `.xml` | TEI from GROBID: the `<p>` elements of the abstract, body and back |
| `.json` | a list of strings, or an object with a `paragraphs` list |
| `.txt`, `.md` | blocks separated by blank lines |

Keep `--threads` at or below `max_containers` (4). `--no_analyze` skips step 2. In the output files, each
paragraph item also carries its `text`.

## Request

| Field | Type | Description |
|---|---|---|
| `paragraphs` | list of strings | The paragraphs of one paper, in order. Required. |
| `analyze` | boolean | Run step 2. Default `true`. |

## Response

| Field | Description |
|---|---|
| `paragraphs` | One item per paragraph: `index`, `is_boilerplate`, `datasets`, `software`. An item has an `error` when the paragraph failed, for example when it is longer than the 8192-token context. |
| `mentions` | The mentions of all paragraphs, deduplicated by name. |
| `record` | The article-level record of step 2. Absent when `analyze` is `false` or when there is no mention. |
| `duration_seconds`, `cost_usd` | Duration of the request and its GPU cost. |
