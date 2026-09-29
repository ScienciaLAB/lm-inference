# Baguette-Software-Dataset - Inference API

Extraction of dataset and software mentions from a paper, with the
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

## Request

One request is one paper. The endpoint is
`https://<workspace>--baguette-software-dataset-app-extract-endpoint.modal.run/`, `$URL` below.

| Input | Request |
|---|---|
| A text | JSON `{"text": "..."}`, or a plain text body, or the form field `text` |
| A file | form field `file` |
| Paragraphs | JSON `{"paragraphs": ["...", "..."]}` |

```shell
# a text
curl -X POST $URL -H 'Content-Type: application/json' \
  -d '{"text": "The data was acquired using the ResearchIR MAX 4.0 software."}'

# a text file, as a file or as the body
curl -X POST $URL -F 'file=@paper.txt'
curl -X POST $URL -H 'Content-Type: text/plain' --data-binary @paper.txt

# a TEI file from GROBID
curl -X POST $URL -F 'file=@paper.tei.xml'
```

Step 2 is skipped with `"analyze": false` in the JSON, `-F analyze=false` in a form, or `?analyze=false` with a
plain text body.

The server does all the splitting:

| Input | Paragraphs |
|---|---|
| text, `.txt`, `.md` | the blocks separated by blank lines |
| `.xml` | TEI from GROBID: the `<p>` elements of the abstract, body and back |
| `.json` | a list of strings, or an object with `paragraphs` or `text` |

A paragraph longer than the input budget (7040 tokens) is split again at sentence boundaries, and the mentions of its
parts are merged.

## Response

| Field | Description |
|---|---|
| `paragraphs` | One item per paragraph: `index`, `text`, `is_boilerplate`, `datasets`, `software`. An item of a split paragraph has `chunks`, the number of parts. An item has an `error` when the paragraph, or one of its parts, failed. |
| `mentions` | The mentions of all paragraphs, deduplicated by name. |
| `record` | The article-level record of step 2. Absent when `analyze` is false or when there is no mention. |
| `duration_seconds`, `cost_usd` | Duration of the request and its GPU cost. |

An input that cannot be read, or that has no paragraph, gives HTTP 400.

## Client

`baguette_client.py` sends the input as it is.

```shell
# a text; the result is printed
python deployments/baguette/baguette_client.py --endpoint $URL \
  --text "The data was acquired using the ResearchIR MAX 4.0 software."

# one or more files
python deployments/baguette/baguette_client.py --endpoint $URL --input_file paper.txt

# a folder, with one JSON file per paper and a CSV summary
python deployments/baguette/baguette_client.py --endpoint $URL \
  --input_folder /path/to/papers \
  --output_dir ./baguette_out \
  --csv_output ./baguette_results.csv \
  --threads 4
```

| Option | Description |
|---|---|
| `--output_dir` | Write one JSON file per paper. Without it the result is printed. Required with `--input_folder`. |
| `--csv_output` | CSV summary. A second run skips the papers that already succeeded. |
| `--threads` | In-flight requests. Keep it at or below `max_containers` (4). |
| `--no_analyze` | Skip step 2. |

The exit code is 1 when a paper failed.
