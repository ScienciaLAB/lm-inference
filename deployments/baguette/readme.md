# Baguette-Software-Dataset - Inference API

Extraction of dataset and software mentions from a paper, with the
[Baguette-Software-Dataset](https://huggingface.co/buckets/dataesr/Baguette-Software-Dataset) model, served by vLLM
on Modal.

| File | Content |
|---|---|
| `inference_baguette.py` | The Modal app: the GPU service and the HTTP endpoint |
| `baguette_client.py` | The client: a text, files, or a folder of papers |

Contents: [Model](#model) · [Deploy](#deploy) · [Request](#request) · [Splitting](#splitting) ·
[Response](#response) · [Client](#client) · [Cost and speed](#cost-and-speed) · [Troubleshooting](#troubleshooting) ·
[Test without a GPU](#test-without-a-gpu) · [Limits](#limits)

## Model

| | |
|---|---|
| Architecture | `LlamaForCausalLM`, 608M parameters, bfloat16, 1.35 GB |
| Context | 8192 tokens |
| Languages | English and French scientific text |
| Source | public Hugging Face bucket `dataesr/Baguette-Software-Dataset` |

The model works in two steps:

1. **Extraction** (validated by the authors): each paragraph gives its dataset and software mentions.
2. **Analysis** (indicative only): the mentions of all paragraphs, deduplicated by name, give one article-level
   record with a PLOS-OSI summary.

The prompts are the two training templates of the model card. The tokenizer has no chat template, so the service
calls the completions API of vLLM with the raw prompt, with greedy decoding and a stop at `<|im_end|>`.

```
<|im_start|>user
<text>{paragraph}</text><|im_end|>
<|im_start|>assistant
```

```
<|im_start|>user
<mentions>{mentions as JSON}</mentions><|im_end|>
<|im_start|>assistant
```

The authors' example loads the model repository `PleIAs/Baguette-Software-Dataset`, which is not public. The
service reads the same files from the bucket, without a token, and keeps them in the Modal volume `baguette-model`.
The download happens when the first container starts.

## Deploy

Install the Modal client and log in once:

```shell
pip install -r deployments/requirements.txt
modal setup
```

Then, from the repository root (`lm-inference/`):

```shell
modal deploy deployments/baguette/inference_baguette.py
```

Modal prints the URL of the endpoint,
`https://<workspace>--baguette-software-dataset-app-extract-endpoint.modal.run/`. It is `$URL` below.

Smoke test on the paragraphs of the authors' example, without deploying:

```shell
modal run deployments/baguette/inference_baguette.py
```

### Deploy parameters

They are environment variables of the deploy command. They are copied into the image, so the containers use the
same values.

| Variable | Default | Description |
|---|---|---|
| `BAGUETTE_GPU` | `L4` | GPU type, a key of `GPU_COST_PER_SECOND` in `lm_inference_utils.py`. `A100-40GB` and `A100_40GB` are the same. An unknown type stops the deploy. |
| `BAGUETTE_MAX_CONTAINERS` | `4` | GPU containers at most |
| `BAGUETTE_MAX_INPUTS` | `1` | Papers served at once by one container |

```shell
BAGUETTE_GPU=A10G BAGUETTE_MAX_INPUTS=8 modal deploy deployments/baguette/inference_baguette.py
```

### Other settings

They are constants of `inference_baguette.py`.

| Constant | Value | Description |
|---|---|---|
| `INPUT_TOKEN_BUDGET` | 200 | Tokens of text in one prompt. See [Splitting](#splitting). |
| `MAX_TOKENS` | 2048 | Tokens of one answer at most |
| `MAX_RETRY_DEPTH` | 3 | Times a failed part is split in two |
| `MIN_RETRY_TOKENS` | 40 | A part with fewer tokens is not split again |
| `MAX_PARALLEL_REQUESTS` | 32 | Prompts of one paper sent to vLLM at the same time |
| `scaledown_window` | 300 s | Idle time before a container stops |

### Redeploy

Deploy when no client is running. During a redeploy, containers of the previous version keep taking requests for
a while: 5 minutes in one case. When the new version changes the arguments of the service, these requests fail
with HTTP 500. The client sends them again. To avoid the problem, stop the app first:

```shell
modal app stop baguette-software-dataset-app
modal deploy deployments/baguette/inference_baguette.py
```

## Request

One request is one paper, sent with `POST`.

| Input | Request |
|---|---|
| A text | JSON `{"text": "..."}`, or a plain text body, or the form field `text` |
| A file | form field `file` |
| Paragraphs | JSON `{"paragraphs": ["...", "..."]}` |

| Parameter | Type | Default | Description |
|---|---|---|---|
| `text` | string | | The text of the paper |
| `file` | file | | A `.txt`, `.md`, `.xml` or `.json` file. Another extension is read as text. |
| `paragraphs` | list of strings | | The paragraphs of the paper, in order |
| `analyze` | boolean | `true` | Run step 2 |

`analyze` is a field of the JSON, a field of the form, or a query parameter (`?analyze=false`) with a plain text
body.

```shell
# a text
curl -X POST $URL -H 'Content-Type: application/json' \
  -d '{"text": "The data was acquired using the ResearchIR MAX 4.0 software."}'

# a text file, as a file or as the body
curl -X POST $URL -F 'file=@paper.txt'
curl -X POST $URL -H 'Content-Type: text/plain' --data-binary @paper.txt

# a TEI file from GROBID, without step 2
curl -X POST $URL -F 'file=@paper.tei.xml' -F 'analyze=false'

# paragraphs
curl -X POST $URL -H 'Content-Type: application/json' \
  -d '{"paragraphs": ["First paragraph.", "Second paragraph."], "analyze": false}'
```

| File | Paragraphs |
|---|---|
| `.txt`, `.md` | see [Splitting](#splitting) |
| `.xml` | TEI from GROBID: the `<p>` elements of the abstract, body and back, in this order. Figure and table captions are not read: GROBID writes them in `<figDesc>`. |
| `.json` | a list of strings, or an object with `paragraphs` or `text` |

Text files are read as UTF-8.

## Splitting

The server does all the splitting. The client sends the input as it is.

The model only works on inputs of the size of a paragraph, whatever its context. With one known mention at each
end of the input, on CPU:

| Input (tokens) | Result |
|---|---|
| up to about 500 | both mentions found |
| 600 | one mention found |
| 800 to 1400 | nothing found. The answer is `{"is_boilerplate": false}`: no error is visible. |
| 2000 and more | repeated text that is not JSON |

So the text of one prompt has at most `INPUT_TOKEN_BUDGET` tokens, 200 by default:

- **A text** is split at blank lines, then into pieces of at most 200 tokens. Each piece is one paragraph of the
  response. A text without blank lines, such as the output of a PDF to text converter, is one block, and gives one
  paragraph per piece.
- **A paragraph** of a TEI or JSON input longer than 200 tokens is split the same way. The mentions of its parts
  are merged into one item of the response, with the number of parts in `chunks`.
- **A failed part**, whose answer is not JSON or is cut at `MAX_TOKENS`, is split in two and extracted again, three
  times at most.

A piece ends at the last sentence end (`.`, `!`, `?`, `;`, `:` before a space) of the second half of its 200 tokens.
Without a sentence end it ends at the last space, and without a space at the token limit.

The value 200 comes from one English paper: 20 paragraphs, 3543 tokens, 177 tokens per paragraph on average, read
as TEI paragraphs and as one block of text.

| Input | Prompts | Software found | Datasets found |
|---|---|---|---|
| TEI paragraphs | 21 | 2 | 3 |
| One block, pieces of 400 tokens | 10 | 1 | 0 |
| One block, pieces of 300 tokens | 13 | 1 | 1 |
| One block, pieces of 200 tokens | 21 | 2 | 2 |
| One block, pieces of 120 tokens | 34 | 2 | 4 |

This is one paper, not an evaluation.

## Response

| Field | Description |
|---|---|
| `num_paragraphs` | Number of items of `paragraphs` |
| `paragraphs` | One item per paragraph, see below |
| `mentions` | `datasets` and `software` of all paragraphs, deduplicated by lower-cased name. The first mention of a name is kept. |
| `record` | The article-level record of step 2. Absent when `analyze` is false or when there is no mention. |
| `record_error`, `record_raw_output` | Present when step 2 failed: the reason, and the answer of the model |
| `duration_seconds` | Duration of the request, with the wait for a container |
| `cost_usd` | `duration_seconds` x the price of the GPU per second |
| `gpu`, `max_inputs` | The deploy parameters of the service |

One item of `paragraphs`:

| Field | Description |
|---|---|
| `index` | Position of the paragraph, from 0 |
| `text` | The text sent to the model |
| `is_boilerplate` | Answer of the model. For a split paragraph, true when all the parts are. |
| `datasets`, `software` | The mentions, as the model writes them |
| `chunks` | Number of parts, when the paragraph was split |
| `error` | Present when the paragraph, or one of its parts, failed. The mentions of the other parts are kept. |
| `raw_output` | With `error`: the answers of the model that could not be read |

Example, for the first paragraph of the authors' example, without step 2. The mention is the answer of the model
on CPU; the duration and the cost are only there to show the format. The fields of a mention are those of the
model and are not checked by the service.

```json
{
  "num_paragraphs": 1,
  "paragraphs": [
    {
      "index": 0,
      "is_boilerplate": false,
      "datasets": [],
      "software": [
        {
          "name": "ResearchIR MAX 4.0",
          "explicit_or_implicit": "explicit",
          "role": "reused",
          "evidence": "The data was acquired using the ResearchIR MAX 4. 0 software"
        }
      ],
      "text": "For the data acquisition, a T420 FLIR thermal camera [...] to a computer."
    }
  ],
  "mentions": {
    "datasets": [],
    "software": [
      {
        "name": "ResearchIR MAX 4.0",
        "explicit_or_implicit": "explicit",
        "role": "reused",
        "evidence": "The data was acquired using the ResearchIR MAX 4. 0 software"
      }
    ]
  },
  "duration_seconds": 1.2,
  "cost_usd": 0.000266,
  "gpu": "L4",
  "max_inputs": 1
}
```

| Status | Meaning |
|---|---|
| 200 | The paper was processed. Paragraphs can still have an `error`. |
| 400 | The input cannot be read, or has no paragraph |
| 500 | The service failed |

## Client

`baguette_client.py` needs only `requests`.

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

| Option | Default | Description |
|---|---|---|
| `--endpoint` | | URL of the endpoint. Required. |
| `--text` | | Text of one paper |
| `--input_file` | | One or more files, one per paper |
| `--input_folder` | | Folder of `.txt`, `.md`, `.xml` and `.json` files, one per paper. Subfolders are not read. |
| `--output_dir` | | Write one JSON file per paper. Without it the result is printed. Required with `--input_folder`. |
| `--csv_output` | | CSV summary. A second run skips the papers that already succeeded. |
| `--threads` | 4 | In-flight requests. Keep it at or below `BAGUETTE_MAX_CONTAINERS` x `BAGUETTE_MAX_INPUTS`. |
| `--no_analyze` | | Skip step 2 |
| `--retries` | 5 | Times a paper is sent again after a server error (HTTP 5xx) or a network error. An input error (HTTP 4xx) is not retried. |
| `--retry_wait` | 20 | Seconds before the first retry. Doubled at each retry: 20, 40, 80, 160, 320. |
| `--timeout` | 1800 | Seconds for one request |

One of `--text`, `--input_file` and `--input_folder` is required.

**Output.** The file of a paper is `<output_dir>/<file name>.json`, for example `paper.tei.xml.json`. It holds the
response. The progress is written on the standard error, so a printed result can be piped. The exit code is 1
when a paper failed.

```
[4/404] paper_a.tei.xml: 25 paragraphs, 17 datasets, 2 software in 10.8s
[5/404] paper_b.tei.xml: 72 paragraphs, 44 datasets, 16 software in 18.7s (2 paragraphs FAILED: output is not a JSON object)
[6/404] paper_c.xml: FAILED (HTTP 400: {"detail":"invalid input: unclosed token: line 1, column 0"})
```

**CSV.** One row per attempt. A paper is done when one of its rows has `success` true.

| Column | Description |
|---|---|
| `document` | File name |
| `paragraphs` | Number of paragraphs of the response |
| `failed_paragraphs` | Paragraphs with an `error` |
| `datasets`, `software` | Number of mentions, deduplicated |
| `runtime_sec`, `cost_usd` | `duration_seconds` and `cost_usd` of the response |
| `success` | False when the request failed |

A paper with failed paragraphs has `success` true and is not sent again. To send it again, remove its row from the
CSV.

## Cost and speed

Measured on one L4, one paper per container, with pieces of 7040 tokens, before the pieces of 200 tokens:

| Paper | Paragraphs | Time | Cost |
|---|---|---|---|
| A | 70 | 5.9 s | $0.0013 |
| B | 99 | 6.2 s | $0.0014 |
| C | 124 | 16.6 s | $0.0037 |
| D | 69 | 20.5 s | $0.0046 |

The time follows the length of the answers more than the number of paragraphs. With pieces of 200 tokens a text
file gives more prompts; this is not measured yet on a GPU.

One paper uses a small part of an L4: the cache of vLLM was between 0.1 % and 9 % full during these runs. To
process a folder faster, raise `BAGUETTE_MAX_INPUTS` and `--threads`. The papers then share the GPU, and `cost_usd`
is an upper bound of the cost of a paper.

The first request after an idle time waits for a container: the start of vLLM takes more than one minute.

## Troubleshooting

| Symptom | Cause | Action |
|---|---|---|
| `RuntimeError: vLLM exited during startup`, with `TokenizersBackend has no attribute all_special_tokens_extended` in the log | vLLM 0.11.0 accepts any `transformers` from 4.55.2, and 5.x removes this attribute | Keep `transformers==4.57.1` in the image |
| HTTP 500, `extract() takes from 2 to 3 positional arguments but 4 were given` | A container of the previous version took the request, see [Redeploy](#redeploy) | Wait for the retries of the client, or stop the app and deploy again |
| `(1 paragraphs FAILED: output is not a JSON object)` on a text file | A version of the service before the pieces of 200 tokens | Deploy the current version |
| `FAILED: output cut at 2048 tokens` | The answer is longer than `MAX_TOKENS`, after 3 splits | Read `raw_output`; raise `MAX_TOKENS` |
| HTTP 400, `no paragraph found in the input` | Empty file, or TEI without `<p>` | Check the input |
| `Unknown GPU type` at deploy | `BAGUETTE_GPU` is not in the price table | Use a key of `GPU_COST_PER_SECOND`, or add the GPU and its price |

The logs of the service are in the Modal dashboard, or with:

```shell
modal app logs baguette-software-dataset-app
```

## Test without a GPU

The model runs on CPU with `transformers`, in 5 to 30 seconds per paragraph on 24 cores. This is how the numbers of
[Splitting](#splitting) were measured.

```shell
mkdir model
for f in config.json special_tokens_map.json tokenizer.json tokenizer_config.json model.safetensors; do
  curl -L -o model/$f https://huggingface.co/buckets/dataesr/Baguette-Software-Dataset/resolve/$f
done
pip install torch 'transformers==4.57.1'
```

```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("model")
model = AutoModelForCausalLM.from_pretrained("model", dtype=torch.float32).eval()
end = tokenizer.convert_tokens_to_ids("<|im_end|>")

paragraph = (
    "For the data acquisition, a T420 FLIR thermal camera with a 0.1 degree C thermal "
    "sensitivity was used. The data was acquired using the ResearchIR MAX 4.0 software "
    "by connecting the camera to a computer."
)
prompt = f"<|im_start|>user\n<text>{paragraph}</text><|im_end|>\n<|im_start|>assistant\n"
inputs = tokenizer(prompt, return_tensors="pt")
with torch.no_grad():
    output = model.generate(
        input_ids=inputs.input_ids,
        attention_mask=inputs.attention_mask,
        max_new_tokens=2048,
        do_sample=False,
        eos_token_id=end,
        pad_token_id=end,
    )
print(tokenizer.decode(output[0][inputs.input_ids.shape[1] :], skip_special_tokens=True))
```

The answer is the software mention of the example of [Response](#response). The answer changes with the context:
on the second sentence alone, the model returns a dataset named `data` and no software.

The functions of `inference_baguette.py` that read and split the input do not need Modal to run. `extract_paragraphs`
takes the function that sends one prompt as an argument, so the whole of step 1 can run on this model.

## Limits

- **Step 2 is indicative.** The authors validated step 1 only. Step 2 also takes all the mentions in one prompt,
  which is not split: on a paper with many mentions the record can fail, with `record_error`.
- **No gold data.** The scores of the model card are an agreement with the teacher model, not with human
  annotations.
- **The output is not constrained.** The answer is read as JSON, and the fields of a mention are not checked.
- **Pieces of a text are not paragraphs.** A piece can start in the middle of a section, and a mention cut
  between two pieces can be lost.
- **Deduplication is by name only.** `R` and `R 4.2.1` are two software mentions.
- **Languages.** English and French. The other languages are not tested by the authors.
- **License.** The model card says that the license is to be set by the owner before a public release.
