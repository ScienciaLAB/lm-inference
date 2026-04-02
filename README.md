# LM-Inference

**LM-Inference** is a repository that contains scripts to deploy **Vision-Language Models (VLMs)** specialized in
**document parsing and understanding**.
Currently, the inference is done on **[Modal.com](https://modal.com)** using their serverless infrastructure.

---

## Overview

This repository brings together multiple VLMs designed for:

- Optical Character Recognition (OCR)
- Layout and structure parsing
- Visual-text reasoning
- Question answering on images

Each model is deployed as an independent **Modal app**.

---

## Setup

The `lm_inference_utils.py` module is imported directly as a flat module (no package install needed). It provides:

- `get_cost_per_second()` — Modal GPU pricing and cost calculation
- `filter_and_aggregate()` — Layout element filtering and aggregation

---

## Deployed Models

| Model | Description | Link |
|---|---|---|
| **[Docling (Granite)](./deployments/docling)** | Document-level parser using Granite Docling for semantic structure and entity extraction. | https://github.com/docling-project/docling-serve |
| **[Docling Serve (Granite)](./deployments/docling-serve-granite)** | Docling-serve deployment on Modal with Granite backend. | https://github.com/docling-project/docling-serve |
| **[DoTS.OCR](./deployments/dots.ocr)** | Vision-Language OCR model for text extraction and layout-aware recognition. | https://github.com/rednote-hilab/dots.ocr |
| **[MinerU](./deployments/minerU)** | PDF document extraction and understanding. | https://github.com/opendatalab/MinerU |
| **[OlmOCR](./deployments/olm-ocr)** | OCR and text parsing model conversion of PDFs and other documents into plain text. | https://github.com/allenai/olmocr |
| **[Qwen2.5 VL](./deployments/qwen)** | Multimodal model for reasoning, summarization, and QA over document content. | https://huggingface.co/Qwen/Qwen2.5-VL-7B-Instruct |

## Lightweight Models (CPU-targeted)

| Model | Description | Link |
|---|---|---|
| PaddlePaddle DocLayout (S, M, L) | Lightweight document layout analysis models for structure recognition and parsing. | https://github.com/paddlepaddle/PaddleOCR |
| PaddlePaddle Block | Lightweight OCR model for text detection and recognition of main body of documents. | |

---

## Local Inference

Scripts for running PaddleOCR and LADaS layout detection locally. All scripts are run as modules from this directory:

```shell
python -m local.paddle_inference_batch ./pdfs -o ./results
python -m local.ladas_img_inference_batch ./images -o ./results --model-file ./model.pt
```

See [local/readme.md](local/readme.md) for full documentation, installation instructions, and all available options.

---

## Deployment on Modal

Each deployment folder includes:

- `model_name_inference.py` — the main Modal entrypoint
- A `README.md` file containing example `curl` commands to interact with the deployed model
