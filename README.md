# 🧠 LM-Inference

**LM-Inference** is a repository that contains scripts to deploy **Vision-Language Models (VLMs)** specialized in *
*document parsing and understanding**.
Currently, the inference is done on **[Modal.com](https://modal.com)** using their serverless infrastructure.

---

## 🚀 Overview

This repository brings together multiple VLMs designed for:

- Optical Character Recognition (OCR)
- Layout and structure parsing
- Visual-text reasoning
- Question answering on images

Each model is deployed as an independent **Modal app**.

---

## 🧩 Deployed Models

| Model                                          | Description                                                                               | Link                                               |
|------------------------------------------------|-------------------------------------------------------------------------------------------|----------------------------------------------------|
| **[Docling (Granite)](./Deployments/docling)** | Document-level parser using Granite Docling for semantic structure and entity extraction. | https://github.com/docling-project/docling-serve   |
| **[DoTS.OCR](./Deployments/dots.ocr)**         | Vision-Language OCR model for text extraction and layout-aware recognition.               | https://github.com/rednote-hilab/dots.ocr          |
| **[OLMOCR](./Deployments/olmOCR)**             | OCR and text parsing model conversion of PDFs and other documents into plain text.        | https://github.com/allenai/olmocr                  |
| **[Qwen2.5 VL](./Deployments/qwen)**           | Multimodal model for reasoning, summarization, and QA over document content.              | https://huggingface.co/Qwen/Qwen2.5-VL-7B-Instruct |

## 🧩 Lightweight Models (CPU-targeted)

| Model                            | Description                                                                         | Link                                      |
|----------------------------------|-------------------------------------------------------------------------------------|-------------------------------------------|
| PaddlePaddle DocLayout (S, M, L) | Lightweight document layout analysis models for structure recognition and parsing.  | https://github.com/paddlepaddle/PaddleOCR |
| PaddlePaddle Block               | Lightweight OCR model for text detection and recognition of main body of documents. |

---

## ⚙️ Deployment on Modal

Each deployment folder includes:

- `model_name_inference.py` — the main Modal entrypoint
- A `README.md` file containing example `curl` commands to interact with the deployed model
