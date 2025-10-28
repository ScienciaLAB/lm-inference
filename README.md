# 🧠 Im-Inference

**Im-Inference** is a repository that contains scripts to deploy **Vision-Language Models (VLMs)** specialized in **document parsing and understanding** on **[Modal.com](https://modal.com)** — enabling scalable, cloud-native inference for complex document intelligence tasks.  
We serve models such as **DoTS.OCR**, **OLMOCR**, **Docling (Granite)**, and **Qwen**.

---

🌐 **Purpose**  
The goal of **Im-Inference** is to build a **cloud-native, production-ready environment** for **document AI**, deploying vision and language models seamlessly through **Modal’s serverless infrastructure**.

---

## 🚀 Overview

This repository brings together multiple VLMs designed for:
- Optical Character Recognition (OCR)
- Layout and structure parsing
- Visual-text reasoning
- Question answering on images

Each model is deployed as an independent **Modal app**, forming a modular pipeline for document understanding workflows.

---

## 🧩 Deployed Models

| Model | Description | Link |
|--------|--------------|------|
| **[Docling (Granite)](./Deployments/docling)** | Document-level parser using Granite Docling for semantic structure and entity extraction. |https://github.com/docling-project/docling-serve|
| **[DoTS.OCR](./Deployments/dots.ocr)** | Vision-Language OCR model for text extraction and layout-aware recognition. |https://github.com/rednote-hilab/dots.ocr|
| **[OLMOCR](./Deployments/olmOCR)** | OCR and text parsing model conversion of PDFs and other documents into plain text. |https://github.com/allenai/olmocr|
| **[Qwen](./Deployments/qwen)** | Multimodal model for reasoning, summarization, and QA over document content. |https://huggingface.co/Qwen/Qwen2-7B|

---

## ⚙️ Deployment on Modal

Each deployment folder includes:
- `model_name_inference.py` — the main Modal entrypoint  
- A `README.md` file containing example `curl` commands to interact with the deployed model


Maintained by [ScienciaLAB](https://www.sciencialab.com)
