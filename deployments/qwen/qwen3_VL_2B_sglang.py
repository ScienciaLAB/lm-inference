import asyncio
import io
import os
import time
from pathlib import Path
from uuid import uuid4

import modal
from fastapi import UploadFile, File, Form

from lm_inference_utils import get_cost_per_second

cuda_version = "12.8.0"
flavor = "devel"
operating_sys = "ubuntu22.04"
tag = f"{cuda_version}-{flavor}-{operating_sys}"
vllm_cache_vol = modal.Volume.from_name("vllm-cache", create_if_missing=True)


image = (
    modal.Image.from_registry(f"nvidia/cuda:{tag}", add_python="3.11")
    .apt_install("libnuma-dev")
    .pip_install(
        # Core packages first
        "torch>=2.7.0,<2.8",
        "numpy<2",
        extra_index_url="https://flashinfer.ai/whl/cu124/torch2.7",
    )
    .pip_install(
        # SGLang with only needed extras (not [all])
        "sglang[srt]>=0.5.0",  # [srt] only, not [all]
        "sgl-kernel>=0.3.0",
    )
    .pip_install(
        # Application dependencies
        "transformers>=4.56.0",
        "huggingface_hub>=0.35.0,<1.0",
        "openai",
        "pillow",
        "opencv-python",
        "PyMuPDF",
        "fastapi[standard]>=0.115,<0.120",
        "pydantic>=2.10.0",
        "requests>=2.32,<3.0",
        "hf-xet>=1.1.5,<1.2",
    )
    .env(
        {
            "HF_HUB_ENABLE_HF_TRANSFER": "1",
            "TOKENIZERS_PARALLELISM": "false",
            "PYTORCH_CUDA_ALLOC_CONF": "max_split_size_mb:512",
        }
    )
    .add_local_file("lm_inference_utils.py", "/root/lm_inference_utils.py")

)

MODEL_PATH = "Qwen/Qwen3-VL-2B-Instruct"
MODEL_REVISION = "89644892e4d85e24eaac8bacfd4f463576704203"

MODEL_VOL_PATH = "/root/.cache/sgl"  # must be absolute!
MODEL_VOL = modal.Volume.from_name("sgl-cache", create_if_missing=True)
volumes = {MODEL_VOL_PATH: MODEL_VOL}

FAST_BOOT = True

hf_cache_vol = modal.Volume.from_name("huggingface-cache", create_if_missing=True)
app = modal.App("qwen-3-vl-2b-instruct-sglang")

GPU_TYPE = os.environ.get("GPU_TYPE", "A100-80GB")
GPU_COUNT = int(os.environ.get("GPU_COUNT", 1))

GPU_CONFIG = f"{GPU_TYPE}:{GPU_COUNT}"

SGL_LOG_LEVEL = "info"  # Changed to info for better debugging

MINUTES = 60  # seconds

MODEL_CHAT_TEMPLATE = "qwen2-vl"


@app.cls(
    gpu=GPU_CONFIG,
    timeout=20 * MINUTES,
    scaledown_window=20 * MINUTES,
    image=image,
    volumes=volumes,
    max_containers=4,
)
@modal.concurrent(max_inputs=100)
class Model:
    @modal.enter()  # what should a container do after it starts but before it gets input?
    def start_runtime(self):
        """Starts an SGL runtime to execute inference."""
        import sglang as sgl

        self.runtime = sgl.Runtime(
            model_path=MODEL_PATH,
            tokenizer_path=MODEL_PATH,  # Using same path for tokenizer
            tp_size=GPU_COUNT,  # tensor parallel size, number of GPUs to split the model over
            log_level=SGL_LOG_LEVEL,
        )
        self.runtime.endpoint.chat_template = sgl.lang.chat_template.get_chat_template(
            MODEL_CHAT_TEMPLATE
        )
        sgl.set_default_backend(self.runtime)

    @modal.fastapi_endpoint(method="POST", label="qwen-3-generate", docs=True)
    async def generate(
        self, question: str = Form(...), image: UploadFile = File(...)
    ) -> dict:
        import sglang as sgl

        start = time.monotonic_ns()
        request_id = uuid4()
        print(f"Generating response to request {request_id}")

        # Read uploaded file and save temporarily
        image_bytes = await image.read()
        image_path = Path(f"/tmp/{uuid4()}.png")
        image_path.write_bytes(image_bytes)

        @sgl.function
        def image_qa(s, image_path, question):
            s += sgl.user(sgl.image(str(image_path)) + question)
            s += sgl.assistant(sgl.gen("answer", max_tokens=1024))

        state = image_qa.run(image_path=image_path, question=question)

        print(
            f"request {request_id} completed in {round((time.monotonic_ns() - start) / 1e9, 2)} seconds"
        )

        # Clean up temporary file
        image_path.unlink(missing_ok=True)

        return {
            "answer": state["answer"],
            "request_id": str(request_id),
            "processing_time": round((time.monotonic_ns() - start) / 1e9, 2),
        }

    @modal.fastapi_endpoint(method="POST", label="qwen-3-extract-text", docs=True)
    async def extract_text_from_image(self, image: UploadFile = File(...)) -> dict:
        import sglang as sgl

        start = time.monotonic_ns()
        request_id = uuid4()
        print(f"Extracting text from image for request {request_id}")

        # Read uploaded file and save temporarily
        image_bytes = await image.read()
        image_path = Path(f"/tmp/{uuid4()}.png")
        image_path.write_bytes(image_bytes)

        @sgl.function
        def extract_text(s, image_path):
            s += sgl.user(
                sgl.image(str(image_path))
                + "Extract all text from this image. Provide the text in a clean, readable format without any additional commentary."
            )
            s += sgl.assistant(sgl.gen("extracted_text", max_tokens=2048))

        state = extract_text.run(image_path=image_path)

        print(
            f"text extraction for request {request_id} completed in {round((time.monotonic_ns() - start) / 1e9, 2)} seconds"
        )
        print(f"Extracted text: {state['extracted_text'][:200]}...")  # First 200 chars

        # Clean up temporary file
        image_path.unlink(missing_ok=True)

        return {
            "extracted_text": state["extracted_text"],
            "request_id": str(request_id),
            "processing_time": round((time.monotonic_ns() - start) / 1e9, 2),
        }

    def extract_pdf_pages(self, pdf_content: bytes, dpi: int = 150) -> dict:
        """
        Extract all pages from a PDF as images.

        Args:
            pdf_content: PDF file content as bytes
            dpi: Resolution for image extraction (default: 150)

        Returns:
            Dictionary containing extracted images and metadata for each page
        """
        import fitz  # PyMuPDF
        import base64

        def extract_page_as_image(pdf_doc, page_num, dpi):
            """Extract a single page as base64 image"""
            page = pdf_doc.load_page(page_num)
            mat = fitz.Matrix(dpi / 72, dpi / 72)  # Scale factor for DPI
            pix = page.get_pixmap(matrix=mat)
            img_data = pix.tobytes("png")

            return {
                "page_num": page_num,
                "image_base64": base64.b64encode(img_data).decode("utf-8"),
                "width": pix.width,
                "height": pix.height,
            }

        try:
            # We could use pdf2image as alternative
            pdf_doc = fitz.open(stream=pdf_content, filetype="pdf")
            total_pages = len(pdf_doc)

            print(f"📄 Processing PDF with {total_pages} pages at {dpi} DPI")

            # Extract all pages as images
            print("🖼️  Extracting pages as images...")
            pdf_conversion_start = time.monotonic_ns()
            page_images = []
            for page_num in range(total_pages):
                page_data = extract_page_as_image(pdf_doc, page_num, dpi)
                page_images.append(page_data)
            pdf_conversion_time = (time.monotonic_ns() - pdf_conversion_start) / 1e9

            print(f"✅ Extracted {len(page_images)} page images")
            print(
                f"⏱️ PDF to image conversion took {round(pdf_conversion_time, 2)} seconds"
            )

            # Close the PDF document
            pdf_doc.close()

            return {
                "success": True,
                "pages": page_images,
                "total_pages": total_pages,
                "pdf_conversion_time": round(pdf_conversion_time, 2),
            }
        except Exception as e:
            return {
                "error": f"PDF processing failed: {str(e)}",
                "success": False,
                "pages": [],
                "total_pages": 0,
                "pdf_conversion_time": 0,
            }

    @modal.fastapi_endpoint(method="POST", label="qwen-3-extract-pdf", docs=True)
    async def extract_pdf(
        self, pdf: UploadFile = File(...), dpi: int = Form(150)
    ) -> dict:
        overall_conversion_start = time.monotonic()

        print(f"📄 Processing PDF document {pdf.filename} with {dpi} DPI")

        # Read PDF content
        pdf_content = await pdf.read()

        overall_conversion_start = time.monotonic_ns()

        # Extract PDF pages
        extraction_result = self.extract_pdf_pages(pdf_content, dpi)

        if not extraction_result["success"]:
            duration = time.monotonic() - overall_conversion_start
            try:
                cost_per_sec = get_cost_per_second(GPU_TYPE.upper())
            except Exception:
                cost_per_sec = 0.0
            total_cost = duration * cost_per_sec

            return {
                "error": extraction_result["error"],
                "total_pages": 0,
                "extracted_text": [],
                "success": False,
                "cost_info": {
                    "duration_seconds": round(duration, 2),
                    "cost_usd": round(total_cost, 6),
                },
            }

        pages = extraction_result["pages"]

        # image_conversion_time = (time.monotonic_ns() - overall_conversion_start) / 1e9
        pdf_conversion_start = time.monotonic_ns()

        extracted_texts = []

        import sglang as sgl
        import base64

        # Set the default backend for this execution
        sgl.set_default_backend(self.runtime)

        tasks = []
        for page_data in pages:
            img_data = base64.b64decode(page_data["image_base64"])
            image_binary = io.BytesIO(img_data)

            image = UploadFile(
                filename=f"page_{page_data['page_num']}.png", file=image_binary
            )

            task = self.generate.local(
                image=image,
                question="Extract all text from this image. Provide the text in a clean, readable format without any additional commentary.",
            )
            tasks.append((task, page_data))

        responses = await asyncio.gather(
            *[task for task, _ in tasks], return_exceptions=True
        )

        for (task, page_data), response in zip(tasks, responses):
            if isinstance(response, Exception):
                print(
                    f"❌ Error processing page {page_data['page_num'] + 1}: {str(response)}"
                )
                extracted_texts.append(
                    {
                        "page_number": page_data["page_num"] + 1,
                        "text": "",
                        "error": str(response),
                        "width": page_data["width"],
                        "height": page_data["height"],
                    }
                )
            else:
                extracted_text = response[
                    "answer"
                ]  # generate() returns "answer", not "extracted_text"
                print(
                    f"✅ Page {page_data['page_num'] + 1} extracted text: {extracted_text[:100]}..."
                )

                extracted_texts.append(
                    {
                        "page_number": page_data["page_num"] + 1,
                        "text": extracted_text,
                        "width": page_data["width"],
                        "height": page_data["height"],
                    }
                )

                print(f"✅ Page {page_data['page_num'] + 1} processed successfully")

        # Combine all extracted text
        all_text = "\n\n".join(
            [page.get("text", "") for page in extracted_texts if not page.get("error")]
        )

        pdf_conversion_time = (time.monotonic_ns() - pdf_conversion_start) / 1e9
        result = {
            "filename": pdf.filename,
            "total_pages": len(pages),
            "dpi": dpi,
            "extracted_text": extracted_texts,
            "combined_text": all_text,
            "success": True,
            "pdf_conversion_time": pdf_conversion_time,
        }

        overall_duration = time.monotonic() - overall_conversion_start
        try:
            cost_per_sec = get_cost_per_second(GPU_TYPE.upper())
        except Exception:
            cost_per_sec = 0.0
        total_cost = overall_duration * cost_per_sec

        result["cost_info"] = {
            "duration_seconds": round(overall_duration, 2),
            "cost_usd": round(total_cost, 6),
        }

        print(
            f"✅ PDF processing completed. Combined text length: {len(all_text)} characters"
        )
        print(f"First 200 chars of combined text: {all_text[:200]}...")
        print(f"💰 Cost: {result['cost_info']}")

        return result

    @modal.exit()  # what should a container do before it shuts down?
    def shutdown_runtime(self):
        self.runtime.shutdown()


if __name__ == "__main__":
    app.serve()
