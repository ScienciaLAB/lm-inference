import base64
import os
import time
from pathlib import Path
from uuid import uuid4
import modal
from fastapi import UploadFile, File, Form
import fitz  # PyMuPDF

from lm_inference_utils import get_cost_per_second

cuda_version = "12.8.0"  # should be no greater than host CUDA version
flavor = "devel"  #  includes full CUDA toolkit
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
        "sglang[srt]>=0.4.10,<0.5.0",  # [srt] only, not [all]
        "sgl-kernel>=0.2.8,<0.3",
    )
    .pip_install(
        # Application dependencies
        "vllm>=0.4.0",
        "transformers>=4.54.0,<4.60.0",
        "huggingface_hub>=0.35.0,<1.0",
        "openai",
        "pillow",
        "opencv-python",
        "PyMuPDF",
        "fastapi[standard]>=0.115,<0.120",
        "pydantic>=2.9.2,<2.11",
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

MODEL_PATH = "Qwen/Qwen2.5-VL-7B-Instruct"
MODEL_REVISION = "cc594898137f460bfe9f0759e9844b3ce807cfb5"

MODEL_VOL_PATH = "/root/.cache/sgl"  # must be absolute!
MODEL_VOL = modal.Volume.from_name("sgl-cache", create_if_missing=True)
volumes = {MODEL_VOL_PATH: MODEL_VOL}

FAST_BOOT = True

hf_cache_vol = modal.Volume.from_name("huggingface-cache", create_if_missing=True)
app = modal.App("qwen-2.5-vl-7b-instruct-sglang")

GPU_TYPE = os.environ.get("GPU_TYPE", "l40s")
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
@modal.concurrent(max_inputs=4)
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

    @modal.fastapi_endpoint(method="POST", label="generate", docs=True)
    async def generate(
        self, question: str = Form(...), image: UploadFile = File(...)
    ) -> dict:
        from pathlib import Path
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
        from pathlib import Path
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
            # Load PDF from bytes
            pdf_doc = fitz.open(stream=pdf_content, filetype="pdf")
            total_pages = len(pdf_doc)

            print(f"📄 Processing PDF with {total_pages} pages at {dpi} DPI")

            # Extract all pages as images
            print("🖼️  Extracting pages as images...")
            page_images = []
            for page_num in range(total_pages):
                page_data = extract_page_as_image(pdf_doc, page_num, dpi)
                page_images.append(page_data)

            print(f"✅ Extracted {len(page_images)} page images")

            # Close the PDF document
            pdf_doc.close()

            return {"success": True, "pages": page_images, "total_pages": total_pages}
        except Exception as e:
            return {
                "error": f"PDF processing failed: {str(e)}",
                "success": False,
                "pages": [],
                "total_pages": 0,
            }

    @modal.fastapi_endpoint(method="POST", label="qwen-3-extract-pdf", docs=True)
    async def extract_pdf(
        self, pdf: UploadFile = File(...), dpi: int = Form(150)
    ) -> dict:
        # Start timing
        start_time = time.monotonic()

        print(f"📄 Processing PDF document {pdf.filename} with {dpi} DPI")

        # Read PDF content
        pdf_content = await pdf.read()

        # Extract PDF pages
        extraction_result = self.extract_pdf_pages(pdf_content, dpi)

        # ----- ERROR PATH (ADD COST INFO) -----
        if not extraction_result["success"]:
            duration = time.monotonic() - start_time
            try:
                cost_per_sec = get_cost_per_second(GPU_TYPE.upper())
            except Exception:
                cost_per_sec = 0.0
            total_cost = duration * cost_per_sec

            return {
                "error": extraction_result["error"],
                "total_pages": 0,
                "extracted_text": [],
                "combined_text": "",
                "success": False,
                "cost_info": {
                    "duration_seconds": round(duration, 2),
                    "cost_usd": round(total_cost, 6),
                },
            }
        # --------------------------------------

        pages = extraction_result["pages"]

        # Process each page with the model to extract text
        extracted_texts = []

        import sglang as sgl

        # Set the default backend for this execution
        sgl.set_default_backend(self.runtime)

        for page_data in pages:
            try:
                # Save the image temporarily
                temp_img_path = Path(f"/tmp/{uuid4()}.png")
                img_data = base64.b64decode(page_data["image_base64"])
                temp_img_path.write_bytes(img_data)

                # Extract text from this page using SGLang
                @sgl.function
                def extract_page_text(s, image_path):
                    s += sgl.user(
                        sgl.image(str(image_path))
                        + "Extract all text from this image. Provide the text in a clean, readable format without any additional commentary."
                    )
                    s += sgl.assistant(sgl.gen("extracted_text", max_tokens=2048))

                state = extract_page_text.run(image_path=temp_img_path)

                extracted_text = state["extracted_text"]
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

                # Clean up temporary file
                temp_img_path.unlink(missing_ok=True)

            except Exception as e:
                print(f"❌ Error processing page {page_data['page_num'] + 1}: {str(e)}")
                extracted_texts.append(
                    {
                        "page_number": page_data["page_num"] + 1,
                        "text": "",
                        "error": f"Page processing failed: {str(e)}",
                        "width": page_data["width"],
                        "height": page_data["height"],
                    }
                )

        # Combine all extracted text
        all_text = "\n\n".join(
            [page.get("text", "") for page in extracted_texts if not page.get("error")]
        )

        result = {
            "filename": pdf.filename,
            "total_pages": len(pages),
            "dpi": dpi,
            "extracted_text": extracted_texts,
            "combined_text": all_text,
            "success": True,
        }

        # ---------- ADD COST INFO (SUCCESS PATH) ----------
        duration = time.monotonic() - start_time
        try:
            cost_per_sec = get_cost_per_second(GPU_TYPE.upper())
        except Exception:
            cost_per_sec = 0.0
        total_cost = duration * cost_per_sec

        result["cost_info"] = {
            "duration_seconds": round(duration, 2),
            "cost_usd": round(total_cost, 6),
        }
        # --------------------------------------------------

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
