import asyncio
import base64
import json
import os
import time
from pathlib import Path
from typing import BinaryIO, Dict, List, Union
from uuid import uuid4

import fitz  # PyMuPDF
import modal
from fastapi import UploadFile
from openai import AsyncOpenAI

# Modal configuration
cuda_version = "12.8.0"
flavor = "devel"
operating_sys = "ubuntu22.04"
tag = f"{cuda_version}-{flavor}-{operating_sys}"
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
        # Application dependencies
        "vllm>=0.4.0",
        "transformers>=4.54.0,<4.60.0",
        "huggingface_hub>=0.35.0,<1.0",
        "flashinfer-python==0.2.0.post2",
        "openai",
        "pillow",
        "opencv-python",
        "PyMuPDF",
        "fastapi[standard]>=0.115,<0.120",
        "pydantic>=2.9.2,<2.11",
        "requests>=2.32,<3.0",
        "hf-xet>=1.1.5,<1.2",
    )
    .env({
        "HF_HUB_ENABLE_HF_TRANSFER": "1",
        "TOKENIZERS_PARALLELISM": "false",
        "PYTORCH_CUDA_ALLOC_CONF": "max_split_size_mb:512"
    })
)


# Model configuration
MODEL_PATH = "Qwen/Qwen2.5-VL-7B-Instruct"

# GPU and performance settings
GPU_TYPE = os.environ.get("GPU_TYPE", "L40S")  # L40S works well for 7B model
GPU_COUNT = int(os.environ.get("GPU_COUNT", "1"))
GPU_CONFIG = f"{GPU_TYPE}:{GPU_COUNT}"
GPU_MEMORY_UTILIZATION = 0.9
VLLM_PORT = 8000
MINUTES = 60

# Modal volumes and app setup
MODEL_VOL_PATH = Path("/cache")
MODEL_VOL = modal.Volume.from_name("sgl-cache", create_if_missing=True)
volumes = {MODEL_VOL_PATH: MODEL_VOL}
hf_cache_vol = modal.Volume.from_name("huggingface-cache", create_if_missing=True)
app = modal.App("qwen-2.5-vl-7b-instruct-vllm")


@app.function(
    image=image,
    memory=16384,  # 16GB memory for 7B model
    cpu=4,
    timeout=600,
    volumes={
        "/root/.cache/huggingface": hf_cache_vol,
    },
    secrets=[modal.Secret.from_name("document-qa-api-key")]
)
def extract_pdf_pages(pdf_content: bytes, dpi: int = 150) -> List[Dict]:
    """
    Extract all pages from a PDF as base64 encoded images
    """
    try:
        pdf_doc = fitz.open(stream=pdf_content, filetype="pdf")
        total_pages = len(pdf_doc)
        print(f"📄 Processing PDF with {total_pages} pages at {dpi} DPI")
        page_images = []

        for page_num in range(total_pages):
            page = pdf_doc.load_page(page_num)
            mat = fitz.Matrix(dpi/72, dpi/72)
            pix = page.get_pixmap(matrix=mat)
            img_data = pix.tobytes("png")
            page_data = {
                "page_number": page_num + 1,
                "image_base64": base64.b64encode(img_data).decode('utf-8'),
                "width": pix.width,
                "height": pix.height,
                "dpi": dpi
            }
            page_images.append(page_data)

        pdf_doc.close()
        print(f"✅ Extracted {len(page_images)} page images")

        return page_images

    except Exception as e:
        print(f"❌ PDF processing failed: {str(e)}")
        return []


@app.cls(
    gpu=GPU_CONFIG,
    timeout=20 * MINUTES,
    scaledown_window=20 * MINUTES,  # ✅ updated from container_idle_timeout
    image=image,
    volumes=volumes,
)
@modal.concurrent(max_inputs=100)
class VLMMModel:
    model_name: str = modal.parameter()  # ✅ replaces __init__ constructor

    @modal.enter()
    def start_vllm_client(self):
        self.client = AsyncOpenAI(
            base_url=f"http://localhost:{VLLM_PORT}/v1",
            api_key=os.environ.get("API_KEY", None)
        )
        print("🚀 vLLM client initialized and ready")

    @modal.fastapi_endpoint(method="POST", docs=True)  # ✅ updated decorator
    async def generate(self, question: str, image: Union[Path, str, BinaryIO]):
        from pathlib import Path

        start = time.monotonic_ns()
        request_id = uuid4()
        print(f"Generating response to request {request_id}")

        if isinstance(image, (str, Path)):
            image_path = Path(image)
            with open(image_path, 'rb') as f:
                image_data = f.read()
        else:
            image_data = image.read()

        image_base64 = base64.b64encode(image_data).decode('utf-8')

        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": question},
                    {"type": "image_url", "image_url": {
                        "url": f"data:image/png;base64,{image_base64}"}
                     }
                ]
            }
        ]

        try:
            response = await self.client.chat.completions.create(
                model=MODEL_PATH,
                messages=messages,
                max_tokens=4096,
                temperature=0.1,
                stream=False
            )

            answer = response.choices[0].message.content
            print(f"request {request_id} completed in {round((time.monotonic_ns() - start) / 1e9, 2)} seconds")
            return answer

        except Exception as e:
            print(f"Error in generate: {str(e)}")
            return f"Error processing request: {str(e)}"

    @modal.fastapi_endpoint(method="POST", docs=True)  # ✅ updated decorator
    async def extract_document_structure(self, pages: List[Dict]) -> Dict:
        """
        Extract document structure from PDF pages using vLLM
        """
        print(f"🔍 Extracting structure from {len(pages)} pages using vLLM")
        tasks = []

        for page_data in pages:
            task = self._extract_page_structure(
                page_data["image_base64"],
                page_data["page_number"]
            )
            tasks.append(task)

        results = await asyncio.gather(*tasks, return_exceptions=True)

        document_structure = {
            "document_info": {
                "total_pages": len(pages),
                "extraction_timestamp": time.time(),
                "model_used": MODEL_PATH
            },
            "pages": [],
            "summary": {
                "total_passages": 0,
                "total_references": 0,
                "structure_counts": {
                    "header_passages": 0,
                    "body_passages": 0,
                    "annex_passages": 0
                }
            }
        }

        for i, result in enumerate(results):
            if isinstance(result, Exception) or not result.get("success", False):
                error_msg = str(result) if isinstance(result, Exception) else result.get("error", "Unknown error")
                page_result = {
                    "page_number": pages[i]["page_number"],
                    "success": False,
                    "error": error_msg,
                    "header": {"passages": []},
                    "body": {"passages": []},
                    "annex": {"passages": []}
                }

            else:
                page_result = result
                if result["success"]:
                    data = result["data"]
                    for section in ["header", "body", "annex"]:
                        if section in data:
                            passages = data[section].get("passages", [])
                            document_structure["summary"]["structure_counts"][f"{section}_passages"] += len(passages)
                            document_structure["summary"]["total_passages"] += len(passages)
                            for passage in passages:
                                references = passage.get("references", [])
                                document_structure["summary"]["total_references"] += len(references)
                else:
                    page_result = {
                        "page_number": pages[i]["page_number"],
                        "success": False,
                        "error": result.get("error", "Unknown error"),
                        "header": {"passages": []},
                        "body": {"passages": []},
                        "annex": {"passages": []}
                    }

            document_structure["pages"].append(page_result)

        print(f"✅ Document structure extraction completed")
        print(f"📊 Found {document_structure['summary']['total_passages']} total passages")
        print(f"📊 Found {document_structure['summary']['total_references']} total references")

        return document_structure

    async def _extract_page_structure(self, image_base64: str, page_number: int) -> Dict:
        """
        Extract structured text from a single page image using vLLM
        """
        structure_prompt = """<your_prompt_contents>"""  # keep this same

        try:
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": structure_prompt},
                        {"type": "image_url", "image_url": {
                            "url": f"data:image/png;base64,{image_base64}"}
                         }
                    ]
                }
            ]

            response = await self.client.chat.completions.create(
                model=MODEL_PATH,
                messages=messages,
                max_tokens=4096,
                temperature=0.1,
                stream=False
            )

            content = response.choices[0].message.content
            try:
                start_idx = content.find('{')
                end_idx = content.rfind('}') + 1
                if start_idx != -1 and end_idx != 0:
                    json_str = content[start_idx:end_idx]
                    parsed_result = json.loads(json_str)
                    parsed_result["page_number"] = page_number
                    for section in ["header", "body", "annex"]:
                        if section not in parsed_result:
                            parsed_result[section] = {"passages": []}
                        elif "passages" not in parsed_result[section]:
                            parsed_result[section]["passages"] = []

                    return {
                        "success": True,
                        "data": parsed_result,
                        "raw_response": content
                    }
                else:
                    raise ValueError("No JSON found in response")

            except (json.JSONDecodeError, ValueError) as e:
                return {
                    "success": False,
                    "error": f"JSON parsing failed: {str(e)}",
                    "raw_response": content,
                    "page_number": page_number,
                    "data": {
                        "page_number": page_number,
                        "header": {"passages": []},
                        "body": {"passages": []},
                        "annex": {"passages": []}
                    }
                }

        except Exception as e:
            return {
                "success": False,
                "error": f"Structure extraction failed: {str(e)}",
                "page_number": page_number,
                "data": {
                    "page_number": page_number,
                    "header": {"passages": []},
                    "body": {"passages": []},
                    "annex": {"passages": []}
                }
            }


@app.function(
    image=image,
    memory=16384,
    cpu=4,
    timeout=600,
    volumes={
        "/root/.cache/huggingface": hf_cache_vol,
    },
    secrets=[modal.Secret.from_name("document-qa-api-key")]
)
@modal.web_endpoint(method="POST")
async def process_pdf_document_structure(pdf: UploadFile, dpi: int = 150) -> Dict:
    """
    Main entry point for PDF document structure extraction using vLLM
    
    Args:
        pdf: Uploaded PDF file
        dpi: Resolution for image extraction
        
    Returns:
        Complete document structure in JSON format
    """
    print(f"📄 Starting document structure extraction for: {pdf.filename}")
    
    try:
        # Step 1: Extract PDF pages as images
        print("🖼️  Step 1: Extracting PDF pages as images...")
        pdf_content = pdf.file.read()
        pages = extract_pdf_pages.remote(pdf_content, dpi)
        
        if not pages:
            return {
                "success": False,
                "error": "Failed to extract pages from PDF",
                "document_info": {
        "filename": pdf.filename,
                    "total_pages": 0
                }
            }
        
        print(f"✅ Extracted {len(pages)} pages")
        
        # Step 2: Extract document structure using vLLM
        print("🧠 Step 2: Extracting document structure with vLLM...")
        
        # Get the model instance and process pages
        with VLMMModel() as model:
            document_structure = await model.extract_document_structure(pages)
        
        # Add filename to document info
        document_structure["document_info"]["filename"] = pdf.filename
        document_structure["document_info"]["dpi"] = dpi
        document_structure["success"] = True
        
        print(f"🎉 Document structure extraction completed successfully!")
        
        return document_structure
        
    except Exception as e:
        print(f"❌ Document processing failed: {str(e)}")
        return {    
            "success": False,
            "error": f"Document processing failed: {str(e)}",
            "document_info": {
                "filename": pdf.filename,
                "total_pages": 0
            }
        }