
import asyncio
import base64
import json
import os
import time
from pathlib import Path
from typing import BinaryIO, List, Union
from uuid import uuid4

import modal
from fastapi import UploadFile
from openai import AsyncOpenAI
import fitz  # PyMuPDF

cuda_version = "12.8.0"  # should be no greater than host CUDA version
flavor = "devel"  #  includes full CUDA toolkit
operating_sys = "ubuntu22.04"
tag = f"{cuda_version}-{flavor}-{operating_sys}"

image = (
    modal.Image.from_registry(f"nvidia/cuda:{tag}", add_python="3.11")
    .pip_install(
        "vllm>=0.4.0",
        "huggingface_hub[hf_transfer]==0.26.2",
        "flashinfer-python==0.2.0.post2",  # pinning, very unstable
        "pillow",  # Required for image processing in VL models
        "opencv-python",  # Additional image processing support
        "PyMuPDF",  # PDF processing
        "transformers==4.54.1",
        "numpy<2",
        "fastapi[standard]==0.115.4",
        "pydantic==2.9.2",
        "requests==2.32.3",
        "starlette==0.41.2",
        "torch==2.7.1",
        "sglang[all]==0.4.10.post2",
        "sgl-kernel==0.2.8",
        "hf-xet==1.1.5",
        extra_index_url="https://flashinfer.ai/whl/cu124/torch2.5",
    )
    .env({
        "HF_HUB_ENABLE_HF_TRANSFER": "1",  # faster model transfers
        "TOKENIZERS_PARALLELISM": "false",  # Avoid tokenizer warnings
        "PYTORCH_CUDA_ALLOC_CONF": "max_split_size_mb:512"  # Optimize CUDA memory
    })
)

MODEL_PATH = "Qwen/Qwen2.5-VL-7B-Instruct"
MODEL_REVISION = "cc594898137f460bfe9f0759e9844b3ce807cfb5"

# Model-specific configuration for Qwen2.5-VL-7B-Instruct
MODEL_CONFIG = {
    "max_model_len": 32768,  # Maximum sequence length
    "max_num_batched_tokens": 8192,  # Batch size for tokens
    "max_num_seqs": 256,  # Maximum concurrent sequences
    "rope_scaling_factor": 2.0,  # RoPE scaling factor
    "rope_original_max_position": 32768,  # Original position embeddings
    "rope_type": "yarn",  # RoPE type for extended context
    "image_limit": 2,  # Maximum images per request
    "video_limit": 0,  # No video support
}

MODEL_VOL_PATH = Path()
MODEL_VOL = modal.Volume.from_name("sgl-cache", create_if_missing=True)
volumes = {MODEL_VOL_PATH: MODEL_VOL}
FAST_BOOT = True

hf_cache_vol = modal.Volume.from_name("huggingface-cache", create_if_missing=True)
app = modal.App("qwen-2.5-vl-7b-instruct-sglang")

GPU_TYPE = os.environ.get("GPU_TYPE", "l40s")
GPU_COUNT = os.environ.get("GPU_COUNT", 1)

GPU_CONFIG = f"{GPU_TYPE}:{GPU_COUNT}"

SGL_LOG_LEVEL = "error"  # try "debug" or "info" if you have issues

# GPU configuration - L40S recommended for 7B model
MINUTES = 60  # seconds
VLLM_PORT = 8000

# Memory and performance settings
GPU_MEMORY_UTILIZATION = 0.9  # Use 90% of GPU memory
SWAP_SPACE = 4  # GB of swap space


MODEL_PATH = "Qwen/Qwen2.5-VL-7B-Instruct"
MODEL_REVISION = "cc594898137f460bfe9f0759e9844b3ce807cfb5"
TOKENIZER_PATH = "Qwen/Qwen2.5-VL-7B-Instruct"
MODEL_CHAT_TEMPLATE = "qwen2-vl"


@app.cls(
    gpu=GPU_CONFIG,
    timeout=20 * MINUTES,
    container_idle_timeout=20 * MINUTES,
    image=image,
    volumes=volumes,
)
class Model:
    @modal.enter()  # what should a container do after it starts but before it gets input?
    def start_runtime(self):
        """Starts an SGL runtime to execute inference."""
        import sglang as sgl

        self.runtime = sgl.Runtime(
            model_path=MODEL_PATH,
            tokenizer_path=TOKENIZER_PATH,
            tp_size=GPU_COUNT,  # t_ensor p_arallel size, number of GPUs to split the model over
            log_level=SGL_LOG_LEVEL,
        )
        self.runtime.endpoint.chat_template = sgl.lang.chat_template.get_chat_template(
            MODEL_CHAT_TEMPLATE
        )
        sgl.set_default_backend(self.runtime)

    @modal.web_endpoint(method="POST", docs=True)
    def generate(self, question: str, image: Union[Path, str, BinaryIO]):
        from pathlib import Path
        import sglang as sgl

        start = time.monotonic_ns()
        request_id = uuid4()
        print(f"Generating response to request {request_id}")

        image_path = Path(f"/tmp/{uuid4()}-{image_filename}")
        image_path.write_bytes(response.content)

        @sgl.function
        def image_qa(s, image_path, question):
            s += sgl.user(sgl.image(str(image_path)) + question)
            s += sgl.assistant(sgl.gen("answer"))

        state = image_qa.run(
            image_path=image_path, 
            question=question
        )

        # show the question and image in the terminal for demonstration purposes
        print(Colors.BOLD, Colors.GRAY, "Question: ", question, Colors.END, sep="")
        terminal_image = from_file(image_path)
        terminal_image.draw()
        print(
            f"request {request_id} completed in {round((time.monotonic_ns() - start) / 1e9, 2)} seconds"
        )

        return state["answer"]

    @modal.exit()  # what should a container do before it shuts down?
    def shutdown_runtime(self):
        self.runtime.shutdown()




@app.function(
    image=vllm_image,
    memory=8000,  # Reduced memory since no GPU needed
    cpu=4,
    timeout=600,
    volumes={
        "/root/.cache/huggingface": hf_cache_vol,
        "/root/.cache/vllm": vllm_cache_vol,
    },
    secrets=[modal.Secret.from_name("document-qa-api-key")]
)
def extract_pdf_pages(pdf_content: bytes, dpi: int = 150) -> dict:
    """
    Extract all pages from a PDF as images and process them in parallel with VLLM.
    
    Args:
        pdf_content: PDF file content as bytes
        dpi: Resolution for image extraction (default: 150)
    
    Returns:
        Dictionary containing extracted text and metadata for each page
    """
    import fitz  # PyMuPDF
    import base64

    def extract_page_as_image(pdf_doc, page_num, dpi):
        """Extract a single page as base64 image"""
        page = pdf_doc.load_page(page_num)
        mat = fitz.Matrix(dpi/72, dpi/72)  # Scale factor for DPI
        pix = page.get_pixmap(matrix=mat)
        img_data = pix.tobytes("png")
        
        return {
            "page_num": page_num,
            "image_base64": base64.b64encode(img_data).decode('utf-8'),
            "width": pix.width,
            "height": pix.height
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
        
        return page_images
        
        
    except Exception as e:
        return {
            "error": f"PDF processing failed: {str(e)}",
            "total_pages": 0,
            "pages": []
        }


async def call_vllm_with_openai_client_async(messages: list, base_url: str = None, api_key: str = None, model: str = "Qwen/Qwen2.5-VL-7B-Instruct"):
    """
    Call vLLM server using AsyncOpenAI client through HTTP.
    
    Args:
        messages: List of messages for the chat completion
        base_url: Base URL of the vLLM server (defaults to localhost:8000)
        api_key: API key for authentication
        model: Model name to use
    
    Returns:
        Response from the vLLM server
    """
    # Default to localhost if no base_url provided
    if base_url is None:
        base_url = "http://localhost:8000/v1"
    
    # Initialize AsyncOpenAI client
    client = AsyncOpenAI(
        base_url=base_url,
        api_key=api_key or os.environ.get("API_KEY", None)
    )
    
    try:
        # Call the chat completion endpoint asynchronously
        response = await client.chat.completions.create(
            model=model,
            messages=messages,
            max_tokens=4096,
            temperature=0.1,
            stream=False
        )
        
        return {
            "success": True,
            "content": response.choices[0].message.content,
            "usage": response.usage.dict() if response.usage else None
        }
        
    except Exception as e:
        return {
            "success": False,
            "error": f"vLLM API call failed: {str(e)}",
            "content": None
        }
    finally:
        await client.close()

async def process_pages_async(pages: list, vllm_base_url: str, api_key: str, model: str, max_concurrent: int = None):
    """
    Process PDF pages asynchronously with controlled concurrency.
    
    Args:
        pages: List of page data from PDF extraction
        vllm_base_url: Base URL for vLLM server
        api_key: API key for authentication
        model: Model name to use
        max_concurrent: Maximum concurrent requests (defaults to vLLM config)
    
    Returns:
        List of processed page results
    """
    if max_concurrent is None:
        # Use vLLM's max_num_seqs but be conservative for vision tasks
        max_concurrent = min(MODEL_CONFIG["max_num_seqs"] // 4, 8)  # Conservative limit
    
    print(f"🔄 Processing {len(pages)} pages with max {max_concurrent} concurrent requests")
    
    async def process_single_page(page_data):
        """Process a single page with vLLM"""
        try:
            # Create message for vision-language model
            messages = [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": "Please extract and transcribe all text from this image. Provide the text in a clean, readable format."
                        },
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/png;base64,{page_data['image_base64']}"
                            }
                        }
                    ]
                }
            ]
            
            # Call vLLM using async OpenAI client
            vllm_response = await call_vllm_with_openai_client_async(
                messages=messages,
                base_url=vllm_base_url,
                api_key=api_key,
                model=model
            )
            
            if vllm_response["success"]:
                return {
                    "page_number": page_data["page_num"] + 1,  # 1-indexed for user
                    "text": vllm_response["content"],
                    "width": page_data["width"],
                    "height": page_data["height"],
                    "usage": vllm_response.get("usage")
                }
            else:
                return {
                    "page_number": page_data["page_num"] + 1,
                    "text": "",
                    "error": vllm_response["error"],
                    "width": page_data["width"],
                    "height": page_data["height"]
                }
                
        except Exception as e:
            return {
                "page_number": page_data["page_num"] + 1,
                "text": "",
                "error": f"Page processing failed: {str(e)}",
                "width": page_data["width"],
                "height": page_data["height"]
            }
    
    # Create semaphore to limit concurrent requests
    semaphore = asyncio.Semaphore(max_concurrent)
    
    async def process_with_semaphore(page_data):
        async with semaphore:
            return await process_single_page(page_data)
    
    # Process all pages concurrently with controlled concurrency
    tasks = [process_with_semaphore(page_data) for page_data in pages]
    results = await asyncio.gather(*tasks, return_exceptions=True)
    
    # Handle any exceptions that occurred
    processed_results = []
    for i, result in enumerate(results):
        if isinstance(result, Exception):
            processed_results.append({
                "page_number": pages[i]["page_num"] + 1,
                "text": "",
                "error": f"Task failed with exception: {str(result)}",
                "width": pages[i]["width"],
                "height": pages[i]["height"]
            })
        else:
            processed_results.append(result)
    
    return processed_results


@app.function(
    image=vllm_image,
    memory=8000,
    cpu=4,
    timeout=600,
    volumes={
        "/root/.cache/huggingface": hf_cache_vol,
        "/root/.cache/vllm": vllm_cache_vol,
    },
    secrets=[modal.Secret.from_name("document-qa-api-key")]
)
@modal.web_endpoint(method="POST")
async def extract_pdf(pdf: UploadFile, dpi=150):
    """
    Web endpoint for PDF extraction using vLLM via OpenAI client with async processing.
    """
    
    print(f"📄 Processing PDF document {pdf.filename} with {dpi} DPI")
    pdf_content = pdf.file.read()
    
    # Call the PDF extraction function
    pages = extract_pdf_pages.remote(pdf_content, dpi)
    
    if "error" in pages:
        return {"error": pages["error"], "total_pages": 0, "extracted_text": []}
    
    # Process pages asynchronously with controlled concurrency
    vllm_base_url = f"http://localhost:{VLLM_PORT}/v1"
    api_key = os.environ.get("API_KEY")
    
    # Use async processing with concurrency limits
    extracted_texts = await process_pages_async(
        pages=pages,
        vllm_base_url=vllm_base_url,
        api_key=api_key,
        model=MODEL_PATH
    )
    
    result = {
        "filename": pdf.filename,
        "total_pages": len(pages),
        "dpi": dpi,
        "extracted_text": extracted_texts,
        "success": True
    }
    
    return result
        


if __name__ == "__main__":
    app.serve()