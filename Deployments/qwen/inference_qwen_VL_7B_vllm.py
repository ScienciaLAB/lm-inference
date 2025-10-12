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

# Create Modal image with vLLM dependencies
image = (
    modal.Image.from_registry(f"nvidia/cuda:{tag}", add_python="3.11")
    .pip_install(
        "vllm>=0.4.0",
        "huggingface_hub[hf_transfer]==0.26.2",
        "flashinfer-python==0.2.0.post2",
        "PyMuPDF",
        "fastapi[standard]==0.115.4",
        "openai",
        "requests==2.32.3",
        "pydantic==2.9.2",
        "transformers==4.54.1",
        "torch==2.7.1",
        "numpy<2",
        extra_index_url="https://flashinfer.ai/whl/cu124/torch2.5",
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
    
    Args:
        pdf_content: PDF file content as bytes
        dpi: Resolution for image extraction (default: 150)
    
    Returns:
        List of dictionaries containing page data
    """
    try:
        # Load PDF from bytes
        pdf_doc = fitz.open(stream=pdf_content, filetype="pdf")
        total_pages = len(pdf_doc)
        
        print(f"📄 Processing PDF with {total_pages} pages at {dpi} DPI")
        
        # Extract all pages as images
        page_images = []
        for page_num in range(total_pages):
            page = pdf_doc.load_page(page_num)
            mat = fitz.Matrix(dpi/72, dpi/72)  # Scale factor for DPI
            pix = page.get_pixmap(matrix=mat)
            img_data = pix.tobytes("png")
            
            page_data = {
                "page_number": page_num + 1,  # 1-indexed
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
    container_idle_timeout=20 * MINUTES,
    image=image,
    volumes=volumes,
)
@modal.concurrent(max_inputs=100)
class VLMMModel:
    def __init__(self):
        self.client = None
    
    @modal.enter()
    def start_vllm_client(self):
        """Setup OpenAI client for vLLM communication"""
        # Setup OpenAI client for vLLM communication
        self.client = AsyncOpenAI(
            base_url=f"http://localhost:{VLLM_PORT}/v1",
            api_key=os.environ.get("API_KEY", None)
        )
        print("🚀 vLLM client initialized and ready")

    @modal.web_endpoint(method="POST", docs=True)
    async def generate(self, question: str, image: Union[Path, str, BinaryIO]):
        """Image QA functionality using vLLM"""
        from pathlib import Path

        start = time.monotonic_ns()
        request_id = uuid4()
        print(f"Generating response to request {request_id}")

        # Handle different image input types
        if isinstance(image, (str, Path)):
            image_path = Path(image)
            with open(image_path, 'rb') as f:
                image_data = f.read()
        else:
            # For BinaryIO, read directly
            image_data = image.read()

        # Convert image to base64
        image_base64 = base64.b64encode(image_data).decode('utf-8')

        # Create message for vision-language model
        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": question
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/png;base64,{image_base64}"
                        }
                    }
                ]
            }
        ]
        
        try:
            # Call vLLM using OpenAI client
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

    @modal.web_endpoint(method="POST", docs=True)
    async def extract_document_structure(self, pages: List[Dict]) -> Dict:
        """
        Extract document structure from PDF pages using vLLM
        
        Args:
            pages: List of page data from PDF extraction
            
        Returns:
            Dictionary containing complete document structure
        """
        print(f"🔍 Extracting structure from {len(pages)} pages using vLLM")
        
        # Process all pages concurrently
        tasks = []
        for page_data in pages:
            task = self._extract_page_structure(
                page_data["image_base64"], 
                page_data["page_number"]
            )
            tasks.append(task)
        
        # Wait for all pages to be processed
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Process results and create final document structure
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
        
        # Process each page result
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                # Handle exceptions
                page_result = {
                    "page_number": pages[i]["page_number"],
                    "success": False,
                    "error": f"Processing failed: {str(result)}",
                    "header": {"passages": []},
                    "body": {"passages": []},
                    "annex": {"passages": []}
                }
            else:
                page_result = result
                if result.get("success", False):
                    # Count passages and references
                    data = result["data"]
                    for section in ["header", "body", "annex"]:
                        if section in data:
                            passages = data[section].get("passages", [])
                            document_structure["summary"]["structure_counts"][f"{section}_passages"] += len(passages)
                            document_structure["summary"]["total_passages"] += len(passages)
                            
                            # Count references in each passage
                            for passage in passages:
                                references = passage.get("references", [])
                                document_structure["summary"]["total_references"] += len(references)
                else:
                    # Failed page - provide empty structure
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
    
    Args:
            image_base64: Base64 encoded image data
            page_number: Page number for reference
    
    Returns:
            Dictionary containing extracted text structure with passages
        """
        structure_prompt = """
        Analyze this document page and extract all text content, organizing it into three main structural parts:

        1. **Header**: Title, author, date, or any header information at the top of the page
        2. **Body**: Main text content, paragraphs, sections, and core document content
        3. **Annex**: References, footnotes, appendices, or supplementary information at the bottom

        For each structural part, extract all text passages (paragraphs) with the following information:
        - Full text content of each passage
        - References within the text (with character offsets indicating where they appear)
        - Approximate coordinates of the passage on the page

        Return the result as a JSON object with this exact structure:
        {
            "page_number": <page_number>,
            "header": {
                "passages": [
                    {
                        "text": "Full text content of the passage",
                        "references": [
                            {
                                "text": "reference text",
                                "start_offset": <character_position_start>,
                                "end_offset": <character_position_end>,
                                "type": "citation|footnote|url|other"
                            }
                        ],
                        "coordinates": {
                            "x": <approximate_x_position>,
                            "y": <approximate_y_position>,
                            "width": <approximate_width>,
                            "height": <approximate_height>
                        }
                    }
                ]
            },
            "body": {
                "passages": [
                    {
                        "text": "Full text content of the passage",
                        "references": [
                            {
                                "text": "reference text",
                                "start_offset": <character_position_start>,
                                "end_offset": <character_position_end>,
                                "type": "citation|footnote|url|other"
                            }
                        ],
                        "coordinates": {
                            "x": <approximate_x_position>,
                            "y": <approximate_y_position>,
                            "width": <approximate_width>,
                            "height": <approximate_height>
                        }
                    }
                ]
            },
            "annex": {
                "passages": [
                    {
                        "text": "Full text content of the passage",
                        "references": [
                            {
                                "text": "reference text",
                                "start_offset": <character_position_start>,
                                "end_offset": <character_position_end>,
                                "type": "citation|footnote|url|other"
                            }
                        ],
                        "coordinates": {
                            "x": <approximate_x_position>,
                            "y": <approximate_y_position>,
                            "width": <approximate_width>,
                            "height": <approximate_height>
                        }
                    }
                ]
            }
        }

        Important guidelines:
        - Extract ALL text content from the page
        - Identify references like citations [1], (Smith et al., 2023), URLs, footnotes, etc.
        - Provide character offsets for references within the text
        - Estimate coordinates based on visual position on the page
        - If a section is empty, use an empty passages array
        - Be thorough in text extraction - don't miss any content
        """
        
        try:
            messages = [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": structure_prompt
                        },
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/png;base64,{image_base64}"
                            }
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
            
            # Try to parse the JSON response
            content = response.choices[0].message.content
            
            # Extract JSON from the response (in case there's extra text)
            try:
                # Look for JSON in the response
                start_idx = content.find('{')
                end_idx = content.rfind('}') + 1
                if start_idx != -1 and end_idx != 0:
                    json_str = content[start_idx:end_idx]
                    parsed_result = json.loads(json_str)
                    parsed_result["page_number"] = page_number
                    
                    # Validate the structure has the required sections
                    required_sections = ["header", "body", "annex"]
                    for section in required_sections:
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
                # If JSON parsing fails, return the raw response with empty structure
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
    memory=16384,  # 16GB memory for 7B model
    cpu=4,
    timeout=600,
    volumes={
        "/root/.cache/huggingface": hf_cache_vol,
    },
    secrets=[modal.Secret.from_name("document-qa-api-key")]
)
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


