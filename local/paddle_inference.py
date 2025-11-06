import argparse
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

from paddleocr import LayoutDetection
from pdf2image import convert_from_path


# from paddleocr import LayoutDetection
#
# model = LayoutDetection(model_name="PP-DocLayoutV2")
# output = model.predict("https://paddle-model-ecology.bj.bcebos.com/paddlex/imgs/demo_image/layout.jpg", batch_size=1,
#                        layout_nms=True)


def save_image(page_data, page_num, output_dir):
    """Convert a single PDF page to image"""
    try:
        image = page_data
        image_path = output_dir / f"page_{page_num:04d}.jpg"
        image.save(image_path, 'JPEG')
        return page_num, image_path, None
    except Exception as e:
        return page_num, None, str(e)


def split_image_headers(page_data, page_num, output_dir, dpi):
    """Split a single PDF page into top 5% and bottom 5% header images"""
    try:
        image = page_data
        width, height = image.size

        # Calculate 5% of height
        header_height = int(height * 0.05)

        # Create top 5% crop
        top_image = image.crop((0, 0, width, header_height))
        top_path = output_dir / f"page_{page_num:04d}_top.jpg"
        top_image.save(top_path, 'JPEG')

        # Create bottom 5% crop
        bottom_image = image.crop((0, height - header_height, width, height))
        bottom_path = output_dir / f"page_{page_num:04d}_bottom.jpg"
        bottom_image.save(bottom_path, 'JPEG')

        return page_num, [top_path, bottom_path], None
    except Exception as e:
        return page_num, None, str(e)


def main():
    parser = argparse.ArgumentParser(description='Process PDF documents with PaddleOCR layout detection')
    parser.add_argument('--input', '-i', required=True, help='Input PDF document path')
    parser.add_argument('--output', '-o', required=True, help='Output directory for processed images and results')
    parser.add_argument('--dpi', type=int, default=150, help='DPI for PDF to image conversion (default: 150)')
    parser.add_argument('--num-workers', type=int, default=1, help='Number of worker processes for parallel processing (default: 1)')

    args = parser.parse_args()

    # Validate input file
    if not os.path.exists(args.input):
        print(f"Error: Input file '{args.input}' does not exist")
        sys.exit(1)

    if not args.input.lower().endswith('.pdf'):
        print(f"Error: Input file must be a PDF document")
        sys.exit(1)

    # Validate num_workers
    num_workers = os.cpu_count() if args.num_workers < 1 else args.num_workers
    print(f"Number of workers: {num_workers}")

    # Create output directory if it doesn't exist
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Processing PDF: {args.input}")
    print(f"Output directory: {output_dir}")
    print(f"DPI: {args.dpi}")
    print(f"Number of workers: {args.num_workers}")


    print("Loading model...")
    model_start_time = time.time()
    model = LayoutDetection(model_name="PP-DocLayoutV2")
    model_end_time = time.time()
    model_load_time = model_end_time - model_start_time
    print(f"Model loaded in {model_load_time:.2f} seconds... starting predictions.")

    # Start overall timing
    overall_start_time = time.time()

    try:
        converted_images = pdf_to_images(args.input, output_dir, args.dpi)

        print(f"Processing layout detection with {args.num_workers} workers...")
        image_paths = [str(image_info[1]) for image_info in converted_images]

        print(f"Starting model prediction on {len(image_paths)} images...")

        start_time = time.time()
        output = model.predict(
            image_paths,
            batch_size=os.cpu_count(),
            layout_nms=True
        )
        end_time = time.time()
        prediction_time = end_time - start_time
        print(f"Model prediction completed in {prediction_time:.2f} seconds")

        for i, res in enumerate(output):
            base_name = f"res_{i}"

            res.save_to_img(save_path=str(output_dir / f"{base_name}.jpg"))
            res.save_to_json(save_path=str(output_dir / f"{base_name}.json"))

        overall_end_time = time.time()
        total_processing_time = overall_end_time - overall_start_time
        avg_time_per_image = total_processing_time / len(image_paths)
        print(f"Total processing time: {total_processing_time:.2f} seconds")
        print(f"Average time per image: {avg_time_per_image:.2f} seconds")

    except Exception as e:
        print(f"Error during processing: {str(e)}")
        sys.exit(1)


def pdf_to_images(input_pdf: Path, output_dir: Path, dpi: int) -> list[Any]:
    print("Converting PDF to images...")
    images = convert_from_path(input_pdf, dpi=dpi, thread_count=os.cpu_count())
    converted_images = []
    total_pages = len(images)
    for page_num, image in enumerate(images, 1):
        page_num, image_data, error = save_image(image, page_num, output_dir)
        converted_images.append((page_num, image_data, None))
    print(f"Converted {total_pages} pages to images")
    return converted_images


if __name__ == "__main__":
    main()
