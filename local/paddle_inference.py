import argparse
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from paddleocr import LayoutDetection
from pdf2image import convert_from_path
from PIL import Image


# from paddleocr import LayoutDetection
#
# model = LayoutDetection(model_name="PP-DocLayoutV2")
# output = model.predict("https://paddle-model-ecology.bj.bcebos.com/paddlex/imgs/demo_image/layout.jpg", batch_size=1,
#                        layout_nms=True)


def convert_pdf_page_to_image(page_data, page_num, output_dir, dpi):
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
    parser.add_argument('--headers', action='store_true', help='Extract top 5% and bottom 5% of each page as separate images')

    args = parser.parse_args()

    # Validate input file
    if not os.path.exists(args.input):
        print(f"Error: Input file '{args.input}' does not exist")
        sys.exit(1)

    if not args.input.lower().endswith('.pdf'):
        print(f"Error: Input file must be a PDF document")
        sys.exit(1)

    # Validate num_workers
    if args.num_workers < 1:
        print("Error: num-workers must be at least 1")
        sys.exit(1)

    # Create output directory if it doesn't exist
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Processing PDF: {args.input}")
    print(f"Output directory: {output_dir}")
    print(f"DPI: {args.dpi}")
    print(f"Number of workers: {args.num_workers}")
    if args.headers:
        print("Headers mode enabled: extracting top and bottom 5% of each page")

    print("Loading model...")
    model_start_time = time.time()
    model = LayoutDetection(model_name="PP-DocLayoutV2")
    model_end_time = time.time()
    model_load_time = model_end_time - model_start_time
    print(f"Model loaded in {model_load_time:.2f} seconds... starting predictions.")

    # Start overall timing
    overall_start_time = time.time()

    try:
        # Convert PDF to images page by page
        print("Converting PDF to images...")
        images = convert_from_path(args.input, dpi=args.dpi)
        total_pages = len(images)
        print(f"Converted {total_pages} pages to images")

        # Parallel PDF to image conversion
        if args.headers:
            print(f"Extracting headers from pages with {args.num_workers} workers...")
        else:
            print(f"Converting pages to images with {args.num_workers} workers...")

        image_conversion_futures = []

        with ThreadPoolExecutor(max_workers=args.num_workers) as executor:
            for page_num, image in enumerate(images, 1):
                if args.headers:
                    future = executor.submit(split_image_headers, image, page_num, output_dir, args.dpi)
                else:
                    future = executor.submit(convert_pdf_page_to_image, image, page_num, output_dir, args.dpi)
                image_conversion_futures.append(future)

            # Collect results from image conversion
            converted_images = []
            for future in as_completed(image_conversion_futures):
                page_num, image_data, error = future.result()
                if error:
                    print(f"Error converting page {page_num}: {error}")
                else:
                    if args.headers:
                        # image_data is a list of [top_path, bottom_path]
                        for img_path in image_data:
                            print(f"Saved image: {img_path}")
                            converted_images.append((page_num, img_path, None))
                    else:
                        # image_data is a single image_path
                        print(f"Saved image: {image_data}")
                        converted_images.append((page_num, image_data, None))

        # Sort by page number to maintain order
        converted_images.sort(key=lambda x: x[0])

        # Parallel image recognition processing
        print(f"Processing layout detection with {args.num_workers} workers...")
        recognition_futures = []

        # Convert to list of image paths
        image_paths = [str(image_info[1]) for image_info in converted_images]

        print(f"Starting model prediction on {len(image_paths)} images...")
        if args.headers:
            print("Processing header images (top and bottom 5% of each page)")

        start_time = time.time()
        output = model.predict(
            image_paths,
            batch_size=args.num_workers/2,
            layout_nms=True
        )
        end_time = time.time()
        prediction_time = end_time - start_time
        print(f"Model prediction completed in {prediction_time:.2f} seconds")

        for i, res in enumerate(output):
            page_start_time = time.time()
            if args.headers:
                # For headers mode, use a different naming scheme
                base_name = f"header_{i+1:04d}"
            else:
                # Normal mode
                base_name = f"res_{i}"

            print(f"Processing result {i+1}...")
            res.save_to_img(save_path=str(output_dir / f"{base_name}.jpg"))
            res.save_to_json(save_path=str(output_dir / f"{base_name}.json"))
            page_end_time = time.time()
            page_time = page_end_time - page_start_time
            print(f"Result {i+1} processed in {page_time:.2f} seconds")

        overall_end_time = time.time()
        total_processing_time = overall_end_time - overall_start_time
        avg_time_per_image = total_processing_time / len(image_paths)
        print(f"Total processing time: {total_processing_time:.2f} seconds")
        print(f"Average time per image: {avg_time_per_image:.2f} seconds")

    except Exception as e:
        print(f"Error during processing: {str(e)}")
        sys.exit(1)


if __name__ == "__main__":
    main()
