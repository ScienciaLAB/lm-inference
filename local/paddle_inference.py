import argparse
import json
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, List

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


def filter_layout_elements(json_file_path: str) -> Dict[str, List[Dict]]:
    """
    Filter layout detection results to extract only tables, figures, and equations.

    Args:
        json_file_path: Path to the JSON file containing layout detection results

    Returns:
        Dictionary containing filtered elements by type
    """
    try:
        with open(json_file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        # Categorize boxes by type
        filtered_elements = {'tables': [], 'figures': [], 'equations': []}

        for box in data.get('boxes', []):
            label = box.get('label', '').lower()
            if 'table' in label:
                filtered_elements['tables'].append(box)
            elif any(x in label for x in ['figure', 'image', 'chart']):
                filtered_elements['figures'].append(box)
            elif any(x in label for x in ['equation', 'formula']):
                filtered_elements['equations'].append(box)

        return filtered_elements

    except:
        return {'tables': [], 'figures': [], 'equations': []}


def aggregate_filtered_elements(output_dir: Path, all_results: List[Dict]) -> None:
    """
    Aggregate all filtered elements across all pages and save to a combined JSON file.

    Args:
        output_dir: Output directory path
        all_results: List of filtered results from all JSON files
    """
    aggregated_data = {
        'metadata': {
            'total_pages': len(all_results),
            'processed_files': []
        },
        'tables': [],
        'figures': [],
        'equations': [],
        'summary': {
            'total_tables': 0,
            'total_figures': 0,
            'total_equations': 0
        }
    }

    # Aggregate all results
    for result in all_results:
        if result:
            aggregated_data['tables'].extend(result.get('tables', []))
            aggregated_data['figures'].extend(result.get('figures', []))
            aggregated_data['equations'].extend(result.get('equations', []))

            # Add file info to metadata
            if result.get('tables') or result.get('figures') or result.get('equations'):
                if result.get('tables'):
                    file_info = result['tables'][0].get('file', 'unknown')
                elif result.get('figures'):
                    file_info = result['figures'][0].get('file', 'unknown')
                elif result.get('equations'):
                    file_info = result['equations'][0].get('file', 'unknown')
                else:
                    file_info = 'unknown'

                if file_info not in aggregated_data['metadata']['processed_files']:
                    aggregated_data['metadata']['processed_files'].append(file_info)

    # Update summary
    aggregated_data['summary']['total_tables'] = len(aggregated_data['tables'])
    aggregated_data['summary']['total_figures'] = len(aggregated_data['figures'])
    aggregated_data['summary']['total_equations'] = len(aggregated_data['equations'])

    # Save aggregated results
    aggregated_file = output_dir / 'aggregated_elements.json'
    try:
        with open(aggregated_file, 'w', encoding='utf-8') as f:
            json.dump(aggregated_data, f, indent=2, ensure_ascii=False)

        print(f"\n=== GROBID Analysis Summary ===")
        print(f"Total tables found: {aggregated_data['summary']['total_tables']}")
        print(f"Total figures found: {aggregated_data['summary']['total_figures']}")
        print(f"Total equations found: {aggregated_data['summary']['total_equations']}")
        print(f"Aggregated results saved to: {aggregated_file}")
        print("=" * 30)

    except Exception as e:
        print(f"Error saving aggregated results: {str(e)}")


def process_grobid_output(output_dir: Path) -> None:
    """
    Process all JSON files in the output directory to extract tables, figures, and equations.

    Args:
        output_dir: Directory containing the JSON output files
    """

    # Find all JSON result files
    json_files = list(output_dir.glob('res_*.json'))

    if not json_files:
        print("No JSON result files found for GROBID processing.")
        return

    all_filtered_results = []

    # Process each JSON file
    for json_file in sorted(json_files):
        filtered_result = filter_layout_elements(str(json_file))
        all_filtered_results.append(filtered_result)

        # Save filtered results for this file
        if any(filtered_result.values()):
            output_file = output_dir / f"filtered_{json_file.name}"
            try:
                with open(output_file, 'w', encoding='utf-8') as f:
                    json.dump(filtered_result, f, indent=2, ensure_ascii=False)
            except Exception as e:
                print(f"Error saving filtered results for {json_file.name}: {str(e)}")

    # Create aggregated output
    aggregate_filtered_elements(output_dir, all_filtered_results)


def main():
    parser = argparse.ArgumentParser(description='Process PDF documents with PaddleOCR layout detection')
    parser.add_argument('--input', '-i', required=True, help='Input PDF document path')
    parser.add_argument('--output', '-o', required=True, help='Output directory for processed images and results')
    parser.add_argument('--dpi', type=int, default=70, help='DPI for PDF to image conversion (default: 150)')
    parser.add_argument('--num-workers', type=int, default=1,
                        help='Number of worker processes for parallel processing (default: 1)')

    parser.add_argument(
        '--model-name',
        choices=['PP-DocLayout-L', 'PP-DocLayout-M', 'PP-DocLayout-S', 'PP-DocLayoutV2', 'PP-DocBlockLayout'],
        default='PP-DocLayout-S',
        help='Model name for layout detection (default: PP-DocLayout-S)')
    parser.add_argument('--grobid', action='store_true',
                        help='Parse JSON output to extract only tables, figures, and equations and create aggregated results')

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
    print(f"Model: {args.model_name}")
    if args.grobid:
        print("GROBID processing: ENABLED")

    print("Loading model...")
    model_start_time = time.time()
    model = LayoutDetection(model_name=args.model_name)
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
            res.page_index = i + 1

            res.save_to_img(save_path=str(output_dir / f"{base_name}.jpg"))
            res.save_to_json(save_path=str(output_dir / f"{base_name}.json"))

        overall_end_time = time.time()
        total_processing_time = overall_end_time - overall_start_time
        avg_time_per_image = total_processing_time / len(image_paths)
        print(f"Total processing time: {total_processing_time:.2f} seconds")
        print(f"Average time per image: {avg_time_per_image:.2f} seconds")

        # Process GROBID output if requested
        if args.grobid:
            process_grobid_output(output_dir)

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
