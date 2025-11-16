#!/usr/bin/env python3
"""
Batch processing script for PaddleOCR layout detection.
Processes multiple PDF documents in a directory using the DocumentProcessor class.
"""

import os
import sys
import argparse
import time
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from multiprocessing import cpu_count
import threading

# Import the DocumentProcessor class and filtering functions from the existing module
from paddle_inference import (
    DocumentProcessor,
    filter_and_aggregate,
    load_transform_elements
)


class BatchProcessor:
    def __init__(self, model_name="PP-DocLayout-S", dpi=70, temp_dir=None, workers=None, only=None, cleanup_images=False):
        """
        Initialize the batch processor.

        Args:
            model_name (str): Model name for layout detection
            dpi (int): DPI for PDF to image conversion
            temp_dir (str): Custom temporary directory path
            workers (int): Number of parallel workers (default: CPU count)
            only (str): Filter type - "display", "paratext", or "grobid"
            cleanup_images (bool): Whether to clean up intermediate image files
        """
        self.model_name = model_name
        self.dpi = dpi
        self.temp_dir = temp_dir
        self.workers = workers if workers else cpu_count()
        self.only = only
        self.cleanup_images = cleanup_images
        self.progress_lock = threading.Lock()
        self.processed_count = 0
        self.total_files = 0
        self.failed_files = []
        self.successful_files = []

    def discover_pdf_files(self, input_dir):
        """
        Discover all PDF files in the input directory.

        Args:
            input_dir (str): Path to input directory

        Returns:
            list: List of PDF file paths
        """
        pdf_files = []
        input_path = Path(input_dir)

        if not input_path.exists():
            raise ValueError(f"Input directory does not exist: {input_dir}")

        if not input_path.is_dir():
            raise ValueError(f"Input path is not a directory: {input_dir}")

        # Find all PDF files in the directory
        for pdf_file in input_path.glob("*.pdf"):
            pdf_files.append(str(pdf_file))

        if not pdf_files:
            raise ValueError(f"No PDF files found in directory: {input_dir}")

        print(f"Found {len(pdf_files)} PDF files to process")
        return sorted(pdf_files)

    def process_single_document(self, processor, pdf_path, output_dir):
        """
        Process a single document using the provided processor.

        Args:
            processor (DocumentProcessor): Initialized document processor
            pdf_path (str): Path to PDF file
            output_dir (str): Output directory

        Returns:
            tuple: (pdf_path, success, error_message, processing_time)
        """
        start_time = time.time()
        pdf_name = os.path.basename(pdf_path)

        try:
            # Process the document
            result = processor.process_document(pdf_path, output_dir)

            # Apply filtering if requested
            if result.get('success', False):
                bounding_boxes = load_transform_elements(
                    result['output_dir']
                )

                if self.only:
                    figure_type_aggregation = {
                        "figure": ["figure", "image", "chart", "figure_text", "chart_text"],
                        "table": ["table", "table_text"],
                        "equation": ["equation", "formula", "equation_text"]
                    }

                    paratext_type_aggregation = {
                        "headnote": ["header"],
                        "footer": ["footer"]
                    }

                    if self.only == "display":
                        bounding_boxes = filter_and_aggregate(bounding_boxes, figure_type_aggregation)
                    elif self.only == "paratext":
                        bounding_boxes = filter_and_aggregate(bounding_boxes, paratext_type_aggregation)
                    elif self.only == "grobid":
                        grobid_type_aggregation = {**figure_type_aggregation, **paratext_type_aggregation}
                        bounding_boxes = filter_and_aggregate(bounding_boxes, grobid_type_aggregation)


            # Apply cleanup if requested
            if result.get('success', False) and self.cleanup_images:
                doc_output_dir = Path(result['output_dir'])
                main_output_dir = Path(result['main_output_dir'])
                pdf_name = Path(pdf_path).stem
                processor.cleanup_temp_files(
                    doc_output_dir,
                    main_output_dir,
                    cleanup_images=True,
                    cleanup_rename=True,
                    pdf_name=pdf_name
                )

            processing_time = time.time() - start_time

            if result.get('success', False):
                return pdf_path, True, None, processing_time
            else:
                error_msg = result.get('error', 'Unknown error')
                return pdf_path, False, error_msg, processing_time

        except Exception as e:
            import traceback
            processing_time = time.time() - start_time
            error_msg = f"Exception: {str(e)}\nStacktrace:\n{traceback.format_exc()}"
            return pdf_path, False, error_msg, processing_time

    def update_progress(self, success, pdf_path, error_msg=None):
        """
        Update progress tracking.

        Args:
            success (bool): Whether processing was successful
            pdf_path (str): Path to processed PDF
            error_msg (str): Error message if processing failed
        """
        with self.progress_lock:
            self.processed_count += 1
            pdf_name = os.path.basename(pdf_path)

            if success:
                self.successful_files.append(pdf_path)
                print(f"({self.processed_count}/{self.total_files}) ✓ {pdf_name}")
            else:
                self.failed_files.append((pdf_path, error_msg))
                print(f"({self.processed_count}/{self.total_files}) ✗ {pdf_name} - {error_msg}")

    def process_documents(self, input_dir, output_dir):
        """
        Process all PDF documents in the input directory.

        Args:
            input_dir (str): Input directory containing PDF files
            output_dir (str): Output directory for results

        Returns:
            bool: True if all files processed successfully, False otherwise
        """
        try:
            # Discover PDF files
            pdf_files = self.discover_pdf_files(input_dir)
            self.total_files = len(pdf_files)

            # Create output directory if it doesn't exist
            os.makedirs(output_dir, exist_ok=True)

            print(f"Starting batch processing with {self.workers} workers...")
            print(f"Model: {self.model_name}, DPI: {self.dpi}")
            print("-" * 60)

            # Create processor instances for each worker
            processors = []
            for _ in range(self.workers):
                processor = DocumentProcessor(
                    model_name=self.model_name,
                    dpi=self.dpi,
                    temp_dir=self.temp_dir,
                    preload_model=True
                )
                processors.append(processor)

            start_time = time.time()

            # Process files in parallel
            with ThreadPoolExecutor(max_workers=self.workers) as executor:
                # Submit all tasks
                future_to_processor = {}
                for i, pdf_path in enumerate(pdf_files):
                    processor = processors[i % self.workers]
                    future = executor.submit(
                        self.process_single_document,
                        processor,
                        pdf_path,
                        output_dir
                    )
                    future_to_processor[future] = (pdf_path, processor)

                # Process completed tasks
                for future in as_completed(future_to_processor):
                    pdf_path, processor = future_to_processor[future]

                    try:
                        pdf_path, success, error_msg, processing_time = future.result()
                        self.update_progress(success, pdf_path, error_msg)
                    except Exception as e:
                        self.update_progress(False, pdf_path, f"Unexpected error: {str(e)}")

            total_time = time.time() - start_time

            # Print summary
            self.print_summary(total_time)

            return len(self.failed_files) == 0

        except Exception as e:
            import traceback
            print(f"Batch processing failed: {str(e)}")
            print("Stacktrace:")
            print(traceback.format_exc())
            return False

    def print_summary(self, total_time):
        """
        Print processing summary.

        Args:
            total_time (float): Total processing time in seconds
        """
        print("-" * 60)
        print("BATCH PROCESSING SUMMARY")
        print("-" * 60)
        print(f"Total files: {self.total_files}")
        print(f"Successful: {len(self.successful_files)}")
        print(f"Failed: {len(self.failed_files)}")
        print(f"Total time: {total_time:.2f} seconds")

        if self.total_files > 0:
            avg_time = total_time / self.total_files
            print(f"Average time per document: {avg_time:.2f} seconds")

        if self.failed_files:
            print("\nFailed files:")
            for pdf_path, error_msg in self.failed_files:
                pdf_name = os.path.basename(pdf_path)
                print(f"  • {pdf_name}: {error_msg}")


def main():
    """Main function to run the batch processor."""
    parser = argparse.ArgumentParser(
        description='Process PDF documents with PaddleOCR layout detection',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python paddle_inference_batch.py ./input_pdfs --output ./results
  python paddle_inference_batch.py ./input_pdfs -o ./results --workers 4
  python paddle_inference_batch.py ./input_pdfs -o ./results --model-name PP-DocLayout-L --dpi 150
  python paddle_inference_batch.py ./input_pdfs -o ./results --only display --cleanup-images
        """
    )

    parser.add_argument(
        'input_dir',
        help='Input directory containing PDF files to process'
    )

    parser.add_argument(
        '--output', '-o',
        default='output',
        help='Output directory for processed images and results (default: output)'
    )

    parser.add_argument(
        '--model-name',
        choices=['PP-DocLayout-L', 'PP-DocLayout-M', 'PP-DocLayout-S', 'PP-DocLayoutV2',
                 'PP-DocBlockLayout'],
        default='PP-DocLayout-S',
        help='Model name for layout detection (default: PP-DocLayout-S)'
    )

    parser.add_argument(
        '--dpi',
        type=int,
        default=70,
        help='DPI for PDF to image conversion (default: 70)'
    )

    parser.add_argument(
        '--workers', '-w',
        type=int,
        default=None,
        help=f'Number of parallel workers (default: CPU count, {cpu_count()})'
    )

    parser.add_argument(
        '--temp-dir',
        help='Temporary directory for processing (default: auto-generated)'
    )

    parser.add_argument('--only',
                        choices=["display", "paratext", "grobid"],
                        help='Parse JSON output to extract only "display" elements: '
                             'tables, figures, and equations, "paratext": sugar coat such '
                             'headers and footers, "grobid": both display and paratext, '
                             'and create aggregated results')

    parser.add_argument('--cleanup-images',
                        action='store_true',
                        help='Clean up intermediate image files and keep only the aggregated JSON file '
                             'named after the input PDF (e.g., document.pdf → document.json)')

    args = parser.parse_args()

    # Create and run batch processor
    processor = BatchProcessor(
        model_name=args.model_name,
        dpi=args.dpi,
        temp_dir=args.temp_dir,
        workers=args.workers,
        only=args.only,
        cleanup_images=args.cleanup_images
    )

    success = processor.process_documents(args.input_dir, args.output)

    # Exit with appropriate code
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()