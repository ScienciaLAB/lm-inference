"""
Batch processing script for PaddleOCR layout detection.
Processes multiple PDF documents in a directory using the DocumentProcessor class.
"""

import os
import sys
import argparse
import time
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
from multiprocessing import cpu_count, set_start_method, Manager
import json

# Per-process globals, set once by _init_worker
_worker_processor = None
_verbose = False


def _detect_available_gpus():
    """
    Return the list of visible GPU ids on this node.

    Respects CUDA_VISIBLE_DEVICES if set (e.g. by the scheduler); otherwise
    queries nvidia-smi. Returns an empty list when no GPU is found.
    """
    visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    if visible is not None and visible.strip():
        return [d.strip() for d in visible.split(",") if d.strip()]

    try:
        import subprocess

        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=index", "--format=csv,noheader"],
            capture_output=True,
            text=True,
            check=True,
        )
        return [line.strip() for line in result.stdout.splitlines() if line.strip()]
    except Exception:
        return []


def _init_worker(model_name, dpi, temp_dir, gpu_queue=None, verbose=False):
    """Initializer run once per worker process — loads the model."""
    global _worker_processor, _verbose
    _verbose = verbose

    # Pin this worker to a single GPU (one worker per GPU). Must happen BEFORE the
    # model is built, because PaddlePaddle reads CUDA_VISIBLE_DEVICES at import time
    # (paddle is imported lazily inside PaddleDocumentProcessor._load_model).
    if gpu_queue is not None:
        try:
            gpu_id = gpu_queue.get_nowait()
            os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
            print(f"[worker-{os.getpid()}] pinned to GPU {gpu_id}")
        except Exception:
            print(f"[worker-{os.getpid()}] no GPU assignment, using default device")

    from .paddle_inference import PaddleDocumentProcessor

    t0 = time.time()
    print(f"[worker-{os.getpid()}] Loading model {model_name}...")
    _worker_processor = PaddleDocumentProcessor(
        model_name=model_name,
        dpi=dpi,
        temp_dir=temp_dir,
        preload_model=True,
    )
    print(f"[worker-{os.getpid()}] Model loaded in {time.time() - t0:.1f}s")


def process_single_pdf_worker(args):
    """
    Worker function for processing a single PDF.
    Reuses the processor loaded by _init_worker.
    """
    (pdf_path, output_dir, only, cleanup_images) = args

    from .paddle_inference import (
        filter_and_aggregate,
        load_transform_elements,
    )

    pdf_name = Path(pdf_path).stem
    start_time = time.time()

    try:
        if _verbose:
            print(f"[{pdf_name}] Processing...")
        result = _worker_processor.process_document(pdf_path, output_dir)

        # Apply filtering if requested
        if result.get("success", False):
            bounding_boxes = load_transform_elements(result["output_dir"])

            if only:
                figure_type_aggregation = {
                    "figure": [
                        "figure",
                        "image",
                        "chart",
                        "figure_text",
                        "chart_text",
                    ],
                    "table": ["table", "table_text"],
                    "equation": ["equation", "formula", "equation_text"],
                }

                paratext_type_aggregation = {
                    "headnote": ["header"],
                    "footer": ["footer"],
                }

                if only == "display":
                    bounding_boxes = filter_and_aggregate(
                        bounding_boxes, figure_type_aggregation
                    )
                elif only == "paratext":
                    bounding_boxes = filter_and_aggregate(
                        bounding_boxes, paratext_type_aggregation
                    )
                elif only == "grobid":
                    grobid_type_aggregation = {
                        **figure_type_aggregation,
                        **paratext_type_aggregation,
                    }
                    bounding_boxes = filter_and_aggregate(
                        bounding_boxes, grobid_type_aggregation
                    )

            # Save the final aggregated JSON named after the PDF
            final_json_path = os.path.join(
                result["main_output_dir"], f"{pdf_name}.json"
            )
            with open(final_json_path, "w", encoding="utf-8") as f:
                json.dump(bounding_boxes, f, indent=4, ensure_ascii=False)

        # Apply cleanup if requested
        if result.get("success", False) and cleanup_images:
            doc_output_dir = Path(result["output_dir"])
            main_output_dir = Path(result["main_output_dir"])
            _worker_processor.cleanup_temp_files(
                doc_output_dir,
                main_output_dir,
                cleanup_images=True,
                cleanup_rename=False,
                pdf_name=pdf_name,
            )

        processing_time = time.time() - start_time

        if result.get("success", False):
            return pdf_path, True, None, processing_time
        else:
            error_msg = result.get("error", "Unknown error")
            return pdf_path, False, error_msg, processing_time

    except Exception as e:
        processing_time = time.time() - start_time
        error_msg = f"Exception: {str(e)}"
        return pdf_path, False, error_msg, processing_time


class BatchProcessor:
    def __init__(
        self,
        model_name="PP-DocLayout-S",
        dpi=72,
        temp_dir=None,
        workers=None,
        only=None,
        cleanup_images=False,
        force=False,
        verbose=False,
    ):
        """
        Initialize the batch processor.

        Args:
            model_name (str): Model name for layout detection
            dpi (int): DPI for PDF to image conversion
            temp_dir (str): Custom temporary directory path
            workers (int): Number of parallel workers (default: CPU count)
            only (str): Filter type - "display", "paratext", or "grobid"
            cleanup_images (bool): Whether to clean up intermediate image files
            verbose (bool): Print which PDF each worker starts processing
        """
        self.model_name = model_name
        self.dpi = dpi
        self.temp_dir = temp_dir
        # Default worker count; overridden to one-per-GPU in process_documents when
        # GPUs are present. Capped at 4 to avoid memory issues on CPU-only runs.
        self.workers = min(workers if workers else cpu_count(), 4)
        self.only = only
        self.cleanup_images = cleanup_images
        self.force = force
        self.verbose = verbose
        self.processed_count = 0
        self.skipped_files = []
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

    def update_progress(self, success, pdf_path, error_msg=None):
        """
        Update progress tracking.

        Args:
            success (bool): Whether processing was successful
            pdf_path (str): Path to processed PDF
            error_msg (str): Error message if processing failed
        """
        self.processed_count += 1
        pdf_name = os.path.basename(pdf_path)

        if success:
            self.successful_files.append(pdf_path)
            print(f"({self.processed_count}/{self.total_files}) ✓ {pdf_name}")
        else:
            self.failed_files.append((pdf_path, error_msg))
            print(
                f"({self.processed_count}/{self.total_files}) ✗ {pdf_name} - {error_msg}"
            )

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

            # Create output directory if it doesn't exist
            os.makedirs(output_dir, exist_ok=True)

            # Skip already-processed files unless --force
            if not self.force:
                remaining = []
                for pdf_path in pdf_files:
                    pdf_stem = Path(pdf_path).stem
                    output_json = Path(output_dir) / f"{pdf_stem}.json"
                    if output_json.exists():
                        self.skipped_files.append(pdf_path)
                    else:
                        remaining.append(pdf_path)
                if self.skipped_files:
                    print(
                        f"Skipping {len(self.skipped_files)} already-processed "
                        f"files (use --force to reprocess)"
                    )
                pdf_files = remaining

            if not pdf_files:
                print("Nothing to process.")
                return True

            self.total_files = len(pdf_files)

            # Run one worker (one model copy) per GPU: a single GPU scales down to
            # 1 worker, multiple GPUs spread one model each so they are used equally.
            gpu_ids = _detect_available_gpus()
            n_gpus = len(gpu_ids)
            gpu_queue = None
            manager = None
            if n_gpus >= 1:
                self.workers = n_gpus
            if n_gpus > 1:
                # Hand each worker a distinct GPU id via a picklable queue.
                manager = Manager()
                gpu_queue = manager.Queue()
                for gpu_id in gpu_ids:
                    gpu_queue.put(gpu_id)
                print(
                    f"Detected {n_gpus} GPUs {gpu_ids}: "
                    f"running 1 worker per GPU ({self.workers} workers)"
                )
            elif n_gpus == 1:
                print(f"Detected 1 GPU {gpu_ids}: running 1 worker")

            print(f"Starting batch processing with {self.workers} workers...")
            print(f"Model: {self.model_name}, DPI: {self.dpi}")

            # Prepare task arguments (model config is handled by the initializer)
            tasks = [
                (pdf_path, output_dir, self.only, self.cleanup_images)
                for pdf_path in pdf_files
            ]

            start_time = time.time()

            # Process files in parallel using ProcessPoolExecutor
            # Model is loaded once per worker via _init_worker
            with ProcessPoolExecutor(
                max_workers=self.workers,
                initializer=_init_worker,
                initargs=(
                    self.model_name,
                    self.dpi,
                    self.temp_dir,
                    gpu_queue,
                    self.verbose,
                ),
            ) as executor:
                # Submit all tasks
                future_to_pdf = {
                    executor.submit(process_single_pdf_worker, task): task[0]
                    for task in tasks
                }

                # Process completed tasks
                for future in as_completed(future_to_pdf):
                    pdf_path = future_to_pdf[future]

                    try:
                        pdf_path, success, error_msg, processing_time = future.result()
                        self.update_progress(success, pdf_path, error_msg)
                    except Exception as e:
                        self.update_progress(
                            False, pdf_path, f"Unexpected error: {str(e)}"
                        )

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
        description="Process PDF documents with PaddleOCR layout detection",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python paddle_inference_batch.py ./input_pdfs --output ./results
  python paddle_inference_batch.py ./input_pdfs -o ./results --workers 4
  python paddle_inference_batch.py ./input_pdfs -o ./results --model-name PP-DocLayout-L --dpi 150
  python paddle_inference_batch.py ./input_pdfs -o ./results --only display --cleanup-images
        """,
    )

    parser.add_argument(
        "input_dir", help="Input directory containing PDF files to process"
    )

    parser.add_argument(
        "--output",
        "-o",
        default="output",
        help="Output directory for processed images and results (default: output)",
    )

    parser.add_argument(
        "--model-name",
        choices=[
            "PP-DocLayout-L",
            "PP-DocLayout-M",
            "PP-DocLayout-S",
            "PP-DocLayoutV2",
            "PP-DocBlockLayout",
        ],
        default="PP-DocLayout-S",
        help="Model name for layout detection (default: PP-DocLayout-S)",
    )

    parser.add_argument(
        "--dpi",
        type=int,
        default=72,
        help="DPI for PDF to image conversion (default: 72)",
    )

    parser.add_argument(
        "--workers",
        "-w",
        type=int,
        default=None,
        help="Number of parallel workers (default: CPU count, max: 4)",
    )

    parser.add_argument(
        "--temp-dir",
        help="Temporary directory for processing (default: auto-generated)",
    )

    parser.add_argument(
        "--only",
        choices=["display", "paratext", "grobid"],
        help='Parse JSON output to extract only "display" elements: '
        'tables, figures, and equations, "paratext": sugar coat such '
        'headers and footers, "grobid": both display and paratext, '
        "and create aggregated results",
    )

    parser.add_argument(
        "--cleanup-images",
        action="store_true",
        help="Clean up intermediate image files and keep only the aggregated JSON file "
        "named after the input PDF (e.g., document.pdf → document.json)",
    )

    parser.add_argument(
        "--force",
        action="store_true",
        help="Reprocess all files even if output already exists",
    )

    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Print which PDF each worker starts processing (in addition to the "
        "per-file completion lines)",
    )

    args = parser.parse_args()

    # Create and run batch processor
    processor = BatchProcessor(
        model_name=args.model_name,
        dpi=args.dpi,
        temp_dir=args.temp_dir,
        workers=args.workers,
        only=args.only,
        cleanup_images=args.cleanup_images,
        force=args.force,
        verbose=args.verbose,
    )

    success = processor.process_documents(args.input_dir, args.output)

    # Exit with appropriate code
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    # Set multiprocessing start method to 'spawn' for PaddlePaddle compatibility
    try:
        set_start_method("spawn", force=True)
    except RuntimeError:
        pass

    main()
