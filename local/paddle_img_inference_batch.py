#!/usr/bin/env python3
"""
Batch processing script for PaddleOCR layout detection on IMAGES.
Processes multiple PNG/JPG files in a directory using parallel workers.
"""

import os
import sys
import argparse
import time
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from multiprocessing import cpu_count
import threading

from paddle_inference import PaddleDocumentProcessor


class ImageBatchProcessor:
    def __init__(
        self,
        model_name="PP-DocLayout-S",
        workers=None,
    ):
        """
        Initialize the batch processor for images.
        """
        self.model_name = model_name
        self.workers = workers if workers else cpu_count()
        self.progress_lock = threading.Lock()
        self.processed_count = 0
        self.total_files = 0
        self.failed_files = []
        self.successful_files = []

    def discover_image_files(self, input_dir):
        """
        Discover all PNG and JPG files in the input directory.
        """
        image_files = []
        input_path = Path(input_dir)

        if not input_path.exists():
            raise ValueError(f"Input directory does not exist: {input_dir}")

        # Find PNG and JPG
        extensions = ["*.png", "*.jpg", "*.jpeg", "*.PNG", "*.JPG"]
        for ext in extensions:
            image_files.extend(list(input_path.glob(ext)))

        if not image_files:
            raise ValueError(f"No image files found in directory: {input_dir}")

        # Sort to ensure consistent order
        return sorted([str(f) for f in image_files])

    def process_single_image(self, processor, img_path, output_dir):
        """
        Process a single image using the provided processor.
        """
        start_time = time.time()
        base_name = Path(img_path).stem

        try:
            # 1. Ensure Model is Loaded (Lazy Loading in Thread)
            if processor.model is None:
                processor._load_model()

            # 2. Run Prediction directly on image path
            # layout_nms=True cleans up overlapping boxes
            preds = processor.model.predict([img_path], layout_nms=True)

            result = preds[0]

            # Create a folder for this image
            img_output_dir = os.path.join(output_dir, base_name)
            os.makedirs(img_output_dir, exist_ok=True)

            # Save Visual (Image with boxes) inside image folder
            result.save_to_img(
                save_path=os.path.join(img_output_dir, f"{base_name}.jpg")
            )

            # Save JSON (Coordinates) inside image folder
            result.save_to_json(
                save_path=os.path.join(img_output_dir, f"{base_name}.json")
            )

            processing_time = time.time() - start_time
            return img_path, True, None, processing_time

        except Exception as e:
            processing_time = time.time() - start_time
            error_msg = f"Exception: {str(e)}"
            return img_path, False, error_msg, processing_time

    def update_progress(self, success, img_path, error_msg=None):
        """
        Update progress tracking.
        """
        with self.progress_lock:
            self.processed_count += 1
            img_name = os.path.basename(img_path)

            if success:
                self.successful_files.append(img_path)
                if self.processed_count % 5 == 0:
                    print(f"({self.processed_count}/{self.total_files}) ✓ {img_name}")
            else:
                self.failed_files.append((img_path, error_msg))
                print(
                    f"({self.processed_count}/{self.total_files}) ✗ {img_name} - {error_msg}"
                )

    def process_images(self, input_dir, output_dir):
        """
        Process all Images in the input directory.
        """
        try:
            # Discover files
            img_files = self.discover_image_files(input_dir)
            self.total_files = len(img_files)

            # Create output directory
            os.makedirs(output_dir, exist_ok=True)

            print(f"Starting batch processing with {self.workers} workers...")
            print(f"Model: {self.model_name}")
            print(f"Target: {self.total_files} Images")
            print("-" * 60)

            # Create processor instances for each worker
            processors = []
            for _ in range(self.workers):
                processor = PaddleDocumentProcessor(
                    model_name=self.model_name,
                    dpi=72,  # Irrelevant for images, but required by init
                    preload_model=True,
                )
                processors.append(processor)

            start_time = time.time()

            # Process files in parallel
            with ThreadPoolExecutor(max_workers=self.workers) as executor:
                future_to_processor = {}

                for i, img_path in enumerate(img_files):
                    processor = processors[i % self.workers]

                    future = executor.submit(
                        self.process_single_image, processor, img_path, output_dir
                    )
                    future_to_processor[future] = (img_path, processor)

                for future in as_completed(future_to_processor):
                    img_path, processor = future_to_processor[future]
                    try:
                        img_path, success, error_msg, processing_time = future.result()
                        self.update_progress(success, img_path, error_msg)
                    except Exception as e:
                        self.update_progress(
                            False, img_path, f"Unexpected error: {str(e)}"
                        )

            total_time = time.time() - start_time
            self.print_summary(total_time)
            return len(self.failed_files) == 0

        except Exception as e:
            import traceback

            print(f"Batch processing failed: {str(e)}")
            print(traceback.format_exc())
            return False

    def print_summary(self, total_time):
        print("-" * 60)
        print("IMAGE BATCH PROCESSING SUMMARY")
        print("-" * 60)
        print(f"Total files: {self.total_files}")
        print(f"Successful: {len(self.successful_files)}")
        print(f"Failed: {len(self.failed_files)}")
        print(f"Total time: {total_time:.2f} seconds")

        if self.total_files > 0:
            avg_time = total_time / self.total_files
            pps = self.total_files / total_time
            print(f"Average time per image: {avg_time:.2f} seconds")
            print(f"Throughput: {pps:.2f} Images/Second")

        if self.failed_files:
            print("\nFailed files:")
            for img_path, error_msg in self.failed_files:
                print(f"  • {os.path.basename(img_path)}: {error_msg}")


def main():
    parser = argparse.ArgumentParser(
        description="Process Images with PaddleOCR layout detection"
    )

    parser.add_argument("input_dir", help="Input directory containing PNG/JPG files")
    parser.add_argument("--output", "-o", default="output", help="Output directory")

    parser.add_argument(
        "--model-name",
        choices=["PP-DocLayout-L", "PP-DocLayout-M", "PP-DocLayout-S"],
        default="PP-DocLayout-S",
        help="Model name",
    )

    parser.add_argument(
        "--workers",
        "-w",
        type=int,
        default=None,
        help=f"Number of parallel workers (default: {cpu_count()})",
    )

    args = parser.parse_args()

    processor = ImageBatchProcessor(
        model_name=args.model_name,
        workers=args.workers,
    )

    success = processor.process_images(args.input_dir, args.output)
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
