"""
Batch processing script for LADaS (YOLO) layout detection.
Processes multiple PNG/JPG files in a directory using parallel workers.
"""
import os
import sys
import argparse
import time
import threading
import traceback
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from multiprocessing import cpu_count

from ultralytics import YOLO

from base_inference import BaseDocumentProcessor


class LADaSDocumentProcessor(BaseDocumentProcessor):
    def __init__(
        self,
        model_name: str = None,
        dpi: int = 70,
        temp_dir: str = None,
        preload_model: bool = False,
    ):
        super().__init__(model_name, dpi, temp_dir, preload_model)

    def _load_model(self) -> None:
        if self.model is None:
            self.model = YOLO(str(self.model_name), verbose=False) 


class LADaSBatchRunner:
    def __init__(
        self,
        model_file: str,
        workers: int = None,
    ):
        """
        Initialize the batch processor.
        """
        self.model_file = model_file
        self.workers = workers if workers else max(1, cpu_count() - 1)
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

        extensions = ['*.png', '*.jpg', '*.jpeg', '*.PNG', '*.JPG']
        for ext in extensions:
            image_files.extend(list(input_path.glob(ext)))

        if not image_files:
            raise ValueError(f"No image files found in directory: {input_dir}")

        return sorted([str(f) for f in image_files])

    def process_single_image(self, processor, img_path, output_dir):
        """
        Process a single image using the assigned processor.
        """
        start_time = time.time()
        base_name = Path(img_path).stem

        try:
            if processor.model is None:
                processor._load_model()

            # We run on a single image. stream=False ensures we get the result object immediately.
            results = processor.model.predict(img_path, verbose=False, stream=False)
            save_path = os.path.join(output_dir, f"{base_name}.json")
            for res in results:
                with open(save_path, "w", encoding="utf-8") as f:
                    f.write(res.to_json())
            processing_time = time.time() - start_time
            return img_path, True, None, processing_time

        except Exception as e:
            processing_time = time.time() - start_time
            error_msg = f"{type(e).__name__}: {str(e)}"
            return img_path, False, error_msg, processing_time

    def update_progress(self, success, img_path, error_msg=None):
        with self.progress_lock:
            self.processed_count += 1
            img_name = os.path.basename(img_path)

            if success:
                self.successful_files.append(img_path)
                # Print every 5 files to reduce console spam
                if self.processed_count % 5 == 0 or self.processed_count == self.total_files:
                    print(f"[{self.processed_count}/{self.total_files}] ✓ {img_name}")
            else:
                self.failed_files.append((img_path, error_msg))
                print(f"[{self.processed_count}/{self.total_files}] ✗ {img_name} - {error_msg}")

    def process_images(self, input_dir, output_dir):
        try:
            img_files = self.discover_image_files(input_dir)
            self.total_files = len(img_files)
            os.makedirs(output_dir, exist_ok=True)

            print(f"Starting batch processing with {self.workers} workers...")
            print(f"Model: {self.model_file}")
            print(f"Input: {input_dir}")
            print(f"Target: {self.total_files} Images")

            # We create one processor instance per worker to avoid race conditions.
            processors = []
            for _ in range(self.workers):
                proc = LADaSDocumentProcessor(
                    model_name=self.model_file,
                    preload_model=False # Lazy load inside the thread
                )
                processors.append(proc)

            start_time = time.time()

            with ThreadPoolExecutor(max_workers=self.workers) as executor:
                future_to_file = {}
                
                for i, img_path in enumerate(img_files):
                    assigned_processor = processors[i % self.workers]
                    
                    future = executor.submit(
                        self.process_single_image, 
                        assigned_processor, 
                        img_path, 
                        output_dir
                    )
                    future_to_file[future] = img_path
                for future in as_completed(future_to_file):
                    img_path = future_to_file[future]
                    try:
                        _, success, error_msg, _ = future.result()
                        self.update_progress(success, img_path, error_msg)
                    except Exception as e:
                        self.update_progress(False, img_path, f"Critical Thread Error: {e}")

            total_time = time.time() - start_time
            self.print_summary(total_time)
            return len(self.failed_files) == 0

        except Exception:
            traceback.print_exc()
            return False

    def print_summary(self, total_time):
        print("\n" + "=" * 60)
        print("PROCESSING SUMMARY")
        print("=" * 60)
        print(f"Total files: {self.total_files}")
        print(f"Successful:  {len(self.successful_files)}")
        print(f"Failed:      {len(self.failed_files)}")
        print(f"Total time:  {total_time:.2f}s")
        
        if self.total_files > 0:
            avg = total_time / self.total_files
            fps = self.total_files / total_time
            print(f"Average:     {avg:.2f} s/img")
            print(f"Throughput:  {fps:.2f} fps")

        if self.failed_files:
            print("\nErrors:")
            for path, msg in self.failed_files:
                print(f"  {os.path.basename(path)}: {msg}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Parallel LADaS (YOLO) Inference")
    parser.add_argument("input", help="Folder containing PNG/JPG images")
    parser.add_argument("--output", "-o", default="output_ladas", help="Output directory")
    parser.add_argument("--model-file", required=True, help="Path to .pt model file")
    
    parser.add_argument(
        "--workers", "-w", type=int, default=None, 
        help="Number of threads. Default: CPU Count - 1. WARNING: High worker count on GPU may cause OOM."
    )

    args = parser.parse_args()

    runner = LADaSBatchRunner(
        model_file=args.model_file,
        workers=args.workers
    )

    success = runner.process_images(
        input_dir=args.input, 
        output_dir=args.output
    )

    sys.exit(0 if success else 1)