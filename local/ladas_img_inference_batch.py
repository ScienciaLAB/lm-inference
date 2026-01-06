import time
import gc
import argparse
from pathlib import Path

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
            print("Loading model...")
            start_time = time.time()

            self.model = YOLO(Path(self.model_name), verbose=True)
            load_time = time.time() - start_time
            print(f"Model loaded in {load_time:.2f} seconds")


class DocLayNetRunner(LADaSDocumentProcessor):

    def process_images(self, input_dir: str, output_dir: str, chunk_size=50):
        self._load_model()
        
        input_path = Path(input_dir)
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        print(f"Scanning {input_dir}...")
        image_paths = sorted(list(input_path.glob("*.png")))
        
        # Fallback to jpg if no png found
        if not image_paths:
            image_paths = sorted(list(input_path.glob("*.jpg")))
            
        total_images = len(image_paths)
        print(f"Found {total_images} images.")
        
        if total_images == 0:
            print("No images found. Check path.")
            return

        #  Process in Chunks
        processed_count = 0
        
        # We loop through the list in steps of chunk_size (default 50)
        for i in range(0, total_images, chunk_size):
            chunk_files = image_paths[i : i + chunk_size]
            chunk_str_paths = [str(p) for p in chunk_files]
            
            print(f"Processing batch {i} to {min(i + chunk_size, total_images)}...")
            
            # stream=True (Memory Efficient)
            results = self.model.predict(chunk_str_paths, stream=True, verbose=False)
            
            for res in results:
                base_name = Path(res.path).stem
                # Save Individual JSON file
                with open(output_path / f"{base_name}.json", "w", encoding="utf-8") as f:
                    f.write(res.to_json())
                
                processed_count += 1
            
            # Force cleanup to prevent memory crash
            del results
            gc.collect()

        print(f"Finished! Processed {processed_count} images.")
        print(f"Individual JSON files saved to: {output_path}")
        return processed_count

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate LADaS on DocLayNet Images")
    parser.add_argument("input", help="Folder containing PNG images")
    parser.add_argument("--output", "-o", default="output_ladas", help="Output directory")
    parser.add_argument("--model-file", required=True, help="Path to .pt model file")
    
    parser.add_argument("--batch-size", type=int, default=50, help="Images processed per memory clear")

    args = parser.parse_args()

    processor = DocLayNetRunner(
        model_name=args.model_file,
        preload_model=True
    )

    try:
        processor.process_images(
            input_dir=args.input, 
            output_dir=args.output,
            chunk_size=args.batch_size
        )
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        print(f"Error: {e}")