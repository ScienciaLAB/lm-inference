import os
import shutil
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from paddleocr import LayoutDetection


class BaseDocumentProcessor:
    def __init__(self, model_name: str = None, dpi: int = 70, temp_dir: str = None,
                 preload_model: bool = False):
        self.model_name = model_name
        self.dpi = dpi
        self.model: Optional[Union[LayoutDetection, Any]] = None
        self.cleanup_after_processing = True

        # Create temporary directory
        if temp_dir is None:
            self.temp_dir = Path(tempfile.mkdtemp(prefix="tmp_inference_"))
        else:
            self.temp_dir = Path(temp_dir)
            self.temp_dir.mkdir(parents=True, exist_ok=True)

        if preload_model:
            self._load_model()

        print(f"Using temporary directory: {self.temp_dir}")

    def __del__(self):
        """Cleanup temporary directory when object is destroyed"""
        try:
            import shutil
            if hasattr(self, 'temp_dir') and self.temp_dir.exists():
                shutil.rmtree(self.temp_dir)
                print(f"Cleaned up temporary directory: {self.temp_dir}")
        except OSError:
            pass

    def _ensure_directories(self, pdf_path: str, output_dir: Path) -> Path:
        doc_name = Path(pdf_path).stem
        doc_output_dir = output_dir / doc_name
        doc_output_dir.mkdir(parents=True, exist_ok=True)
        return doc_output_dir

    def _load_model(self) -> None:
        raise NotImplementedError("Not implemented in the base class. Please implement in subclass.")

    def pdf_to_images(self, pdf_path: str, output_dir: Path) -> List[Path]:
        from pdf2image import convert_from_path

        print(f"Converting PDF to images: {pdf_path}")
        images = convert_from_path(pdf_path, dpi=self.dpi, thread_count=os.cpu_count())
        image_paths = []

        for i, image in enumerate(images, 1):
            image_path = output_dir / f"page_{i:04d}.jpg"
            # TODO: consider saving pngs
            image.save(image_path, 'JPEG')
            image_paths.append(image_path)

        print(f"Converted {len(images)} pages to images")
        return image_paths

    def process(self, pdf_path: str, output_dir: Path) -> List[Dict]:
        self._load_model()
        raise NotImplementedError("Not implemented in the base class. Please implement in subclass.")

    def cleanup_temp_files(self, doc_output_dir: Path, main_output_dir: Path = None, cleanup_images: bool = False,
                           cleanup_rename: bool = False, pdf_name: str = None) -> None:
        """
        Clean up temporary files generated during processing

        Args:
            doc_output_dir: Directory containing processed files (document subdirectory)
            main_output_dir: Main output directory where aggregated files are saved
            cleanup_images: Whether to remove intermediate image files and entire directory
            cleanup_rename: Whether to rename aggregated JSON to PDF name
            pdf_name: Name of the input PDF file (without extension) for renaming
        """
        if not self.cleanup_after_processing and not cleanup_images:
            return

        try:
            # Rename aggregated JSON to PDF name if requested (before cleanup)
            if cleanup_rename and pdf_name and main_output_dir:
                # Find the aggregated JSON file in main output directory
                aggregated_files = list(main_output_dir.glob('aggregated_*_elements.json'))
                if aggregated_files:
                    # Take the first aggregated file found
                    source_file = aggregated_files[0]
                    target_file = main_output_dir / f"{pdf_name}.json"

                    # Rename/move the file
                    source_file.rename(target_file)
                    print(f"Renamed aggregated output to: {target_file}")
                else:
                    print("No aggregated JSON file found to rename")

            # Remove entire document subdirectory if cleanup_images is True
            if cleanup_images and doc_output_dir.exists():
                shutil.rmtree(doc_output_dir)
                print(f"Removed temporary directory: {doc_output_dir}")

        except OSError as e:
            print(f"Error during cleanup: {str(e)}")

    def process_document(self, pdf_path: str, output_dir: str = "output") -> Dict[str, Any]:
        raise NotImplementedError("Not implemented in the base class. Please implement in subclass.")

