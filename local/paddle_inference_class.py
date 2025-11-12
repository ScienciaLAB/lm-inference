import json
import os
import tempfile
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

from paddleocr import LayoutDetection


class DocumentProcessor:
    def __init__(self, model_name: str = "PP-DocLayout-S", dpi: int = 70, temp_dir: str = None, preload_model:bool = False):
        self.model_name = model_name
        self.dpi = dpi
        self.model: Optional[LayoutDetection] = None
        self.cleanup_after_processing = True

        # Create temporary directory
        if temp_dir is None:
            self.temp_dir = Path(tempfile.mkdtemp(prefix="paddle_inference_"))
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
        except:
            pass

    def _ensure_directories(self, pdf_path: str, output_dir: Path) -> Path:
        doc_name = Path(pdf_path).stem
        doc_output_dir = output_dir / doc_name
        doc_output_dir.mkdir(parents=True, exist_ok=True)
        return doc_output_dir

    def _load_model(self) -> None:
        if self.model is None:
            from paddleocr import LayoutDetection

            print("Loading model...")
            start_time = time.time()
            self.model = LayoutDetection(model_name=self.model_name)
            load_time = time.time() - start_time
            print(f"Model loaded in {load_time:.2f} seconds")

    def pdf_to_images(self, pdf_path: str, output_dir: Path) -> List[Path]:
        from pdf2image import convert_from_path

        print(f"Converting PDF to images: {pdf_path}")
        images = convert_from_path(pdf_path, dpi=self.dpi, thread_count=os.cpu_count())
        image_paths = []

        for i, image in enumerate(images, 1):
            image_path = output_dir / f"page_{i:04d}.jpg"
            image.save(image_path, 'JPEG')
            image_paths.append(image_path)

        print(f"Converted {len(images)} pages to images")
        return image_paths

    def process(self, pdf_path: str, output_dir: Path) -> List[Dict]:
        self._load_model()

        image_paths = self.pdf_to_images(pdf_path, output_dir)
        image_path_strings = [str(path) for path in image_paths]

        start_time = time.time()
        output = self.model.predict(
            image_path_strings,
            batch_size=os.cpu_count(),
            layout_nms=True
        )
        inference_time = time.time() - start_time
        print(f"Process completed in {inference_time:.2f} seconds")

        # Save results
        for i, res in enumerate(output):
            res.page_index = i + 1
            base_name = f"res_{i}"

            res.save_to_img(save_path=str(output_dir / f"{base_name}.jpg"))
            res.save_to_json(save_path=str(output_dir / f"{base_name}.json"))

        return output

    def cleanup_temp_files(self, doc_output_dir: Path) -> None:
        if self.cleanup_after_processing:
            # Keep final results, remove intermediate files if needed
            pass  # For now, keep all generated files

    def process_document(self, pdf_path: str, output_dir: str = "output") -> Dict[str, Any]:
        if not os.path.exists(pdf_path):
            raise FileNotFoundError(f"Input file '{pdf_path}' does not exist")

        if not pdf_path.lower().endswith('.pdf'):
            raise ValueError("Input file must be a PDF document")

        print(f"Processing document: {pdf_path}")
        start_time = time.time()

        output_path = Path(output_dir)
        doc_output_dir = self._ensure_directories(pdf_path, output_path)

        try:
            output = self.process(pdf_path, doc_output_dir)
            self.cleanup_temp_files(doc_output_dir)

            processing_time = time.time() - start_time
            num_pages = len(output)
            avg_time_per_page = processing_time / num_pages

            result = {
                'success': True,
                'pdf_path': pdf_path,
                'output_dir': str(doc_output_dir),
                'num_pages': num_pages,
                'processing_time': processing_time,
                'avg_time_per_page': avg_time_per_page,
                'model_used': self.model_name
            }

            print(f"Document processed in {processing_time:.2f} seconds "
                  f"({avg_time_per_page:.2f} sec/page)")
            return result

        except Exception as e:
            error_result = {
                'success': False,
                'pdf_path': pdf_path,
                'error': str(e),
                'processing_time': time.time() - start_time
            }
            print(f"Error processing {pdf_path}: {str(e)}")
            return error_result


def process_grobid_output(output_dir: Path) -> None:
    print(f"Processing GROBID analysis on: {output_dir}")

    json_files = list(output_dir.glob('res_*.json'))
    if not json_files:
        print("No JSON files found for GROBID processing")
        return

    all_filtered_results = []

    for json_file in sorted(json_files):
        filtered_result = filter_layout_elements(str(json_file))
        all_filtered_results.append(filtered_result)

    # Create aggregated output
    aggregate_filtered_elements(output_dir, all_filtered_results)


def filter_layout_elements(json_file_path: str) -> Dict[str, List[Dict]]:
    try:
        with open(json_file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

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

    for result in all_results:
        if result:
            aggregated_data['tables'].extend(result.get('tables', []))
            aggregated_data['figures'].extend(result.get('figures', []))
            aggregated_data['equations'].extend(result.get('equations', []))

    aggregated_data['summary']['total_tables'] = len(aggregated_data['tables'])
    aggregated_data['summary']['total_figures'] = len(aggregated_data['figures'])
    aggregated_data['summary']['total_equations'] = len(aggregated_data['equations'])

    aggregated_file = output_dir / 'aggregated_elements.json'
    with open(aggregated_file, 'w', encoding='utf-8') as f:
        json.dump(aggregated_data, f, indent=2, ensure_ascii=False)

    print(f"Found {aggregated_data['summary']['total_tables']} tables, "
          f"{aggregated_data['summary']['total_figures']} figures, "
          f"{aggregated_data['summary']['total_equations']} equations")


def process_elements(output_dir: str, element_types: List[str], filter_func, filter_type: str = "custom") -> None:
    """
    Unified method to process layout results and extract specified element types

    Args:
        output_dir: Directory containing the JSON files to process
        element_types: List of element types to extract (e.g., ['tables', 'figures', 'equations'])
        filter_func: Function to filter elements from JSON files
        filter_type: Type of processing ('display', 'paratext', or 'custom') for logging/filename
    """
    output_path = Path(output_dir)
    print(f"Processing {filter_type} elements in: {output_path}")

    json_files = list(output_path.glob('res_*.json'))
    if not json_files:
        print(f"No JSON files found for {filter_type} processing")
        return

    all_filtered_results = []

    for json_file in sorted(json_files):
        filtered_result = filter_func(str(json_file))
        all_filtered_results.append(filtered_result)

    # Create aggregated output
    aggregate_elements(output_path, all_filtered_results, element_types, filter_type)


def filter_display_elements(json_file_path: str) -> Dict[str, List[Dict]]:
    """Filter JSON results to extract only display elements (tables, figures, equations)"""
    try:
        with open(json_file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

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


def filter_paratext_elements(json_file_path: str) -> Dict[str, List[Dict]]:
    """Filter JSON results to extract only paratext elements (headers, footers)"""
    try:
        with open(json_file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        filtered_elements = {'headers': [], 'footers': []}

        for box in data.get('boxes', []):
            label = box.get('label', '').lower()
            if any(x in label for x in ['header', 'heading', 'title']):
                filtered_elements['headers'].append(box)
            elif any(x in label for x in ['footer', 'footnote']):
                filtered_elements['footers'].append(box)

        return filtered_elements

    except:
        return {'headers': [], 'footers': []}


def aggregate_elements(output_dir: Path, all_results: List[Dict], element_types: List[str] = None, filter_type: str = "custom") -> None:
    """
    Unified method to aggregate elements and save to JSON file

    Args:
        output_dir: Directory to save the aggregated results
        all_results: List of filtered results from all pages
        element_types: List of element types to include (e.g., ['tables', 'figures', 'equations'])
        filter_type: Type of filtering applied ('display', 'paratext', or 'custom')
    """
    if element_types is None:
        element_types = []

    # Initialize aggregated data structure
    aggregated_data = {
        'metadata': {
            'total_pages': len(all_results),
            'processed_files': [],
            'filter_type': filter_type
        },
        'elements': {element_type: [] for element_type in element_types},
        'summary': {f'total_{element_type}': 0 for element_type in element_types}
    }

    # Aggregate elements from all results
    for result in all_results:
        if result:
            for element_type in element_types:
                aggregated_data['elements'][element_type].extend(result.get(element_type, []))

    # Calculate summaries
    for element_type in element_types:
        aggregated_data['summary'][f'total_{element_type}'] = len(aggregated_data['elements'][element_type])

    # Generate filename
    if filter_type in ['display', 'paratext']:
        filename = f'aggregated_{filter_type}_elements.json'
    else:
        element_str = '_'.join(element_types)
        filename = f'aggregated_{element_str}_elements.json'

    aggregated_file = output_dir / filename

    # Save to file
    with open(aggregated_file, 'w', encoding='utf-8') as f:
        json.dump(aggregated_data, f, indent=2, ensure_ascii=False)

    # Print summary
    summary_parts = []
    for element_type in element_types:
        count = aggregated_data['summary'][f'total_{element_type}']
        summary_parts.append(f"{count} {element_type}")

    print(f"Found {', '.join(summary_parts)}")
    print(f"Results saved to: {aggregated_file}")




if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='Process PDF documents with PaddleOCR layout detection')
    parser.add_argument('input', help='Input PDF document path')
    parser.add_argument('--output', '-o', default='output', help='Output directory for processed images and results')
    parser.add_argument('--model-name',
                        choices=['PP-DocLayout-L', 'PP-DocLayout-M', 'PP-DocLayout-S', 'PP-DocLayoutV2',
                                 'PP-DocBlockLayout'],
                        default='PP-DocLayout-S', help='Model name for layout detection (default: PP-DocLayout-S)')
    parser.add_argument('--dpi', type=int, default=70, help='DPI for PDF to image conversion (default: 70)')
    parser.add_argument('--temp-dir', help='Temporary directory for processing (default: auto-generated)')
    parser.add_argument('--only',
                        choices=["display", "paratext"],
                        help='Parse JSON output to extract only "display" elements: '
                             'tables, figures, and equations, "paratext": sugar coat such '
                             'headers and footers, and create aggregated results')

    args = parser.parse_args()

    processor = DocumentProcessor(
        model_name=args.model_name,
        dpi=args.dpi,
        temp_dir=args.temp_dir,
        preload_model=True
    )

    result = processor.process_document(args.input, args.output)

    # Apply filtering if requested
    if args.only:
        if args.only == "display":
            process_elements(
                result['output_dir'],
                ['tables', 'figures', 'equations'],
                filter_display_elements,
                'display'
            )
        elif args.only == "paratext":
            process_elements(
                result['output_dir'],
                ['headers', 'footers'],
                filter_paratext_elements,
                'paratext'
            )

    print("Processing result:", result)
