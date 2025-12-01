import json
import os
import time
from pathlib import Path
from typing import Any, Dict, List

from transformers import AutoModel

from base_inference import BaseDocumentProcessor


class LADaSDocumentProcessor(BaseDocumentProcessor):
    def __init__(self, model_name: str = None, dpi: int = 70, temp_dir: str = None,
                 preload_model: bool = False):
        super().__init__(model_name, dpi, temp_dir, preload_model)

    def _load_model(self) -> None:
        if self.model is None:
            print("Loading model...")
            start_time = time.time()

            self.model = AutoModel.from_pretrained(Path(self.model_name), from_pt=True)
            load_time = time.time() - start_time
            print(f"Model loaded in {load_time:.2f} seconds")

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
                'main_output_dir': str(output_path),
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


def load_transform_elements(
        document_output_dir: str,
) -> List[Dict[str, Any]]:
    document_output_path = Path(document_output_dir)

    json_files = list(document_output_path.glob('res_*.json'))
    if not json_files:
        print(f"No JSON files found in {document_output_dir} for processing")
        return []

    standard_elements = []

    for json_file in sorted(json_files):
        page_number = int(json_file.stem.split('_')[1]) + 1
        with open(json_file, 'r', encoding='utf-8') as f:
            data = json.load(f)

        for element in data['boxes']:
            coords = element.get('coordinate')
            x1, y1, x2, y2 = coords
            x = int(x1)
            y = int(y1)
            width = int(x2 - x1)
            height = int(y2 - y1)

            # Create standard format element
            standard_element = {
                "page": page_number,
                "x": x,
                "y": y,
                "width": width,
                "height": height,
                "type": element.get('label')
            }

            standard_elements.append(standard_element)

    return standard_elements


def filter_and_aggregate(bounding_boxes: List[Dict[str, any]], type_aggregation: Dict[str, List[str]]) -> List[
    Dict[str, Any]]:
    inverted_dict = {}
    for main_type, sub_types in type_aggregation.items():
        for sub_type in sub_types:
            inverted_dict[sub_type] = main_type

    filtered_boxes = []
    for b in bounding_boxes:
        if b['type'] in inverted_dict:
            b['type'] = inverted_dict[b['type']]
            filtered_boxes.append(b)

    return filtered_boxes


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='Process PDF documents with PaddleOCR layout detection')
    parser.add_argument('input', help='Input PDF document path')
    parser.add_argument('--output', '-o', default='output', help='Output directory for processed images and results')
    parser.add_argument(
        '--model-file',
        required=True,
        help='Model file for layout'
    )
    parser.add_argument('--dpi', type=int, default=70, help='DPI for PDF to image conversion (default: 70)')
    parser.add_argument('--temp-dir', help='Temporary directory for processing (default: auto-generated)')
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

    processor = LADaSDocumentProcessor(
        model_name=args.model_file,
        dpi=args.dpi,
        temp_dir=args.temp_dir,
        preload_model=True
    )

    result = processor.process_document(args.input, args.output)

    if not result.get('success', False):
        print(f"Processing failed: {result.get('error', 'Unknown error')}")

    bounding_boxes = load_transform_elements(
        result['output_dir']
    )

    if args.only:
        figure_type_aggregation = {
            "figure": ["figure", "image", "chart", "figure_text", "chart_text"],
            "table": ["table", "table_text"],
            "equation": ["equation", "formula", "equation_text"]
        }

        paratext_type_aggregation = {
            "headnote": ["header"],
            "footer": ["footer"]
        }

        if args.only == "display":
            bounding_boxes = filter_and_aggregate(bounding_boxes, figure_type_aggregation)
        elif args.only == "paratext":
            bounding_boxes = filter_and_aggregate(bounding_boxes, paratext_type_aggregation)
        elif args.only == "grobid":
            grobid_type_aggregation = {**figure_type_aggregation, **paratext_type_aggregation}
            bounding_boxes = filter_and_aggregate(bounding_boxes, grobid_type_aggregation)

    output_file = Path(args.input).stem + ".json"
    output_file_path = Path(result['main_output_dir']) / output_file

    with open(output_file_path, 'w', encoding='utf-8') as f:
        json.dump(bounding_boxes, f, indent=2, ensure_ascii=False)

    print(f"Results saved to: {output_file_path}")

    # Apply cleanup if requested
    if hasattr(args, 'cleanup_images') and args.cleanup_images:
        doc_output_dir = Path(result['output_dir'])
        main_output_dir = Path(result['main_output_dir'])
        pdf_name = Path(args.input).stem
        processor.cleanup_temp_files(
            doc_output_dir,
            main_output_dir,
            cleanup_images=True,
            cleanup_rename=True,
            pdf_name=pdf_name
        )

    print("Processing result:", result)
