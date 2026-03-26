from pathlib import Path
from ultralytics import YOLO
import argparse

"""
to run this script, use the following command in the terminal: 
python print_ladas_labels.py --model path/to/your/model.pt
"""


def main(model_path):
    # Load model
    model = YOLO(Path(model_path))

    # Get class names
    names = model.names

    print("\nLADaS Model Output Labels ===\n")

    if isinstance(names, dict):
        for class_id, class_name in names.items():
            print(f"{class_id}: {class_name}")
    else:
        for class_id, class_name in enumerate(names):
            print(f"{class_id}: {class_name}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Print LADaS model output labels")
    parser.add_argument(
        "--model", required=True, help="Path to LADaS YOLO model file (.pt)"
    )
    args = parser.parse_args()

    main(args.model)
