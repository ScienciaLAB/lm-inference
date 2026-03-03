from typing import Any, Dict, List

# Modal GPU pricing per second
GPU_COST_PER_SECOND = {
    "A10G": 0.000264,
    "A10": 0.000306,
    "L4": 0.000222,
    "L40S": 0.000542,
    "A100_40GB": 0.000583,
    "A100_80GB": 0.000694,
    "H100": 0.001097,
    "H200": 0.001261,
    "B200": 0.001736,
    "T4": 0.000164,
}


def get_cost_per_second(gpu_type: str) -> float:
    """Retrieves the cost per second for a given GPU type."""
    if gpu_type not in GPU_COST_PER_SECOND:
        available = ", ".join(GPU_COST_PER_SECOND.keys())
        raise ValueError(f"Unknown GPU type '{gpu_type}'. Available types: {available}")
    return GPU_COST_PER_SECOND[gpu_type]


def filter_and_aggregate(
    bounding_boxes: List[Dict[str, any]], type_aggregation: Dict[str, List[str]]
) -> List[Dict[str, Any]]:
    inverted_dict = {}
    for main_type, sub_types in type_aggregation.items():
        for sub_type in sub_types:
            inverted_dict[sub_type] = main_type

    filtered_boxes = []
    for b in bounding_boxes:
        if b["type"] in inverted_dict:
            b["type"] = inverted_dict[b["type"]]
            filtered_boxes.append(b)

    return filtered_boxes
