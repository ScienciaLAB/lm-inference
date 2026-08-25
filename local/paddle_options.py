"""Shared CLI options for the PaddleOCR layout detectors.

Kept free of any `paddleocr` import so that the batch runners and the servers
can validate arguments in the parent process without loading the framework.
"""

from typing import Dict, List, Optional, Union

# Layout detectors selectable from the CLI. PP-DocLayoutV3 is the instance
# segmentation successor of PP-DocLayoutV2: it uses a different, finer label
# vocabulary (25 classes, with `display_formula` / `inline_formula` in place of
# the single `formula` class) and additionally returns `polygon_points`.
LAYOUT_MODEL_CHOICES = [
    "PP-DocLayout-L",
    "PP-DocLayout-M",
    "PP-DocLayout-S",
    "PP-DocLayoutV2",
    "PP-DocLayoutV3",
    "PP-DocBlockLayout",
]

LAYOUT_MERGE_BBOXES_MODES = ["union", "large", "small"]


def parse_threshold(value: str) -> Union[float, Dict[Union[int, str], float]]:
    """Parse a --threshold argument.

    Either a single score applied to every class (``0.4``) or per-class scores
    given as comma-separated ``class:score`` pairs, where ``class`` is a class
    index or a label of the model (``inline_formula:0.2,15:0.2``).
    """
    if ":" not in value:
        return float(value)

    thresholds: Dict[Union[int, str], float] = {}
    for pair in value.split(","):
        key, separator, score = pair.partition(":")
        key = key.strip()
        if not key or not separator or not score.strip():
            raise ValueError(f"Malformed threshold entry: '{pair}'")
        thresholds[int(key) if key.isdigit() else key] = float(score)
    return thresholds


def resolve_threshold(
    threshold: Union[float, Dict[Union[int, str], float], None],
    labels: Optional[List[str]],
) -> Union[float, Dict[int, float], None]:
    """Turn label names used as threshold keys into the class indices PaddleOCR expects."""
    if not isinstance(threshold, dict):
        return threshold

    resolved = {}
    for key, score in threshold.items():
        if isinstance(key, str) and not key.isdigit():
            if not labels or key not in labels:
                known = ", ".join(labels) if labels else "none reported by the model"
                raise ValueError(
                    f"Unknown layout label '{key}'. Labels of this model: {known}"
                )
            key = labels.index(key)
        resolved[int(key)] = float(score)
    return resolved
