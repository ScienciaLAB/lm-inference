import os
import argparse
import json
from pathlib import Path
from abc import ABC, abstractmethod


class CaptionMerger(ABC):
    @abstractmethod
    def merge(self, parents, captions):
        """
        Link captions to parent elements and merge their bounding boxes.

        Args:
            parents: list of figure/table dicts
            captions: list of caption dicts

        Returns:
            parents list with updated bounding boxes
        """
        raise NotImplementedError("Subclasses must implement merge()")

    def _merge_boxes(self, parent, caption):
        """Merge two boxes into one (union)."""
        p_x1 = parent["x"]
        p_y1 = parent["y"]
        p_x2 = parent["x"] + parent["width"]
        p_y2 = parent["y"] + parent["height"]

        c_x1 = caption["x"]
        c_y1 = caption["y"]
        c_x2 = caption["x"] + caption["width"]
        c_y2 = caption["y"] + caption["height"]

        parent["x"] = min(p_x1, c_x1)
        parent["y"] = min(p_y1, c_y1)
        parent["width"] = max(p_x2, c_x2) - parent["x"]
        parent["height"] = max(p_y2, c_y2) - parent["y"]


# First approach: Simple threshold-based merger


class ThresholdMerger(CaptionMerger):
    """
    Simple approach:
    - Requires horizontal overlap (same column)
    - Defined threshold: rejects captions beyond max_distance
    """

    def __init__(self, max_distance=50):
        self.max_distance = max_distance

    def merge(self, parents, captions):
        used_captions = set()

        for parent in parents:
            page = parent["page"]

            parent_top = parent["y"]
            parent_bottom = parent["y"] + parent["height"]
            parent_left = parent["x"]
            parent_right = parent["x"] + parent["width"]

            best_caption = None
            best_distance = float("inf")

            for i, caption in enumerate(captions):
                if i in used_captions:
                    continue

                if caption["page"] != page:
                    continue

                caption_top = caption["y"]
                caption_bottom = caption["y"] + caption["height"]
                caption_left = caption["x"]
                caption_right = caption["x"] + caption["width"]

                if caption_top >= parent_bottom:
                    distance = caption_top - parent_bottom
                elif caption_bottom <= parent_top:
                    distance = parent_top - caption_bottom
                else:
                    continue

                #  threshold
                if distance > self.max_distance:
                    continue

                # Must have horizontal overlap
                if caption_right < parent_left or caption_left > parent_right:
                    continue

                if distance < best_distance:
                    best_distance = distance
                    best_caption = (i, caption)

            # Merge ONE best caption
            if best_caption:
                idx, caption = best_caption
                used_captions.add(idx)
                self._merge_boxes(parent, caption)

        return parents


# Second approach: Distance-based merger


class DistanceMerger(CaptionMerger):
    """
    - Checks the 4 directions: above, below, left, right
    - Uses horizontal/vertical overlap logic
    - No hard threshold: ranks ALL candidates by distance
    - Picks the closest valid caption per parent
    """

    def merge(self, parents, captions):
        used_captions = set()

        for parent in parents:
            page = parent["page"]

            candidates = []

            for i, caption in enumerate(captions):
                if i in used_captions:
                    continue

                if caption["page"] != page:
                    continue

                distance, position = self._calculate_distance(parent, caption)

                if distance is None:
                    continue

                candidates.append(
                    {
                        "index": i,
                        "caption": caption,
                        "distance": distance,
                        "position": position,
                    }
                )

            if not candidates:
                continue

            # Rank by distance, pick closest
            candidates.sort(key=lambda x: x["distance"])
            best = candidates[0]

            used_captions.add(best["index"])
            self._merge_boxes(parent, best["caption"])

        return parents

    def _calculate_distance(self, parent, caption):
        """
        Calculate distance between parent and caption.
        Handles: above, below, left, right.
        Returns (distance, position) or (None, None).
        """
        p_top = parent["y"]
        p_bottom = parent["y"] + parent["height"]
        p_left = parent["x"]
        p_right = parent["x"] + parent["width"]

        c_top = caption["y"]
        c_bottom = caption["y"] + caption["height"]
        c_left = caption["x"]
        c_right = caption["x"] + caption["width"]

        h_overlap = (c_left < p_right) and (c_right > p_left)
        v_overlap = (c_top < p_bottom) and (c_bottom > p_top)

        # Boxes intersect
        if h_overlap and v_overlap:
            return None, None

        # Below
        if h_overlap and c_top >= p_bottom:
            return c_top - p_bottom, "below"

        # Above
        if h_overlap and c_bottom <= p_top:
            return p_top - c_bottom, "above"

        # Right
        if v_overlap and c_left >= p_right:
            return c_left - p_right, "right"

        # Left
        if v_overlap and c_right <= p_left:
            return p_left - c_right, "left"

        return None, None


def link_captions_and_merge(paddle_data, merger: CaptionMerger):
    """
    Separates elements by type, then uses the given merger strategy.
    """
    figures = []
    tables = []
    figure_captions = []
    table_captions = []
    paratext_areas = []

    for item in paddle_data:
        item_type = item.get("type", "").lower()

        if item_type in ["figure", "image", "chart"]:
            figures.append(item.copy())
        elif item_type == "table":
            tables.append(item.copy())
        elif item_type in ["figure_title", "chart_title"]:
            figure_captions.append(item.copy())
        elif item_type == "table_title":
            table_captions.append(item.copy())
        elif item_type in ["header", "header_image", "footer", "number"]:
            paratext_areas.append(item.copy())

    # Use the merger strategy
    figures = merger.merge(figures, figure_captions)
    tables = merger.merge(tables, table_captions)

    return figures, tables, paratext_areas


def _to_output_item(item, output_type):
    return {
        "page": int(item["page"]),
        "x": int(item["x"]),
        "y": int(item["y"]),
        "width": int(item["width"]),
        "height": int(item["height"]),
        "type": output_type,
    }


def process_single_json(input_path, output_path, merger: CaptionMerger, only=None):
    """Process one JSON file using the given merger strategy.

    Args:
        only: set of categories to include ("figure", "table", "paratext").
              None means include all.
    """

    with open(input_path, "r", encoding="utf-8") as f:
        paddle_data = json.load(f)

    figures, tables, paratext_areas = link_captions_and_merge(paddle_data, merger)

    output_data = []

    if only is None or "figure" in only:
        for item in figures:
            output_data.append(_to_output_item(item, "figure"))

    if only is None or "table" in only:
        for item in tables:
            output_data.append(_to_output_item(item, "table"))

    if only is None or "paratext" in only:
        for item in paratext_areas:
            output_data.append(_to_output_item(item, "paratext"))

    output_data.sort(key=lambda x: (x["page"], x["y"]))

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(output_data, f, indent=2)

    return len(figures), len(tables), len(paratext_areas)


# MAIN


def main():
    parser = argparse.ArgumentParser(
        description="Link captions to figures/tables and merge bounding boxes"
    )
    parser.add_argument("input_dir", help="Folder containing PaddlePaddle JSONs")
    parser.add_argument("--output", "-o", required=True, help="Output folder")
    parser.add_argument(
        "--strategy",
        "-s",
        choices=["threshold", "distance"],
        default="distance",
    )
    parser.add_argument(
        "--max-distance",
        "-d",
        type=int,
        default=50,
    )
    parser.add_argument(
        "--only",
        help="Comma-separated list of categories to include in output: "
        "figure, table, paratext (default: all)",
    )
    args = parser.parse_args()

    # Parse --only filter
    only = None
    if args.only:
        valid = {"figure", "table", "paratext"}
        only = {s.strip() for s in args.only.split(",")}
        unknown = only - valid
        if unknown:
            parser.error(f"Unknown categories: {', '.join(unknown)}. Valid: {', '.join(sorted(valid))}")

    # Create the merger based on strategy choice
    if args.strategy == "threshold":
        merger = ThresholdMerger(max_distance=args.max_distance)
        print(f"Strategy: THRESHOLD (max_distance={args.max_distance})")
    else:
        merger = DistanceMerger()
        print("Strategy: DISTANCE (rank all, pick closest)")

    if only:
        print(f"Output filter: {', '.join(sorted(only))}")

    os.makedirs(args.output, exist_ok=True)

    json_files = sorted(Path(args.input_dir).glob("*.json"))

    if not json_files:
        print(f"No JSON files found in {args.input_dir}")
        return

    print(f"Processing {len(json_files)} JSON files...")
    print(f"Input:  {args.input_dir}")
    print(f"Output: {args.output}")
    print("-" * 60)

    total_figures = 0
    total_tables = 0
    total_paratext = 0
    success_count = 0
    error_count = 0

    for json_file in json_files:
        output_file = Path(args.output) / json_file.name

        try:
            n_fig, n_tab, n_para = process_single_json(json_file, output_file, merger, only=only)
            total_figures += n_fig
            total_tables += n_tab
            total_paratext += n_para
            success_count += 1
            print(
                f"✓ {json_file.name}: {n_fig} figures, {n_tab} tables, {n_para} paratext"
            )
        except Exception as e:
            error_count += 1
            print(f"✗ {json_file.name}: {e}")

    print("-" * 60)
    print(f"DONE: {success_count} success, {error_count} errors")
    print(
        f"TOTAL: {total_figures} figures, {total_tables} tables, {total_paratext} paratext"
    )


if __name__ == "__main__":
    main()
