import os
import argparse
import json
from pathlib import Path


def calculate_distance(parent, caption):
    """
    Calculate distance between parent and caption.
    Handles: above, below, left, right positions.
    
    Returns:
        (distance, position) if valid relationship
        (None, None) if invalid (overlapping or diagonal)
    """
    
    # Parent edges
    p_top = parent["y"]
    p_bottom = parent["y"] + parent["height"]
    p_left = parent["x"]
    p_right = parent["x"] + parent["width"]
    
    # Caption edges
    c_top = caption["y"]
    c_bottom = caption["y"] + caption["height"]
    c_left = caption["x"]
    c_right = caption["x"] + caption["width"]
    
    # Check overlaps
    # Horizontal overlap: caption and parent share some X range
    h_overlap = (c_left < p_right) and (c_right > p_left)
    
    # Vertical overlap: caption and parent share some Y range
    v_overlap = (c_top < p_bottom) and (c_bottom > p_top)
    
    # Both overlap = boxes intersect = invalid
    if h_overlap and v_overlap:
        return None, None
    
    # BELOW: horizontal overlap + caption is under parent
    if h_overlap and c_top >= p_bottom:
        return c_top - p_bottom, "below"
    
    # ABOVE: horizontal overlap + caption is over parent
    if h_overlap and c_bottom <= p_top:
        return p_top - c_bottom, "above"
    
    # RIGHT: vertical overlap + caption is to the right of parent
    if v_overlap and c_left >= p_right:
        return c_left - p_right, "right"
    
    # LEFT: vertical overlap + caption is to the left of parent
    if v_overlap and c_right <= p_left:
        return p_left - c_right, "left"
    
    # No valid relationship (diagonal)
    return None, None


def link_and_merge(parents, captions):
    """
    For each parent (figure/table), find the nearest caption on same page.
    Uses ranking approach - always picks the closest valid caption.
    
    Args:
        parents: list of figure/table dicts with bounding boxes
        captions: list of caption dicts with bounding boxes
    
    Returns:
        parents list with merged bounding boxes (includes caption area)
    """
    
    used_captions = set()
    
    for parent in parents:
        page = parent["page"]
        
        # Collect ALL valid candidates
        candidates = []
        
        for i, caption in enumerate(captions):
            
            # Skip if already used by another parent
            if i in used_captions:
                continue
            
            # Must be on same page
            if caption["page"] != page:
                continue
            
            # Calculate distance
            distance, position = calculate_distance(parent, caption)
            
            # Skip invalid (overlapping or diagonal)
            if distance is None:
                continue
            
            candidates.append({
                "index": i,
                "caption": caption,
                "distance": distance,
                "position": position
            })
        
        # No candidates found for this parent
        if not candidates:
            continue
        
        # Rank by distance (ascending), pick closest
        candidates.sort(key=lambda x: x["distance"])
        best = candidates[0]
        
        # Mark caption as used
        used_captions.add(best["index"])
        
        # Merge bounding boxes
        c = best["caption"]
        
        p_left = parent["x"]
        p_top = parent["y"]
        p_right = parent["x"] + parent["width"]
        p_bottom = parent["y"] + parent["height"]
        
        c_left = c["x"]
        c_top = c["y"]
        c_right = c["x"] + c["width"]
        c_bottom = c["y"] + c["height"]
        
        # New merged box
        new_left = min(p_left, c_left)
        new_top = min(p_top, c_top)
        new_right = max(p_right, c_right)
        new_bottom = max(p_bottom, c_bottom)
        
        # Update parent with merged box
        parent["x"] = new_left
        parent["y"] = new_top
        parent["width"] = new_right - new_left
        parent["height"] = new_bottom - new_top
    
    return parents


def link_captions_and_merge(paddle_data):
    """
    Links captions to their parent elements and merges bounding boxes.
    
    Separates elements by type:
        - Figures: figure, image, chart
        - Tables: table
        - Figure captions: figure_title, chart_title
        - Table captions: table_title
        - Ignore areas: header, header_image, footer, number
    
    Args:
        paddle_data: list of dicts from PaddlePaddle JSON
    
    Returns:
        (figures, tables, ignore_areas) - all with merged bounding boxes
    """
    
    figures = []
    tables = []
    figure_captions = []
    table_captions = []
    ignore_areas = []
    
    for item in paddle_data:
        item_type = item.get("type", "").lower()
        
        # Figures/Charts/Images
        if item_type in ["figure", "image", "chart"]:
            figures.append(item.copy())
        
        # Tables
        elif item_type == "table":
            tables.append(item.copy())
        
        # Figure/Chart captions
        elif item_type in ["figure_title", "chart_title"]:
            figure_captions.append(item.copy())
        
        # Table captions
        elif item_type == "table_title":
            table_captions.append(item.copy())
        
        # Ignore areas
        elif item_type in ["header", "header_image", "footer", "number"]:
            ignore_areas.append(item.copy())
    
    # Link captions to parents and merge boxes
    figures = link_and_merge(figures, figure_captions)
    tables = link_and_merge(tables, table_captions)
    
    return figures, tables, ignore_areas


def process_single_json(input_path, output_path):
    """
    Process one PaddlePaddle JSON file.
    
    Args:
        input_path: path to input JSON
        output_path: path to output JSON
    
    Returns:
        (num_figures, num_tables, num_ignore) counts
    """
    
    with open(input_path, 'r', encoding='utf-8') as f:
        paddle_data = json.load(f)
    
    figures, tables, ignore_areas = link_captions_and_merge(paddle_data)
    
    # Build output
    output_data = []
    
    for item in figures:
        output_data.append({
            "page": int(item["page"]),
            "x": int(item["x"]),
            "y": int(item["y"]),
            "width": int(item["width"]),
            "height": int(item["height"]),
            "type": "figure"
        })
    
    for item in tables:
        output_data.append({
            "page": int(item["page"]),
            "x": int(item["x"]),
            "y": int(item["y"]),
            "width": int(item["width"]),
            "height": int(item["height"]),
            "type": "table"
        })
    
    for item in ignore_areas:
        output_data.append({
            "page": int(item["page"]),
            "x": int(item["x"]),
            "y": int(item["y"]),
            "width": int(item["width"]),
            "height": int(item["height"]),
            "type": "ignore"
        })
    
    # Sort by page, then y position
    output_data.sort(key=lambda x: (x["page"], x["y"]))
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, indent=2)
    
    return len(figures), len(tables), len(ignore_areas)


def main():
    parser = argparse.ArgumentParser(
        description="Process PaddlePaddle JSON: link captions to figures/tables and merge bounding boxes"
    )
    parser.add_argument(
        "input_dir",
        help="Folder containing PaddlePaddle JSON files"
    )
    parser.add_argument(
        "--output", "-o",
        required=True,
        help="Output folder for processed JSON files"
    )
    args = parser.parse_args()

    # Create output directory
    os.makedirs(args.output, exist_ok=True)
    
    # Find all JSON files
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
    total_ignore = 0
    success_count = 0
    error_count = 0
    
    for json_file in json_files:
        output_file = Path(args.output) / json_file.name
        
        try:
            n_fig, n_tab, n_ign = process_single_json(json_file, output_file)
            total_figures += n_fig
            total_tables += n_tab
            total_ignore += n_ign
            success_count += 1
            print(f" {json_file.name}: {n_fig} figures, {n_tab} tables, {n_ign} ignore")
        except Exception as e:
            error_count += 1
            print(f"✗ {json_file.name}: {e}")
    
    print("-" * 60)
    print(f"DONE: {success_count} success, {error_count} errors")
    print(f"TOTAL: {total_figures} figures, {total_tables} tables, {total_ignore} ignore")


if __name__ == "__main__":
    main()