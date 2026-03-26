import os
import argparse
import json
from pathlib import Path


def link_and_merge(parents, captions):
    """
    For each parent (image/table), find the nearest caption on same page.
    Caption can be ABOVE or BELOW the parent.
    """
    
    used_captions = set()
    
    for parent in parents:
        page = parent["page"]
        
        # Parent box coordinates
        parent_top = parent["y"]
        parent_bottom = parent["y"] + parent["height"]
        parent_left = parent["x"]
        parent_right = parent["x"] + parent["width"]
        
        best_caption = None
        best_distance = float('inf')
        
        for i, caption in enumerate(captions):
            
            # Skip if already used
            if i in used_captions:
                continue
            
            # Rule 1: Must be same page
            if caption["page"] != page:
                continue
            
            # Caption box coordinates
            caption_top = caption["y"]
            caption_bottom = caption["y"] + caption["height"]
            caption_left = caption["x"]
            caption_right = caption["x"] + caption["width"]
            
            # Rule 2: Calculate distance (above OR below)
            
            # Case A: Caption is BELOW parent
            if caption_top >= parent_bottom:
                distance = caption_top - parent_bottom
            
            # Case B: Caption is ABOVE parent
            elif caption_bottom <= parent_top:
                distance = parent_top - caption_bottom
            
            # Case C: They overlap vertically (skip)
            else:
                continue
            
            # Rule 3: Must be close (within 50 points)
            if distance > 50:
                continue
            
            # Rule 4: Must have horizontal overlap (same column)
            if caption_right < parent_left or caption_left > parent_right:
                continue
            
            if distance < best_distance:
                best_distance = distance
                best_caption = (i, caption)
        
        # Merge if we found a caption
        if best_caption:
            idx, caption = best_caption
            used_captions.add(idx)
            
            # Calculate merged bounding box
            caption_left = caption["x"]
            caption_right = caption["x"] + caption["width"]
            caption_top = caption["y"]
            caption_bottom = caption["y"] + caption["height"]
            
            new_left = min(parent_left, caption_left)
            new_top = min(parent_top, caption_top)
            new_right = max(parent_right, caption_right)
            new_bottom = max(parent_bottom, caption_bottom)
            
            # Update parent with merged box
            parent["x"] = new_left
            parent["y"] = new_top
            parent["width"] = new_right - new_left
            parent["height"] = new_bottom - new_top
    
    return parents


def link_captions_and_merge(paddle_data):
    """
    Links captions to their parent elements and merges bounding boxes.
    """
    
    # Separate elements by type
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
    
    # Link and merge
    figures = link_and_merge(figures, figure_captions)
    tables = link_and_merge(tables, table_captions)
    
    return figures, tables, ignore_areas


def process_single_json(input_path, output_path):
    """
    Process one PaddlePaddle JSON file.
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
        description="Process PaddlePaddle JSON: link captions and convert types"
    )
    parser.add_argument("input_dir", help="Folder containing PaddlePaddle JSONs")
    parser.add_argument("--output", "-o", required=True, help="Output folder")
    args = parser.parse_args()

    os.makedirs(args.output, exist_ok=True)
    
    json_files = sorted(Path(args.input_dir).glob("*.json"))
    
    print(f"Processing {len(json_files)} JSON files...")
    print(f"Output: {args.output}")
    print("-" * 50)
    
    total_figures = 0
    total_tables = 0
    total_ignore = 0
    
    for json_file in json_files:
        output_file = Path(args.output) / json_file.name
        
        try:
            n_fig, n_tab, n_ign = process_single_json(json_file, output_file)
            total_figures += n_fig
            total_tables += n_tab
            total_ignore += n_ign
            print(f"✓ {json_file.name}: {n_fig} figures, {n_tab} tables, {n_ign} ignore")
        except Exception as e:
            print(f"✗ {json_file.name}: {e}")
    
    print("-" * 50)
    print(f"TOTAL: {total_figures} figures, {total_tables} tables, {total_ignore} ignore")


if __name__ == "__main__":
    main()