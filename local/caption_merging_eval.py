import argparse
import copy
import json
import random
import statistics
from pathlib import Path

from .caption_merging import DistanceMerger, ThresholdMerger

STRATEGIES = {
    "threshold": lambda args: ThresholdMerger(max_distance=args.max_distance),
    "distance": lambda _: DistanceMerger(),
}


def compute_iou(box_a, box_b):
    """IoU between two dicts with x, y, width, height."""
    x_a = max(box_a["x"], box_b["x"])
    y_a = max(box_a["y"], box_b["y"])
    x_b = min(box_a["x"] + box_a["width"], box_b["x"] + box_b["width"])
    y_b = min(box_a["y"] + box_a["height"], box_b["y"] + box_b["height"])

    inter_area = max(0, x_b - x_a) * max(0, y_b - y_a)
    area_a = box_a["width"] * box_a["height"]
    area_b = box_b["width"] * box_b["height"]

    union = area_a + area_b - inter_area
    if union <= 0:
        return 0.0
    return inter_area / union


def load_ground_truth(file_path):
    """Load evaluation JSON and separate elements by type.

    Returns:
        (figures, figure_captions, tables, table_captions, gt_all)
        where gt_all maps group -> {x, y, width, height} from *_all entries.
    """
    with open(file_path, encoding="utf-8") as f:
        data = json.load(f)

    figures = []
    figure_captions = []
    tables = []
    table_captions = []
    gt_all = {}

    for item in data:
        t = item["type"]
        if t == "figure":
            figures.append(item)
        elif t == "figure_caption":
            figure_captions.append(item)
        elif t == "table":
            tables.append(item)
        elif t == "table_caption":
            table_captions.append(item)
        elif t in ("figure_all", "table_all"):
            gt_all[item["group"]] = {
                "x": item["x"],
                "y": item["y"],
                "width": item["width"],
                "height": item["height"],
            }

    return figures, figure_captions, tables, table_captions, gt_all


def evaluate_single_trial(
    merger, figures, fig_captions, tables, tab_captions, gt_all, iou_threshold
):
    """Run one randomised trial of a merger and score against ground truth.

    Returns:
        dict with correct, total, fig_correct, fig_total, tab_correct, tab_total,
        and list of per-group IoU values.
    """
    figs = copy.deepcopy(figures)
    fcaps = copy.deepcopy(fig_captions)
    tabs = copy.deepcopy(tables)
    tcaps = copy.deepcopy(tab_captions)

    random.shuffle(figs)
    random.shuffle(fcaps)
    random.shuffle(tabs)
    random.shuffle(tcaps)

    merged_figs = merger.merge(figs, fcaps)
    merged_tabs = merger.merge(tabs, tcaps)

    fig_correct = 0
    tab_correct = 0
    ious = []

    for parent in merged_figs:
        group = parent["group"]
        if group in gt_all:
            iou = compute_iou(parent, gt_all[group])
            ious.append(iou)
            if iou >= iou_threshold:
                fig_correct += 1

    for parent in merged_tabs:
        group = parent["group"]
        if group in gt_all:
            iou = compute_iou(parent, gt_all[group])
            ious.append(iou)
            if iou >= iou_threshold:
                tab_correct += 1

    fig_total = len(merged_figs)
    tab_total = len(merged_tabs)

    return {
        "correct": fig_correct + tab_correct,
        "total": fig_total + tab_total,
        "fig_correct": fig_correct,
        "fig_total": fig_total,
        "tab_correct": tab_correct,
        "tab_total": tab_total,
        "ious": ious,
    }


def evaluate_file(merger, file_path, num_trials, iou_threshold):
    """Evaluate a single ground truth file over multiple random trials."""
    figures, fig_captions, tables, tab_captions, gt_all = load_ground_truth(file_path)

    total_groups = len(gt_all)
    fig_total = len(figures)
    tab_total = len(tables)

    if total_groups == 0:
        return {
            "file": file_path.name,
            "total_groups": 0,
            "figure_groups": 0,
            "table_groups": 0,
            "mean_accuracy": None,
            "std_accuracy": None,
            "mean_iou": None,
            "min_trial_accuracy": None,
            "max_trial_accuracy": None,
        }

    trial_accuracies = []
    trial_ious = []
    fig_correct_sum = 0
    tab_correct_sum = 0

    for _ in range(num_trials):
        result = evaluate_single_trial(
            merger, figures, fig_captions, tables, tab_captions, gt_all, iou_threshold
        )
        acc = result["correct"] / result["total"] if result["total"] > 0 else 0.0
        trial_accuracies.append(acc)
        trial_ious.extend(result["ious"])
        fig_correct_sum += result["fig_correct"]
        tab_correct_sum += result["tab_correct"]

    mean_acc = statistics.mean(trial_accuracies)
    std_acc = statistics.pstdev(trial_accuracies)
    mean_iou = statistics.mean(trial_ious) if trial_ious else 0.0

    return {
        "file": file_path.name,
        "total_groups": total_groups,
        "figure_groups": fig_total,
        "table_groups": tab_total,
        "mean_accuracy": round(mean_acc, 4),
        "std_accuracy": round(std_acc, 4),
        "mean_iou": round(mean_iou, 4),
        "min_trial_accuracy": round(min(trial_accuracies), 4),
        "max_trial_accuracy": round(max(trial_accuracies), 4),
        "fig_accuracy": round(fig_correct_sum / (fig_total * num_trials), 4)
        if fig_total > 0
        else None,
        "tab_accuracy": round(tab_correct_sum / (tab_total * num_trials), 4)
        if tab_total > 0
        else None,
    }


def evaluate_strategy(merger, strategy_name, json_files, num_trials, iou_threshold):
    """Evaluate a single merger strategy across all files. Returns aggregate dict."""
    per_file_results = []
    total_groups = 0
    total_fig_groups = 0
    total_tab_groups = 0
    weighted_acc_sum = 0.0
    all_fig_correct = 0
    all_fig_total = 0
    all_tab_correct = 0
    all_tab_total = 0
    worst_file = None
    worst_acc = 1.0

    for json_file in json_files:
        result = evaluate_file(merger, json_file, num_trials, iou_threshold)
        per_file_results.append(result)

        n = result["total_groups"]
        if n == 0:
            print(f"    {result['file']}: no groups (skipped)")
            continue

        total_groups += n
        total_fig_groups += result["figure_groups"]
        total_tab_groups += result["table_groups"]
        weighted_acc_sum += result["mean_accuracy"] * n

        if result["fig_accuracy"] is not None:
            all_fig_correct += result["fig_accuracy"] * result["figure_groups"]
            all_fig_total += result["figure_groups"]
        if result["tab_accuracy"] is not None:
            all_tab_correct += result["tab_accuracy"] * result["table_groups"]
            all_tab_total += result["table_groups"]

        if result["min_trial_accuracy"] < worst_acc:
            worst_acc = result["min_trial_accuracy"]
            worst_file = result["file"]

        print(
            f"    {result['file']}: {n} groups | "
            f"accuracy: {result['mean_accuracy']:.4f} +/- {result['std_accuracy']:.4f} | "
            f"mean IoU: {result['mean_iou']:.4f}"
        )

    if total_groups == 0:
        return {
            "strategy": strategy_name,
            "aggregate": None,
            "per_file": per_file_results,
        }

    overall_acc = weighted_acc_sum / total_groups
    fig_acc = all_fig_correct / all_fig_total if all_fig_total > 0 else None
    tab_acc = all_tab_correct / all_tab_total if all_tab_total > 0 else None

    all_mean_ious = [
        r["mean_iou"] for r in per_file_results if r["mean_iou"] is not None
    ]
    overall_iou = statistics.mean(all_mean_ious) if all_mean_ious else 0.0

    aggregate = {
        "weighted_accuracy": round(overall_acc, 4),
        "mean_iou": round(overall_iou, 4),
        "total_groups": total_groups,
        "figure_groups": total_fig_groups,
        "table_groups": total_tab_groups,
        "figure_accuracy": round(fig_acc, 4) if fig_acc is not None else None,
        "table_accuracy": round(tab_acc, 4) if tab_acc is not None else None,
        "worst_file": worst_file,
        "worst_trial_accuracy": round(worst_acc, 4),
    }

    print(f"  --- {strategy_name} summary ---")
    print(
        f"    Groups: {total_groups} ({total_fig_groups} fig, {total_tab_groups} tab)"
    )
    print(f"    Weighted accuracy: {overall_acc:.4f}")
    print(f"    Mean IoU: {overall_iou:.4f}")
    if fig_acc is not None:
        print(f"    Figure accuracy: {fig_acc:.4f}")
    if tab_acc is not None:
        print(f"    Table accuracy: {tab_acc:.4f}")
    if worst_file:
        print(f"    Worst file (min trial): {worst_file} ({worst_acc:.4f})")

    return {
        "strategy": strategy_name,
        "aggregate": aggregate,
        "per_file": per_file_results,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate caption-to-figure/table linking accuracy of CaptionMerger strategies"
    )
    parser.add_argument(
        "eval_dir",
        nargs="?",
        default="./caption_merging_evaluation",
        help="Directory with ground truth JSON files (default: ./caption_merging_evaluation)",
    )
    parser.add_argument(
        "--output", "-o", required=True, help="Output JSON file for results"
    )
    parser.add_argument(
        "--strategy",
        "-s",
        choices=list(STRATEGIES.keys()) + ["all"],
        default="all",
        help="Merger strategy to evaluate, or 'all' to compare (default: all)",
    )
    parser.add_argument(
        "--max-distance",
        "-d",
        type=int,
        default=50,
        help="Max distance for ThresholdMerger (default: 50)",
    )
    parser.add_argument(
        "--trials",
        "-t",
        type=int,
        default=10,
        help="Number of random shuffle trials per file (default: 10)",
    )
    parser.add_argument(
        "--iou-threshold",
        type=float,
        default=0.5,
        help="IoU threshold for counting a match as correct (default: 0.5)",
    )
    parser.add_argument(
        "--seed", type=int, default=42, help="Random seed (default: 42)"
    )
    args = parser.parse_args()

    random.seed(args.seed)

    eval_dir = Path(args.eval_dir)
    json_files = sorted(eval_dir.glob("*.json"))

    if not json_files:
        print(f"No JSON files found in {eval_dir}")
        return

    # Determine which strategies to run
    if args.strategy == "all":
        strategy_names = list(STRATEGIES.keys())
    else:
        strategy_names = [args.strategy]

    print("Evaluating caption-to-figure/table linking accuracy")
    print(f"  Evaluation dir: {eval_dir}")
    print(f"  Strategies: {', '.join(strategy_names)}")
    print(f"  Trials per file: {args.trials}")
    print(f"  IoU threshold: {args.iou_threshold:.2f}")
    print(f"  Random seed: {args.seed}")
    print("=" * 60)

    all_results = {}

    for name in strategy_names:
        merger = STRATEGIES[name](args)
        print(f"\n  [{name.upper()}]")
        print("-" * 60)

        # Reset seed so each strategy gets the same shuffles
        random.seed(args.seed)

        result = evaluate_strategy(
            merger, name, json_files, args.trials, args.iou_threshold
        )
        all_results[name] = result

    # Print comparison table when evaluating multiple strategies
    if len(strategy_names) > 1:
        print("\n" + "=" * 60)
        print("COMPARISON")
        print(f"  {'Strategy':<15} {'Accuracy':>10} {'Mean IoU':>10} {'Worst':>10}")
        print(f"  {'-' * 15} {'-' * 10} {'-' * 10} {'-' * 10}")
        for name in strategy_names:
            agg = all_results[name].get("aggregate")
            if agg:
                print(
                    f"  {name:<15} {agg['weighted_accuracy']:>10.4f} "
                    f"{agg['mean_iou']:>10.4f} {agg['worst_trial_accuracy']:>10.4f}"
                )
            else:
                print(f"  {name:<15} {'N/A':>10} {'N/A':>10} {'N/A':>10}")

    output = {
        "config": {
            "strategies": strategy_names,
            "num_trials": args.trials,
            "iou_threshold": args.iou_threshold,
            "max_distance": args.max_distance,
            "seed": args.seed,
            "num_files": len(json_files),
        },
        "results": {
            name: {
                "aggregate": all_results[name]["aggregate"],
                "per_file": all_results[name]["per_file"],
            }
            for name in strategy_names
        },
    }

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2)

    print(f"\nResults saved to: {output_path}")


if __name__ == "__main__":
    main()
