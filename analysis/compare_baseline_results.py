from argparse import ArgumentParser
import json
from pathlib import Path


LOWER_IS_BETTER_METRICS = {
    "eer",
    "score",
}


def load_json(path):
    path = Path(path).expanduser().resolve()
    return path, json.loads(path.read_text())


def pairwise_by_dataset(results):
    return {item["dataset_format"]: item for item in results.get("pairwise", [])}


def relative_percent(old, new):
    if old == 0:
        return None
    return ((new - old) / old) * 100.0


def compare_metric(old, new, metric_name):
    result = {
        "old": old,
        "new": new,
        "delta": new - old,
        "relative_percent": relative_percent(old, new),
    }
    if metric_name in LOWER_IS_BETTER_METRICS:
        result["direction"] = "better" if new < old else "worse" if new > old else "same"
    else:
        result["direction"] = "better" if new > old else "worse" if new < old else "same"
    return result


def compare_pairwise(old_results, new_results):
    old_lookup = pairwise_by_dataset(old_results)
    new_lookup = pairwise_by_dataset(new_results)
    comparison = {}
    for dataset in sorted(set(old_lookup) | set(new_lookup)):
        old_item = old_lookup.get(dataset)
        new_item = new_lookup.get(dataset)
        if old_item is None or new_item is None:
            comparison[dataset] = {"status": "missing_in_one_result"}
            continue
        comparison[dataset] = {
            "roc_auc": compare_metric(old_item["roc_auc"], new_item["roc_auc"], "roc_auc"),
            "eer": compare_metric(old_item["eer"], new_item["eer"], "eer"),
        }
    return comparison


def print_pairwise(comparison):
    print("Pairwise summary")
    for dataset, metrics in comparison.items():
        print(f"- {dataset}")
        if "status" in metrics:
            print(f"  status: {metrics['status']}")
            continue
        for metric_name in ("roc_auc", "eer"):
            metric = metrics[metric_name]
            rel = metric["relative_percent"]
            rel_text = "n/a" if rel is None else f"{rel:+.2f}%"
            print(
                f"  {metric_name}: {metric['old']:.6f} -> {metric['new']:.6f} "
                f"({metric['direction']}, delta {metric['delta']:+.6f}, {rel_text})"
            )


def main():
    parser = ArgumentParser(description="Compare two pipeline baseline JSON result files.")
    parser.add_argument("old_results", help="Baseline JSON to compare from.")
    parser.add_argument("new_results", help="Baseline JSON to compare to.")
    parser.add_argument("--output", default=None, help="Optional JSON file for the structured comparison.")
    args = parser.parse_args()

    old_path, old_results = load_json(args.old_results)
    new_path, new_results = load_json(args.new_results)

    comparison = {
        "old_results": str(old_path),
        "new_results": str(new_path),
        "pairwise": compare_pairwise(old_results, new_results),
    }

    print_pairwise(comparison["pairwise"])

    if args.output:
        output_path = Path(args.output).expanduser().resolve()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(comparison, indent=2))
        print(f"\nSaved comparison JSON to {output_path}")


if __name__ == "__main__":
    main()
