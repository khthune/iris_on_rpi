from argparse import ArgumentParser
import json
import os
import re
from pathlib import Path


DEFAULT_OUTPUT_DIR = Path(__file__).resolve().parent / "output" / "compare_pipelines"
LOWER_IS_BETTER_METRICS = {
    "eer",
    "score",
}


def load_json(path):
    path = Path(path).expanduser().resolve()
    return path, json.loads(path.read_text())


def pairwise_by_dataset(results):
    return {item["dataset_format"]: item for item in results.get("pairwise", [])}


def safe_slug(text):
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", str(text)).strip("_") or "results"


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


def get_pairwise_hd_threshold(item):
    if "eer_hd_threshold" in item:
        return item["eer_hd_threshold"]
    if "eer_threshold" in item:
        return -item["eer_threshold"]
    return None


def get_rotation_consistency_summary(item):
    result = item.get("rotation_consistency_classifier")
    if not result:
        return None
    return result.get("summary", {})


def binomial_count_std(wrong, total):
    if wrong is None or total in (None, 0):
        return None
    p = float(wrong) / float(total)
    return (float(total) * p * (1.0 - p)) ** 0.5


def binomial_rate_std(rate, total):
    if rate is None or total in (None, 0):
        return None
    p = float(rate)
    return (p * (1.0 - p) / float(total)) ** 0.5


def fixed_threshold_std_fields(fixed_threshold):
    fixed_threshold = fixed_threshold or {}
    mated_wrong = fixed_threshold.get("mated_wrongly_classified_non_mated")
    non_mated_wrong = fixed_threshold.get("non_mated_wrongly_classified_mated")
    mated_total = fixed_threshold.get("mated_total")
    non_mated_total = fixed_threshold.get("non_mated_total")
    false_reject_rate = fixed_threshold.get("false_reject_rate")
    false_accept_rate = fixed_threshold.get("false_accept_rate")
    return {
        "mated_false_rejects_std": binomial_count_std(mated_wrong, mated_total),
        "non_mated_false_accepts_std": binomial_count_std(non_mated_wrong, non_mated_total),
        "mated_false_reject_rate_std": binomial_rate_std(false_reject_rate, mated_total),
        "false_accept_rate_std": binomial_rate_std(false_accept_rate, non_mated_total),
    }


def selected_pairwise_summary(item):
    rotation_summary = get_rotation_consistency_summary(item)
    if rotation_summary is not None:
        rotation_fixed = item.get("rotation_consistency_classifier", {}).get("fixed_threshold", {})
        return {
            "source": "rotation_consistency_classifier",
            "roc_auc": rotation_summary.get("roc_auc"),
            "eer": rotation_summary.get("eer"),
            "eer_std": rotation_summary.get("eer_std"),
            "accuracy": rotation_summary.get("accuracy"),
            "eer_hd_threshold": rotation_summary.get("eer_hd_threshold"),
            "eer_match_parts_threshold": rotation_summary.get("eer_match_parts_threshold"),
            "mated_false_rejects": rotation_fixed.get("mated_wrongly_classified_non_mated"),
            "non_mated_false_accepts": rotation_fixed.get("non_mated_wrongly_classified_mated"),
            "mated_false_reject_rate": rotation_fixed.get("false_reject_rate"),
            "false_accept_rate": rotation_fixed.get("false_accept_rate"),
            **fixed_threshold_std_fields(rotation_fixed),
        }
    fixed_threshold = item.get("fixed_threshold") or {}
    return {
        "source": "pairwise",
        "roc_auc": item.get("roc_auc"),
        "eer": item.get("eer"),
        "eer_std": item.get("eer_std"),
        "eer_hd_threshold": get_pairwise_hd_threshold(item),
        "mated_false_rejects": fixed_threshold.get("mated_wrongly_classified_non_mated"),
        "non_mated_false_accepts": fixed_threshold.get("non_mated_wrongly_classified_mated"),
        "mated_false_reject_rate": fixed_threshold.get("false_reject_rate"),
        "false_accept_rate": fixed_threshold.get("false_accept_rate"),
        **fixed_threshold_std_fields(fixed_threshold),
    }


def pairwise_summary(item):
    return {
        "roc_auc": item["roc_auc"],
        "eer": item["eer"],
        "eer_hd_threshold": get_pairwise_hd_threshold(item),
    }


def compare_score_summaries(old_summary, new_summary):
    comparison = {
        "roc_auc": compare_metric(old_summary["roc_auc"], new_summary["roc_auc"], "roc_auc"),
        "eer": compare_metric(old_summary["eer"], new_summary["eer"], "eer"),
    }
    if old_summary.get("eer_hd_threshold") is not None and new_summary.get("eer_hd_threshold") is not None:
        comparison["eer_hd_threshold"] = {
            "old": old_summary["eer_hd_threshold"],
            "new": new_summary["eer_hd_threshold"],
        }
    if "accuracy" in old_summary and "accuracy" in new_summary:
        comparison["accuracy"] = compare_metric(old_summary["accuracy"], new_summary["accuracy"], "accuracy")
    return comparison


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
        old_hd_threshold = get_pairwise_hd_threshold(old_item)
        new_hd_threshold = get_pairwise_hd_threshold(new_item)
        if old_hd_threshold is not None and new_hd_threshold is not None:
            comparison[dataset]["eer_hd_threshold"] = {
                "old": old_hd_threshold,
                "new": new_hd_threshold,
            }

        old_rotation = get_rotation_consistency_summary(old_item)
        new_rotation = get_rotation_consistency_summary(new_item)
        if new_rotation is not None:
            old_summary = old_rotation if old_rotation is not None else pairwise_summary(old_item)
            old_label = "rotation consistency classifier" if old_rotation is not None else "pairwise baseline"
            comparison[dataset]["rotation_consistency_classifier"] = {
                "old_source": old_label,
                "new_source": "rotation consistency classifier",
                **compare_score_summaries(old_summary, new_rotation),
            }
        elif old_rotation is not None:
            comparison[dataset]["rotation_consistency_classifier"] = {"status": "missing_in_new_result"}
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
        threshold = metrics.get("eer_hd_threshold")
        if threshold is not None:
            print(f"  EER HD threshold: {threshold['old']:.6f} -> {threshold['new']:.6f}")

        rotation = metrics.get("rotation_consistency_classifier")
        if rotation is not None:
            print("  rotation consistency classifier:")
            if "status" in rotation:
                print(f"    status: {rotation['status']}")
                continue
            print(f"    comparison: {rotation['old_source']} -> {rotation['new_source']}")
            for metric_name in ("roc_auc", "eer", "accuracy"):
                if metric_name not in rotation:
                    continue
                metric = rotation[metric_name]
                rel = metric["relative_percent"]
                rel_text = "n/a" if rel is None else f"{rel:+.2f}%"
                print(
                    f"    {metric_name}: {metric['old']:.6f} -> {metric['new']:.6f} "
                    f"({metric['direction']}, delta {metric['delta']:+.6f}, {rel_text})"
                )
            threshold = rotation.get("eer_hd_threshold")
            if threshold is not None:
                print(f"    EER HD threshold: {threshold['old']:.6f} -> {threshold['new']:.6f}")


def build_multi_pairwise(results_by_path):
    dataset_names = sorted(
        {
            dataset
            for _path, results in results_by_path
            for dataset in pairwise_by_dataset(results)
        }
    )
    entries = []
    for path, results in results_by_path:
        lookup = pairwise_by_dataset(results)
        result_entry = {
            "label": path.stem,
            "path": str(path),
            "datasets": {},
        }
        for dataset in dataset_names:
            item = lookup.get(dataset)
            if item is None:
                result_entry["datasets"][dataset] = {"status": "missing"}
            else:
                result_entry["datasets"][dataset] = selected_pairwise_summary(item)
        entries.append(result_entry)
    return {
        "datasets": dataset_names,
        "results": entries,
    }


def select_multi_pairwise_datasets(multi_pairwise, datasets):
    if not datasets:
        return multi_pairwise

    selected = []
    seen = set()
    for dataset in datasets:
        if dataset not in seen:
            selected.append(dataset)
            seen.add(dataset)

    available = set(multi_pairwise["datasets"])
    missing = [dataset for dataset in selected if dataset not in available]
    if missing:
        print(f"Plot dataset filter skipped missing datasets: {', '.join(missing)}")

    selected = [dataset for dataset in selected if dataset in available]
    return {
        "datasets": selected,
        "results": [
            {
                **entry,
                "datasets": {
                    dataset: entry["datasets"][dataset]
                    for dataset in selected
                },
            }
            for entry in multi_pairwise["results"]
        ],
    }


def print_multi_pairwise(multi_pairwise):
    print("Pairwise EER summary")
    labels = [entry["label"] for entry in multi_pairwise["results"]]
    label_width = max([len("dataset"), *[len(label) for label in labels]])
    print(" " * (label_width + 2) + "  ".join(labels))
    for dataset in multi_pairwise["datasets"]:
        values = []
        for entry in multi_pairwise["results"]:
            metrics = entry["datasets"][dataset]
            if "status" in metrics or metrics.get("eer") is None:
                values.append("missing")
            else:
                values.append(f"{float(metrics['eer']):.6f}")
        print(f"{dataset:<{label_width}}  " + "  ".join(values))


def shorten_text(text, max_length=100):
    text = str(text)
    if len(text) <= max_length:
        return text
    keep = max_length // 2 - 2
    return f"{text[:keep]}...{text[-keep:]}"


def split_dataset_label(label):
    parts = str(label).split("-")
    if len(parts) <= 2:
        return [label]
    midpoint = (len(parts) + 1) // 2
    return ["-".join(parts[:midpoint]), "-".join(parts[midpoint:])]


def normalize_plot_metric(metric):
    aliases = {
        "auc": "roc_auc",
        "roc-auc": "roc_auc",
        "roc_auc": "roc_auc",
        "eer": "eer",
        "accuracy": "accuracy",
        "mated-classified-non-mated": "mated_false_rejects",
        "mated_false_rejects": "mated_false_rejects",
        "non-mated-classified-mated": "non_mated_false_accepts",
        "non_mated_false_accepts": "non_mated_false_accepts",
        "mated-classified-non-mated-rate": "mated_false_reject_rate",
        "mated_false_reject_rate": "mated_false_reject_rate",
        "non-mated-classified-mated-rate": "false_accept_rate",
        "false_accept_rate": "false_accept_rate",
        "frr": "mated_false_reject_rate",
        "far": "false_accept_rate",
    }
    return aliases[metric]


def metric_display_name(metric):
    return {
        "eer": "EER",
        "roc_auc": "ROC AUC",
        "accuracy": "Accuracy",
        "mated_false_rejects": "Mated Classified as Non-Mated",
        "non_mated_false_accepts": "Non-Mated Classified as Mated",
        "mated_false_reject_rate": "Mated False Reject Rate",
        "false_accept_rate": "False Accept Rate",
    }[metric]


def metric_std_key(metric):
    return {
        "eer": "eer_std",
        "mated_false_rejects": "mated_false_rejects_std",
        "non_mated_false_accepts": "non_mated_false_accepts_std",
        "mated_false_reject_rate": "mated_false_reject_rate_std",
        "false_accept_rate": "false_accept_rate_std",
    }.get(metric)


def parse_dataset_filter(values):
    if not values:
        return None
    datasets = []
    for value in values:
        datasets.extend(part.strip() for part in str(value).split(",") if part.strip())
    return datasets or None


def plot_pairwise_metric(multi_pairwise, plot_path, metric, dataset_filter=None, show_error_bars=True):
    metric = normalize_plot_metric(metric)
    selected_datasets = set(dataset_filter) if dataset_filter is not None else None
    datasets = [
        dataset
        for dataset in multi_pairwise["datasets"]
        if selected_datasets is None or dataset in selected_datasets
        if any(
            "status" not in entry["datasets"][dataset] and entry["datasets"][dataset].get(metric) is not None
            for entry in multi_pairwise["results"]
        )
    ]

    if not datasets:
        return None

    plot_path.parent.mkdir(parents=True, exist_ok=True)
    matplotlib_config_dir = DEFAULT_OUTPUT_DIR / "matplotlib"
    matplotlib_config_dir.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(matplotlib_config_dir))

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    from matplotlib.ticker import MaxNLocator

    x_positions = np.arange(len(datasets), dtype=np.float64)
    result_count = len(multi_pairwise["results"])
    width = min(0.8 / max(result_count, 1), 0.24)
    offsets = (np.arange(result_count, dtype=np.float64) - (result_count - 1) / 2.0) * width
    figure_width = max(12.0, len(datasets) * max(1.8, result_count * 0.42))
    figure, axis = plt.subplots(figsize=(figure_width, 6.8))

    color_cycle = plt.get_cmap("tab10").colors
    finite_values = []
    for result_index, entry in enumerate(multi_pairwise["results"]):
        values = []
        std_values = []
        bar_labels = []
        std_key = metric_std_key(metric) if show_error_bars else None
        for dataset in datasets:
            metrics = entry["datasets"][dataset]
            if "status" in metrics or metrics.get(metric) is None:
                values.append(np.nan)
                std_values.append(0.0)
                bar_labels.append("")
            else:
                value = float(metrics[metric])
                std_value = 0.0
                if std_key is not None and metrics.get(std_key) is not None:
                    std_value = float(metrics[std_key])
                values.append(value)
                std_values.append(std_value)
                finite_values.append(value + std_value)
                if metric in {"mated_false_rejects", "non_mated_false_accepts"}:
                    bar_labels.append(f"{int(value)}")
                else:
                    bar_labels.append(f"{value:.4f}")
        bars = axis.bar(
            x_positions + offsets[result_index],
            values,
            width,
            label=entry["label"],
            color=color_cycle[result_index % len(color_cycle)],
            yerr=std_values if any(value > 0 for value in std_values) else None,
            capsize=3,
            error_kw={"elinewidth": 1.0, "capthick": 1.0, "alpha": 0.8},
        )
        axis.bar_label(bars, labels=bar_labels, padding=3, fontsize=7, rotation=90)

    axis.set_ylabel(metric_display_name(metric))
    axis.set_xticks(x_positions)
    axis.set_xticklabels(["\n".join(split_dataset_label(dataset)) for dataset in datasets], fontsize=8)
    axis.tick_params(axis="x", labelrotation=20)
    axis.yaxis.set_major_locator(MaxNLocator(nbins=8))
    axis.set_ylim(0.0, max(finite_values + [0.01]) * 1.35)
    axis.grid(True, axis="y", alpha=0.3)
    legend_columns = min(3, max(1, result_count))
    axis.legend(loc="upper left", frameon=False, ncol=legend_columns, fontsize=8)

    figure.tight_layout(rect=(0, 0.03, 1, 1))
    figure.savefig(plot_path, dpi=300, bbox_inches="tight")
    plt.close(figure)
    return plot_path


def main():
    parser = ArgumentParser(description="Compare two or more pipeline benchmark JSON result files.")
    parser.add_argument("results", nargs="+", help="Pipeline JSON result files to compare.")
    parser.add_argument("--output", default=None, help="Optional JSON file for the structured comparison.")
    parser.add_argument("--plot-output", default=None, help="Optional PNG path for the metric bar chart.")
    parser.add_argument(
        "--plot-metric",
        choices=[
            "eer",
            "roc_auc",
            "roc-auc",
            "auc",
            "accuracy",
            "mated-classified-non-mated",
            "non-mated-classified-mated",
            "mated-classified-non-mated-rate",
            "non-mated-classified-mated-rate",
            "frr",
            "far",
            "false_accept_rate",
        ],
        default="eer",
        help="Metric to plot. Defaults to eer.",
    )
    parser.add_argument(
        "--datasets",
        "--plot-datasets",
        nargs="+",
        default=None,
        help="Only include these dataset names in the metric plot. Accepts space-separated or comma-separated names.",
    )
    parser.add_argument(
        "--no-error-bars",
        action="store_true",
        help="Do not plot std/error bars on the metric chart.",
    )
    parser.add_argument("--no-plot", action="store_true", help="Do not save the metric bar chart.")
    args = parser.parse_args()

    if len(args.results) < 2:
        raise ValueError("Provide at least two pipeline JSON result files.")

    loaded_results = [load_json(path) for path in args.results]
    multi_pairwise = build_multi_pairwise(loaded_results)
    plot_datasets = parse_dataset_filter(args.datasets)

    comparison = {
        "result_files": [str(path) for path, _results in loaded_results],
        "pairwise_multi": multi_pairwise,
        "plot_datasets": plot_datasets,
    }

    if len(loaded_results) == 2:
        old_path, old_results = loaded_results[0]
        new_path, new_results = loaded_results[1]
        comparison.update(
            {
                "old_results": str(old_path),
                "new_results": str(new_path),
                "pairwise": compare_pairwise(old_results, new_results),
            }
        )

    if not args.no_plot:
        plot_metric = normalize_plot_metric(args.plot_metric)
        plot_multi_pairwise = select_multi_pairwise_datasets(multi_pairwise, plot_datasets)
        if args.plot_output:
            plot_path = Path(args.plot_output).expanduser().resolve()
        else:
            plot_name = "_vs_".join(safe_slug(path.stem) for path, _results in loaded_results)
            if len(plot_name) > 160:
                plot_name = f"{safe_slug(loaded_results[0][0].stem)}_vs_{len(loaded_results) - 1}_more"
            plot_name = f"{plot_name}_{plot_metric}.png"
            plot_path = DEFAULT_OUTPUT_DIR / plot_name
        saved_plot = plot_pairwise_metric(
            plot_multi_pairwise,
            plot_path,
            plot_metric,
            show_error_bars=not args.no_error_bars,
        )
        if saved_plot is not None:
            print(f"Saved metric plot to {saved_plot}")

    if args.output:
        output_path = Path(args.output).expanduser().resolve()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(comparison, indent=2))


if __name__ == "__main__":
    main()
