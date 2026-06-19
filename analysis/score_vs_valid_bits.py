from argparse import ArgumentParser
from collections import defaultdict
import csv
import json
from itertools import combinations
import os
from pathlib import Path
import sys
import time

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
ANALYSIS_ROOT = Path(__file__).resolve().parent
if str(ANALYSIS_ROOT) not in sys.path:
    sys.path.insert(0, str(ANALYSIS_ROOT))

from dataset_loaders import DATASET_CHOICES, load_dataset, resolve_dataset, sample_dataset
from filter_loader import load_filter_bank
from iris import IrisClassifier
from pairwise_iris_analysis import precompute_codes
DEFAULT_OUTPUT_DIR = ANALYSIS_ROOT / "output" / "score_vs_valid_bits"


filters, _ = load_filter_bank(None)


def add_figure_metadata(fig, metadata):
    if not metadata:
        return
    text = " | ".join(f"{key}={value}" for key, value in metadata.items() if value is not None)
    if text:
        fig.text(0.01, 0.01, text, ha="left", va="bottom", fontsize=7, family="monospace", wrap=True)


def summarize(values):
    arr = np.asarray(values, dtype=np.float64)
    if arr.size == 0:
        return None
    return {
        "count": int(arr.size),
        "min": float(arr.min()),
        "p05": float(np.quantile(arr, 0.05)),
        "median": float(np.median(arr)),
        "mean": float(arr.mean()),
        "p95": float(np.quantile(arr, 0.95)),
        "max": float(arr.max()),
    }


def best_score_with_valid_bits(base_code, base_mask, candidate_codes, candidate_masks):
    diff = np.bitwise_xor(candidate_codes, base_code)
    combined_mask = np.bitwise_and(candidate_masks, base_mask)
    raw_valid_bits = np.sum(combined_mask, axis=1)
    valid_bits = raw_valid_bits.astype(np.float32)
    mismatch_bits = np.sum(np.bitwise_and(diff, combined_mask), axis=1).astype(np.float32)

    scores = np.full(candidate_codes.shape[0], 2.0, dtype=np.float64)
    valid_rows = valid_bits > 0
    scores[valid_rows] = mismatch_bits[valid_rows] / valid_bits[valid_rows]

    best_index = int(np.argmin(scores))
    return float(scores[best_index]), best_index, int(raw_valid_bits[best_index])


def write_csv(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "category",
        "score",
        "valid_bits",
        "best_offset",
        "direction",
        "label_a",
        "label_b",
        "image_a",
        "image_b",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def bin_rows(rows, categories, bin_size):
    if not rows:
        return []
    max_valid_bits = max(row["valid_bits"] for row in rows)
    bins = []
    start = 0
    while start <= max_valid_bits:
        end = start + bin_size - 1
        record = {"valid_bits_start": int(start), "valid_bits_end": int(end)}
        for category in categories:
            category_rows = [
                row for row in rows if row["category"] == category and start <= row["valid_bits"] <= end
            ]
            record[category] = {
                "score_summary": summarize([row["score"] for row in category_rows]),
                "count": len(category_rows),
            }
        bins.append(record)
        start += bin_size
    return bins


def plot_rows(rows, output_path, metadata):
    category_rows = {
        "mated": [row for row in rows if row["category"] == "mated"],
        "non_mated": [row for row in rows if row["category"] == "non_mated"],
    }

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    colors = {"mated": "#3157ff", "non_mated": "#ff4040"}
    labels = {"mated": "Mated", "non_mated": "Non-mated"}

    for category, category_label in labels.items():
        rows_for_category = category_rows[category]
        axes[0].scatter(
            [row["score"] for row in rows_for_category],
            [row["valid_bits"] for row in rows_for_category],
            s=12,
            alpha=0.22,
            color=colors[category],
            label=category_label,
        )
    axes[0].set_title("Score vs Valid Bit Count")
    axes[0].set_xlabel("Hamming distance")
    axes[0].set_ylabel("Valid bit count")
    axes[0].legend(loc="best")

    for category, category_label in labels.items():
        valid_bits = np.asarray([row["valid_bits"] for row in category_rows[category]], dtype=np.float64)
        if valid_bits.size:
            axes[1].hist(valid_bits, bins=30, alpha=0.45, color=colors[category], label=category_label)
    axes[1].set_title("Valid Bit Count Distribution")
    axes[1].set_xlabel("Valid bit count")
    axes[1].set_ylabel("Pair count")
    axes[1].legend(loc="best")

    add_figure_metadata(fig, metadata)
    fig.tight_layout(rect=(0, 0.06, 1, 1))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def build_rows(labels, image_names, base_codes, base_masks, rotated_codes, rotated_masks, offsets):
    rows = []
    pair_count = len(labels) * (len(labels) - 1) // 2
    started = time.perf_counter()

    for pair_index, (idx_a, idx_b) in enumerate(combinations(range(len(labels)), 2), start=1):
        score_ab, offset_index_ab, valid_bits_ab = best_score_with_valid_bits(
            base_codes[idx_a],
            base_masks[idx_a],
            rotated_codes[idx_b],
            rotated_masks[idx_b],
        )
        score_ba, offset_index_ba, valid_bits_ba = best_score_with_valid_bits(
            base_codes[idx_b],
            base_masks[idx_b],
            rotated_codes[idx_a],
            rotated_masks[idx_a],
        )

        if score_ab <= score_ba:
            score = score_ab
            best_offset = int(offsets[offset_index_ab])
            direction = "a_vs_b"
            valid_bits = valid_bits_ab
        else:
            score = score_ba
            best_offset = int(offsets[offset_index_ba])
            direction = "b_vs_a"
            valid_bits = valid_bits_ba

        rows.append(
            {
                "category": "mated" if labels[idx_a] == labels[idx_b] else "non_mated",
                "score": float(score),
                "valid_bits": int(valid_bits),
                "best_offset": best_offset,
                "direction": direction,
                "label_a": str(labels[idx_a]),
                "label_b": str(labels[idx_b]),
                "image_a": str(image_names[idx_a]),
                "image_b": str(image_names[idx_b]),
            }
        )

        if pair_index == 1 or pair_index % 5000 == 0 or pair_index == pair_count:
            elapsed = time.perf_counter() - started
            print(f"Scored pairs: {pair_index}/{pair_count} in {elapsed:.1f}s")

    return rows


def parse_args():
    parser = ArgumentParser(
        description="Plot iriscode Hamming distance against valid-bit count for any supported dataset."
    )
    parser.add_argument("--dataset", default="casia-v3-twins", choices=DATASET_CHOICES, help="Dataset format.")
    parser.add_argument("--dataset-path", default=None, help="Override dataset image root.")
    parser.add_argument("--rotation", type=int, default=21, help="Number of horizontal offsets to evaluate.")
    parser.add_argument("--max-id", dest="max_identities", type=int, default=100)
    parser.add_argument("--max-img-per-id", dest="max_images_per_identity", type=int, default=5)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--bin-size", type=int, default=128)
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument(
        "--output-name",
        default="score_vs_valid_bits",
        help="Base output name for the PNG, CSV, and JSON files.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    if args.rotation < 1:
        raise ValueError("--rotation must be at least 1")
    if args.max_identities is not None and args.max_identities < 1:
        raise ValueError("--max-id must be at least 1")
    if args.max_images_per_identity is not None and args.max_images_per_identity < 1:
        raise ValueError("--max-img-per-id must be at least 1")
    if args.max_samples is not None and args.max_samples < 2:
        raise ValueError("--max-samples must be at least 2")
    if args.bin_size < 1:
        raise ValueError("--bin-size must be at least 1")

    dataset_path, dataset_format = resolve_dataset(args.dataset_path, args.dataset)
    images, labels, image_names = load_dataset(dataset_path, dataset_format)
    images, labels, image_names = sample_dataset(
        images,
        labels,
        image_names,
        max_samples=args.max_samples,
        max_identities=args.max_identities,
        max_images_per_identity=args.max_images_per_identity,
        seed=args.seed,
    )
    if len(images) < 2:
        raise RuntimeError("Need at least two sampled images.")

    print(f"Filters in use: {len(filters)}")
    classifier = IrisClassifier(filters)
    (
        base_codes,
        base_masks,
        rotated_codes,
        rotated_masks,
        offsets,
        labels,
        image_names,
        skipped,
    ) = precompute_codes(images, labels, image_names, classifier, args.rotation)

    rows = build_rows(labels, image_names, base_codes, base_masks, rotated_codes, rotated_masks, offsets)

    output_name = Path(args.output_name).stem
    output_dir = Path(args.output_dir).expanduser().resolve() / dataset_format
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / f"{output_name}.csv"
    plot_path = output_dir / f"{output_name}.png"
    summary_path = output_dir / f"{output_name}.json"

    write_csv(csv_path, rows)
    metadata = {
        "dataset": dataset_format,
        "dataset_path": str(dataset_path),
        "output_name": output_name,
        "seg_path": os.environ.get("SEG_PATH"),
        "rotation": args.rotation,
        "samples": len(labels),
        "max_identities": args.max_identities,
        "max_images_per_identity": args.max_images_per_identity,
        "max_samples": args.max_samples,
        "seed": args.seed,
        "skipped": len(skipped),
        "filter_count": len(filters),
    }
    plot_rows(rows, plot_path, metadata)

    categories = ["mated", "non_mated"]
    label_counts = defaultdict(int)
    for label in labels:
        label_counts[str(label)] += 1
    summary = {
        **metadata,
        "label_count": len(label_counts),
        "pair_count": len(rows),
        "categories": {
            category: {
                "score_summary": summarize([row["score"] for row in rows if row["category"] == category]),
                "valid_bits_summary": summarize([row["valid_bits"] for row in rows if row["category"] == category]),
                "count": sum(1 for row in rows if row["category"] == category),
            }
            for category in categories
        },
        "valid_bit_bins": bin_rows(rows, categories, args.bin_size),
        "skipped": skipped,
        "csv_path": str(csv_path),
        "plot_path": str(plot_path),
    }
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"Saved CSV to {csv_path}")
    print(f"Saved plot to {plot_path}")
    print(f"Saved summary to {summary_path}")


if __name__ == "__main__":
    main()
