from __future__ import annotations

from argparse import ArgumentParser
import os
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

DEFAULT_OUTPUT_DIR = Path(__file__).resolve().parent / "output" / "rotation_consistency_analysis"
MATPLOTLIB_CONFIG_DIR = DEFAULT_OUTPUT_DIR / "matplotlib"
MATPLOTLIB_CONFIG_DIR.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(MATPLOTLIB_CONFIG_DIR))
os.environ.setdefault("XDG_CACHE_HOME", str(MATPLOTLIB_CONFIG_DIR))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import numpy as np

from dataset_loaders import DATASET_CHOICES, dataset_output_slug, load_dataset, resolve_dataset
from filter_loader import load_filter_bank
from iris import IrisClassifier, get_iris_band


def summarize_label_pairs(labels):
    unique_labels, counts = np.unique(labels, return_counts=True)
    total_pairs = len(labels) * (len(labels) - 1) // 2
    mated_pairs = int(sum(count * (count - 1) // 2 for count in counts))
    return {
        "sample_count": int(len(labels)),
        "class_count": int(len(unique_labels)),
        "total_pairs": int(total_pairs),
        "mated_pairs": mated_pairs,
        "non_mated_pairs": int(total_pairs - mated_pairs),
    }


def split_code_slices(code_length, parts):
    if parts < 1:
        raise ValueError("--parts must be at least 1")
    if parts > code_length:
        raise ValueError(f"--parts cannot be larger than iriscode length ({code_length})")

    boundaries = np.linspace(0, code_length, parts + 1, dtype=int)
    return [slice(int(boundaries[index]), int(boundaries[index + 1])) for index in range(parts)]


def pick_mated_and_non_mated_pairs(images, labels, image_names, seed, max_segmented=None):
    rng = np.random.default_rng(seed)
    order = rng.permutation(len(images))
    label_to_samples = {}
    skipped = []
    limit = len(order) if max_segmented is None else min(len(order), int(max_segmented))

    for raw_index in order[:limit]:
        index = int(raw_index)
        try:
            iris_band, iris_mask = get_iris_band(images[index])
        except Exception as exc:
            skipped.append((index, str(image_names[index]), str(exc)))
            continue
        if iris_band is None or iris_mask is None:
            skipped.append((index, str(image_names[index]), "segmentation returned None"))
            continue

        sample = {
            "idx": index,
            "image": images[index],
            "label": labels[index],
            "image_name": image_names[index],
            "band": iris_band,
            "mask": iris_mask,
        }
        label_to_samples.setdefault(labels[index], []).append(sample)

        labels_with_two = [label for label, samples in label_to_samples.items() if len(samples) >= 2]
        if labels_with_two and len(label_to_samples) >= 2:
            break

    labels_with_two = [label for label, samples in label_to_samples.items() if len(samples) >= 2]
    if not labels_with_two:
        raise RuntimeError("Could not find two successfully segmented images with the same label for a mated pair.")
    if len(label_to_samples) < 2:
        raise RuntimeError("Could not find successfully segmented images from at least two labels for a non-mated pair.")

    mated_label = labels_with_two[int(rng.integers(0, len(labels_with_two)))]
    mated_samples = label_to_samples[mated_label]
    mated_choice = rng.choice(len(mated_samples), size=2, replace=False)
    mated_left = mated_samples[int(mated_choice[0])]
    mated_right = mated_samples[int(mated_choice[1])]

    other_labels = [label for label in label_to_samples if label != mated_label]
    if len(other_labels) >= 2:
        non_labels = rng.choice(np.array(other_labels, dtype=object), size=2, replace=False)
        non_left = label_to_samples[non_labels[0]][int(rng.integers(0, len(label_to_samples[non_labels[0]])))]
        non_right = label_to_samples[non_labels[1]][int(rng.integers(0, len(label_to_samples[non_labels[1]])))]
    else:
        non_left = mated_left
        other_label = other_labels[0]
        non_right = label_to_samples[other_label][int(rng.integers(0, len(label_to_samples[other_label])))]

    return [
        {
            "comparison": "mated",
            "same_class": True,
            "left": mated_left,
            "right": mated_right,
        },
        {
            "comparison": "non_mated",
            "same_class": False,
            "left": non_left,
            "right": non_right,
        },
    ], skipped


def score_parts_over_offsets(base_code, base_mask, candidate_codes, candidate_masks, offsets, parts, min_valid_bits):
    slices = split_code_slices(base_code.shape[0], parts)
    rows = []
    for part_index, code_slice in enumerate(slices, start=1):
        part_base_code = base_code[code_slice]
        part_base_mask = base_mask[code_slice]
        part_candidate_codes = candidate_codes[:, code_slice]
        part_candidate_masks = candidate_masks[:, code_slice]

        diff = np.bitwise_xor(part_candidate_codes, part_base_code)
        combined_mask = np.bitwise_and(part_candidate_masks, part_base_mask)
        valid_bits = np.sum(combined_mask, axis=1)
        mismatch_bits = np.sum(np.bitwise_and(diff, combined_mask), axis=1)

        scores = np.full(candidate_codes.shape[0], np.nan, dtype=np.float64)
        valid = valid_bits >= min_valid_bits
        scores[valid] = mismatch_bits[valid] / valid_bits[valid]

        for offset, score, valid_count in zip(offsets, scores, valid_bits):
            rows.append(
                {
                    "part": part_index,
                    "offset": int(offset),
                    "hamming_distance": float(score) if np.isfinite(score) else None,
                    "valid_bits": int(valid_count),
                }
            )
    return rows


def score_pair_over_offsets(pair, classifier, offsets, parts, min_valid_bits):
    base_code, base_mask, _ = classifier.get_iris_code(pair["left"]["band"], pair["left"]["mask"], offset=0)
    rotated_codes, rotated_masks, _ = classifier.get_iris_codes(pair["right"]["band"], pair["right"]["mask"], offsets=offsets)
    rows = score_parts_over_offsets(
        np.asarray(base_code, dtype=bool),
        np.asarray(base_mask, dtype=bool),
        np.asarray(rotated_codes, dtype=bool),
        np.asarray(rotated_masks, dtype=bool),
        offsets,
        parts,
        min_valid_bits,
    )
    for row in rows:
        row["comparison"] = pair["comparison"]
        row["same_class"] = pair["same_class"]
    return rows


def best_part_row(part_rows):
    finite_rows = [
        row
        for row in part_rows
        if row["hamming_distance"] is not None and np.isfinite(row["hamming_distance"])
    ]
    if not finite_rows:
        return None
    return min(finite_rows, key=lambda row: (row["hamming_distance"], abs(row["offset"])))


def plot_comparison_offsets(output_path, comparison_rows, pairs, metadata):
    figure, axes = plt.subplots(2, 1, figsize=(12, 10), sharex=True, sharey=True)
    cmap = plt.get_cmap("tab10", int(metadata["parts"]))

    for axis, pair in zip(axes, pairs):
        rows = [row for row in comparison_rows if row["comparison"] == pair["comparison"]]
        for part in range(1, int(metadata["parts"]) + 1):
            part_rows = [row for row in rows if row["part"] == part]
            part_offsets = np.array([row["offset"] for row in part_rows], dtype=np.int32)
            scores = np.array(
                [
                    np.nan if row["hamming_distance"] is None else row["hamming_distance"]
                    for row in part_rows
                ],
                dtype=np.float64,
            )
            color = cmap(part - 1)
            axis.plot(part_offsets, scores, color=color, linewidth=1.4, alpha=0.45)
            best = best_part_row(part_rows)
            if best is None:
                label = f"Part {part}: no valid offset"
            else:
                label = f"Part {part}: {best['offset']} pixels"
            axis.scatter(part_offsets, scores, color=color, s=32, alpha=0.85, label=label)
            if best is not None:
                axis.scatter(
                    [best["offset"]],
                    [best["hamming_distance"]],
                    color=color,
                    edgecolor="black",
                    marker="*",
                    s=130,
                    linewidth=0.8,
                    zorder=5,
                )

        left = pair["left"]
        right = pair["right"]
        axis.set_title(
            (
                f"{pair['comparison'].replace('_', '-').upper()} comparison\n"
                f"{left['image_name']}  vs  {right['image_name']}"
            )
        )
        axis.set_ylabel("Hamming distance")
        axis.set_xlabel("Rotation (pixels)")
        axis.set_ylim(0.1, 0.7)
        axis.xaxis.set_major_locator(MaxNLocator(nbins=11, integer=True))
        axis.grid(True, alpha=0.3)

    axes[0].tick_params(axis="x", labelbottom=True)
    axes[0].legend(loc="best", ncols=2 if int(metadata["parts"]) > 6 else 1)

    figure.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(figure)


def main():
    parser = ArgumentParser(
        description="Plot per-part Hamming distance over rotation offsets for one mated and one non-mated pair."
    )
    parser.add_argument("--dataset", dest="dataset_format", default="auto", choices=DATASET_CHOICES)
    parser.add_argument("--parts", type=int, default=5, help="Number of iriscode parts to plot.")
    parser.add_argument(
        "--filters",
        dest="filters",
        default=None,
        help="Optional Python filters file containing a 'filters' list.",
    )
    parser.add_argument("--rotation", type=int, default=21, help="Number of offsets to evaluate around zero.")
    parser.add_argument("--min-valid-bits", type=int, default=1, help="Minimum valid bits required in a subpart/offset.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-name", default=None)
    parser.set_defaults(
        dataset_path=None,
        output_dir=str(DEFAULT_OUTPUT_DIR),
    )
    args = parser.parse_args()

    if args.rotation < 1:
        raise ValueError("--rotation must be at least 1")

    dataset_path, dataset_format = resolve_dataset(args.dataset_path, args.dataset_format)
    output_name = args.output_name or f"{dataset_output_slug(dataset_format)}_mated_nonmated_parts{args.parts}"
    output_dir = Path(args.output_dir).expanduser().resolve()
    figure_path = output_dir / f"{output_name}.png"

    print(f"Using dataset format: {dataset_format}")
    print(f"Using dataset path: {dataset_path}")

    images, labels, image_names = load_dataset(dataset_path, dataset_format)

    pre_summary = summarize_label_pairs(labels)
    print(f"Samples: {pre_summary['sample_count']}")
    print(f"Classes: {pre_summary['class_count']}")
    print(f"Mated pairs: {pre_summary['mated_pairs']}")
    print(f"Non-mated pairs: {pre_summary['non_mated_pairs']}")
    if pre_summary["sample_count"] < 2:
        raise ValueError("The dataset needs at least two images.")
    if pre_summary["mated_pairs"] == 0:
        raise ValueError("The dataset needs at least one mated pair.")
    if pre_summary["non_mated_pairs"] == 0:
        raise ValueError("The dataset needs at least one non-mated pair.")

    selected_filters, filters_source = load_filter_bank(args.filters)
    print(f"Filters in use: {len(selected_filters)}")
    print(f"Filters source: {filters_source}")
    classifier = IrisClassifier(selected_filters)

    pairs, skipped = pick_mated_and_non_mated_pairs(images, labels, image_names, seed=args.seed)
    if skipped:
        print(f"Skipped {len(skipped)} images while selecting display pairs.")
        for skipped_index, skipped_name, reason in skipped[:5]:
            print(f"  skipped[{skipped_index}] {skipped_name}: {reason}")

    offsets = np.arange(args.rotation) - args.rotation // 2
    rows = []
    for pair in pairs:
        rows.extend(score_pair_over_offsets(pair, classifier, offsets, args.parts, args.min_valid_bits))

    metadata = {
        "parts": args.parts,
    }

    plot_comparison_offsets(figure_path, rows, pairs, metadata)

    print("Selected mated comparison:")
    print(f"  {pairs[0]['left']['image_name']} label={pairs[0]['left']['label']}")
    print(f"  {pairs[0]['right']['image_name']} label={pairs[0]['right']['label']}")
    for part in range(1, args.parts + 1):
        best = best_part_row([row for row in rows if row["comparison"] == "mated" and row["part"] == part])
        if best is not None:
            print(f"  part {part}: best rotation={best['offset']} px HD={best['hamming_distance']:.6f}")
    print("Selected non-mated comparison:")
    print(f"  {pairs[1]['left']['image_name']} label={pairs[1]['left']['label']}")
    print(f"  {pairs[1]['right']['image_name']} label={pairs[1]['right']['label']}")
    for part in range(1, args.parts + 1):
        best = best_part_row([row for row in rows if row["comparison"] == "non_mated" and row["part"] == part])
        if best is not None:
            print(f"  part {part}: best rotation={best['offset']} px HD={best['hamming_distance']:.6f}")
    print(f"Saved figure to {figure_path}")


if __name__ == "__main__":
    main()
