from argparse import ArgumentParser
import os
from pathlib import Path
import sys
import time

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
ANALYSIS_ROOT = Path(__file__).resolve().parent
if str(ANALYSIS_ROOT) not in sys.path:
    sys.path.insert(0, str(ANALYSIS_ROOT))
DEFAULT_OUTPUT_DIR = Path(__file__).resolve().parent / "output" / "rotation_EER"
MATPLOTLIB_CONFIG_DIR = DEFAULT_OUTPUT_DIR / "matplotlib"
MATPLOTLIB_CONFIG_DIR.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(MATPLOTLIB_CONFIG_DIR))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

from dataset_loaders import DATASET_CHOICES, dataset_output_slug, load_dataset, resolve_dataset, sample_dataset
from filter_loader import load_filter_bank
from iris import IrisClassifier, get_iris_band, hamming_distances
from pairwise_iris_analysis import evaluate_scores, summarize_label_pairs

def add_figure_metadata(fig, metadata):
    if not metadata:
        return
    text = " | ".join(f"{key}={value}" for key, value in metadata.items() if value is not None)
    if text:
        fig.text(0.01, 0.01, text, ha="left", va="bottom", fontsize=7, family="monospace", wrap=True)


def load_rotation_consistency_helpers():
    try:
        from rotation_part_scoring import evaluate_eer, select_parts, split_code_slices
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "rotation_part_scoring.py is required for rotation_EER.py."
        ) from exc
    return evaluate_eer, select_parts, split_code_slices


def shift_band_vertically(iris_band, iris_mask, offset):
    offset = int(offset)
    if offset == 0:
        return iris_band.copy(), iris_mask.copy()

    shifted_band = np.zeros_like(iris_band)
    shifted_mask = np.zeros_like(iris_mask)
    if offset > 0:
        shifted_band[offset:, :] = iris_band[:-offset, :]
        shifted_mask[offset:, :] = iris_mask[:-offset, :]
    else:
        offset = -offset
        shifted_band[:-offset, :] = iris_band[offset:, :]
        shifted_mask[:-offset, :] = iris_mask[offset:, :]
    return shifted_band, shifted_mask


def segment_samples(images, image_names):
    segmented = []
    sample_count = len(images)
    for index, image in enumerate(images, start=1):
        if index == 1 or index % 100 == 0 or index == sample_count:
            print(f"Segmenting images: {index}/{sample_count}")
        iris_band, iris_mask = get_iris_band(image)
        if iris_band is None or iris_mask is None:
            raise RuntimeError(
                f"Segmentation failed for sample index {index - 1}: {image_names[index - 1]}"
            )
        segmented.append((iris_band, iris_mask))
    return segmented


def precompute_horizontal_candidates(segmented_samples, classifier, max_offset):
    band_shape = segmented_samples[0][0].shape
    offsets = np.arange(-max_offset, max_offset + 1, dtype=np.int16)
    order = np.argsort(np.abs(offsets), kind="stable")
    ordered_offsets = offsets[order]
    range_values = np.arange(0, max_offset + 1, dtype=np.int16)
    range_end_indices = np.array(
        [int(np.max(np.flatnonzero(np.abs(ordered_offsets) <= value))) for value in range_values],
        dtype=np.int32,
    )

    base_codes = []
    base_masks = []
    candidate_codes = []
    candidate_masks = []

    sample_count = len(segmented_samples)
    for index, (iris_band, iris_mask) in enumerate(segmented_samples, start=1):
        if index == 1 or index % 100 == 0 or index == sample_count:
            print(f"Encoding horizontal candidates: {index}/{sample_count}")
        base_code, base_mask, _ = classifier.get_iris_code(iris_band, iris_mask, offset=0)
        codes, masks, _ = classifier.get_iris_codes(iris_band, iris_mask, offsets=offsets)

        base_codes.append(np.asarray(base_code, dtype=bool))
        base_masks.append(np.asarray(base_mask, dtype=bool))
        candidate_codes.append(np.asarray(codes[order], dtype=bool))
        candidate_masks.append(np.asarray(masks[order], dtype=bool))

    return {
        "axis": "horizontal",
        "band_shape": tuple(int(v) for v in band_shape),
        "range_values": range_values,
        "range_end_indices": range_end_indices,
        "base_codes": np.stack(base_codes, axis=0),
        "base_masks": np.stack(base_masks, axis=0),
        "candidate_codes": np.stack(candidate_codes, axis=0),
        "candidate_masks": np.stack(candidate_masks, axis=0),
        "ordered_offsets": ordered_offsets,
        "ordered_vertical_offsets": np.zeros_like(ordered_offsets),
    }


def precompute_vertical_candidates(segmented_samples, classifier, horizontal_range, max_vertical_offset):
    band_shape = segmented_samples[0][0].shape
    horizontal_offsets = np.arange(-horizontal_range, horizontal_range + 1, dtype=np.int16)
    vertical_offsets = np.arange(-max_vertical_offset, max_vertical_offset + 1, dtype=np.int16)
    vertical_order = np.argsort(np.abs(vertical_offsets), kind="stable")
    ordered_vertical_offsets = vertical_offsets[vertical_order]
    range_values = np.arange(0, max_vertical_offset + 1, dtype=np.int16)

    ordered_horizontal_offsets = []
    ordered_vertical_offsets_per_candidate = []
    for vertical_offset in ordered_vertical_offsets:
        for horizontal_offset in horizontal_offsets:
            ordered_horizontal_offsets.append(int(horizontal_offset))
            ordered_vertical_offsets_per_candidate.append(int(vertical_offset))
    ordered_horizontal_offsets = np.asarray(ordered_horizontal_offsets, dtype=np.int16)
    ordered_vertical_offsets_per_candidate = np.asarray(
        ordered_vertical_offsets_per_candidate, dtype=np.int16
    )
    range_end_indices = np.array(
        [
            int(np.max(np.flatnonzero(np.abs(ordered_vertical_offsets_per_candidate) <= value)))
            for value in range_values
        ],
        dtype=np.int32,
    )

    base_codes = []
    base_masks = []
    candidate_codes = []
    candidate_masks = []

    sample_count = len(segmented_samples)
    for index, (iris_band, iris_mask) in enumerate(segmented_samples, start=1):
        if index == 1 or index % 100 == 0 or index == sample_count:
            print(f"Encoding vertical candidates: {index}/{sample_count}")
        base_code, base_mask, _ = classifier.get_iris_code(iris_band, iris_mask, offset=0)
        base_codes.append(np.asarray(base_code, dtype=bool))
        base_masks.append(np.asarray(base_mask, dtype=bool))

        image_candidate_codes = []
        image_candidate_masks = []
        for vertical_offset in ordered_vertical_offsets:
            shifted_band, shifted_mask = shift_band_vertically(iris_band, iris_mask, int(vertical_offset))
            codes, masks, _ = classifier.get_iris_codes(
                shifted_band,
                shifted_mask,
                offsets=horizontal_offsets,
            )
            image_candidate_codes.append(np.asarray(codes, dtype=bool))
            image_candidate_masks.append(np.asarray(masks, dtype=bool))

        candidate_codes.append(np.concatenate(image_candidate_codes, axis=0))
        candidate_masks.append(np.concatenate(image_candidate_masks, axis=0))

    return {
        "axis": "vertical",
        "band_shape": tuple(int(v) for v in band_shape),
        "range_values": range_values,
        "range_end_indices": range_end_indices,
        "base_codes": np.stack(base_codes, axis=0),
        "base_masks": np.stack(base_masks, axis=0),
        "candidate_codes": np.stack(candidate_codes, axis=0),
        "candidate_masks": np.stack(candidate_masks, axis=0),
        "ordered_offsets": ordered_horizontal_offsets,
        "ordered_vertical_offsets": ordered_vertical_offsets_per_candidate,
    }


def sweep_eer(labels, precomputed):
    pair_count = len(labels) * (len(labels) - 1) // 2
    score_lists = [[] for _ in precomputed["range_values"]]
    same_class_list = []
    started = time.perf_counter()

    base_codes = precomputed["base_codes"]
    base_masks = precomputed["base_masks"]
    candidate_codes = precomputed["candidate_codes"]
    candidate_masks = precomputed["candidate_masks"]
    range_end_indices = precomputed["range_end_indices"]

    pair_index = 0
    for idx1 in range(len(labels)):
        for idx2 in range(idx1 + 1, len(labels)):
            scores_12 = hamming_distances(
                candidate_codes[idx2],
                base_codes[idx1],
                candidate_masks[idx2],
                base_masks[idx1],
            )
            scores_21 = hamming_distances(
                candidate_codes[idx1],
                base_codes[idx2],
                candidate_masks[idx1],
                base_masks[idx2],
            )
            prefix_best = np.minimum(
                np.minimum.accumulate(scores_12),
                np.minimum.accumulate(scores_21),
            )
            for range_idx, end_index in enumerate(range_end_indices):
                score_lists[range_idx].append(float(prefix_best[end_index]))

            same_class_list.append(bool(labels[idx1] == labels[idx2]))
            pair_index += 1
            if pair_index == 1 or pair_index % 5000 == 0 or pair_index == pair_count:
                elapsed = time.perf_counter() - started
                print(f"Sweeping pairs: {pair_index}/{pair_count} in {elapsed:.1f}s")

    same_class = np.asarray(same_class_list, dtype=bool)
    curve = []
    for range_value, scores in zip(precomputed["range_values"], score_lists):
        evaluation = evaluate_scores(same_class, np.asarray(scores, dtype=np.float32))
        curve.append(
            {
                "offset_range": int(range_value),
                "eer": float(evaluation["eer"]),
                "eer_percent": float(evaluation["eer"] * 100.0),
                "roc_auc": float(evaluation["roc_auc"]),
                "eer_threshold": float(evaluation["eer_threshold"]),
                "band_shape": [int(v) for v in precomputed["band_shape"]],
            }
        )
    return curve


def sweep_rotation_consistency_eer(
    labels,
    precomputed,
    parts,
    threshold,
    eliminate,
    tolerance_offset,
    min_valid_bits,
    score,
    match_parts,
):
    if precomputed["axis"] != "horizontal":
        raise ValueError("Rotation-consistency EER sweep is currently only supported for --axis horizontal")

    evaluate_rotation_consistency_eer, select_parts, split_code_slices = load_rotation_consistency_helpers()
    base_codes = precomputed["base_codes"]
    base_masks = precomputed["base_masks"]
    candidate_codes = precomputed["candidate_codes"]
    candidate_masks = precomputed["candidate_masks"]
    ordered_offsets = precomputed["ordered_offsets"]
    range_end_indices = precomputed["range_end_indices"]
    range_values = precomputed["range_values"]
    slices = split_code_slices(base_codes.shape[1], parts)
    rows_by_range = [[] for _ in range_values]
    pair_count = len(labels) * (len(labels) - 1) // 2
    started = time.perf_counter()

    pair_index = 0
    for idx1 in range(len(labels)):
        for idx2 in range(idx1 + 1, len(labels)):
            scores_12, offsets_12 = best_part_scores_by_range(
                base_codes[idx1],
                base_masks[idx1],
                candidate_codes[idx2],
                candidate_masks[idx2],
                ordered_offsets,
                slices,
                min_valid_bits,
                range_end_indices,
            )
            scores_21, offsets_21 = best_part_scores_by_range(
                base_codes[idx2],
                base_masks[idx2],
                candidate_codes[idx1],
                candidate_masks[idx1],
                ordered_offsets,
                slices,
                min_valid_bits,
                range_end_indices,
            )

            same_class = bool(labels[idx1] == labels[idx2])
            for range_index in range(len(range_values)):
                result_12 = select_parts(
                    scores_12[range_index],
                    offsets_12[range_index],
                    eliminate,
                    tolerance_offset,
                )
                result_21 = select_parts(
                    scores_21[range_index],
                    offsets_21[range_index],
                    eliminate,
                    tolerance_offset,
                )
                chosen = result_12 if result_12["avg_hd"] <= result_21["avg_hd"] else result_21
                if score == "match-rotation":
                    predicted_mated = bool(chosen["rotation_match_count"] >= match_parts)
                    prediction_mode = "rotation_match_count"
                    prediction_score = float(chosen["rotation_match_count"])
                else:
                    predicted_mated = bool(chosen["avg_hd"] <= threshold)
                    prediction_mode = "hd_threshold"
                    prediction_score = float(chosen["avg_hd"])
                rows_by_range[range_index].append(
                    {
                        "same_class": same_class,
                        "predicted_mated": predicted_mated,
                        "correct": bool(predicted_mated == same_class),
                        "prediction_mode": prediction_mode,
                        "prediction_score": prediction_score,
                        "avg_hd": float(chosen["avg_hd"]),
                        "rotation_match_count": int(chosen["rotation_match_count"]),
                    }
                )

            pair_index += 1
            if pair_index == 1 or pair_index % 5000 == 0 or pair_index == pair_count:
                elapsed = time.perf_counter() - started
                print(f"Rotation consistency sweeping pairs: {pair_index}/{pair_count} in {elapsed:.1f}s")

    curve = []
    for range_value, rows in zip(range_values, rows_by_range):
        evaluation = evaluate_rotation_consistency_eer(rows)
        eer_threshold = evaluation["eer_hd_threshold"]
        if eer_threshold is None:
            eer_threshold = evaluation["eer_match_parts_threshold"]
        curve.append(
            {
                "offset_range": int(range_value),
                "eer": float(evaluation["eer"]),
                "eer_percent": float(evaluation["eer"] * 100.0),
                "roc_auc": float(evaluation["roc_auc"]),
                "eer_threshold": None if eer_threshold is None else float(eer_threshold),
                "eer_hd_threshold": evaluation["eer_hd_threshold"],
                "eer_match_parts_threshold": evaluation["eer_match_parts_threshold"],
                "band_shape": [int(v) for v in precomputed["band_shape"]],
                "score": score,
                "parts": int(parts),
                "eliminate": None if tolerance_offset is not None else int(eliminate),
                "tolerance_offset": None if tolerance_offset is None else int(tolerance_offset),
                "match_parts": None if score == "hd" else int(match_parts),
            }
        )
    return curve


def best_part_scores_by_range(
    base_code,
    base_mask,
    candidate_codes,
    candidate_masks,
    offsets,
    slices,
    min_valid_bits,
    range_end_indices,
):
    part_scores = np.empty((len(range_end_indices), len(slices)), dtype=np.float64)
    part_offsets = np.empty((len(range_end_indices), len(slices)), dtype=np.int16)

    for part_index, code_slice in enumerate(slices):
        part_base_code = base_code[code_slice]
        part_base_mask = base_mask[code_slice]
        part_candidate_codes = candidate_codes[:, code_slice]
        part_candidate_masks = candidate_masks[:, code_slice]

        diff = np.bitwise_xor(part_candidate_codes, part_base_code)
        combined_mask = np.bitwise_and(part_candidate_masks, part_base_mask)
        valid_bits = np.sum(combined_mask, axis=1)
        mismatch_bits = np.sum(np.bitwise_and(diff, combined_mask), axis=1)

        scores = np.full(candidate_codes.shape[0], 2.0, dtype=np.float64)
        valid_rows = valid_bits >= min_valid_bits
        scores[valid_rows] = mismatch_bits[valid_rows] / valid_bits[valid_rows]

        for range_index, end_index in enumerate(range_end_indices):
            prefix_scores = scores[: end_index + 1]
            best_score = float(np.min(prefix_scores))
            tied_indices = np.flatnonzero(np.isclose(prefix_scores, best_score))
            if tied_indices.size > 1:
                best_index = int(tied_indices[np.argmin(np.abs(offsets[tied_indices]))])
            else:
                best_index = int(tied_indices[0])
            part_scores[range_index, part_index] = best_score
            part_offsets[range_index, part_index] = int(offsets[best_index])

    return part_scores, part_offsets


def plot_curve(curve_specs, axis_name, dataset_format, figure_path, metadata=None):
    if isinstance(curve_specs, list) and curve_specs and "curve" in curve_specs[0]:
        specs = curve_specs
    else:
        specs = [{"label": "Hamming distance", "curve": curve_specs}]

    figure, axis = plt.subplots(figsize=(9, 6))
    x_label = "Offset Range (pixels)"
    if axis_name == "horizontal":
        title = "Equal Error Rate across Horizontal Offset Ranges"
    else:
        title = "Equal Error Rate across Vertical Offset Ranges"

    for spec_index, spec in enumerate(specs):
        curve = spec["curve"]
        ranges = np.asarray([item["offset_range"] for item in curve], dtype=np.float64)
        eers = np.asarray([item["eer"] for item in curve], dtype=np.float64)
        (line,) = axis.plot(ranges, eers, lw=2, label=spec["label"])
        color = line.get_color()
        axis.scatter(ranges, eers, s=18, color=color)
        best_index = int(np.argmin(eers))
        best_range = ranges[best_index]
        best_eer = eers[best_index]
        axis.scatter(
            [best_range],
            [best_eer],
            s=80,
            marker="*",
            color=color,
            edgecolor="black",
            zorder=5,
        )
        axis.annotate(
            f"best={int(best_range)}\nEER={best_eer:.4f}",
            xy=(best_range, best_eer),
            xytext=(8 + 7 * (spec_index % 2), 10 + 14 * (spec_index % 3)),
            textcoords="offset points",
            fontsize=8,
            bbox={"boxstyle": "round,pad=0.25", "fc": "white", "ec": "0.6", "alpha": 0.85},
            arrowprops={"arrowstyle": "->", "color": "0.35", "lw": 0.8},
        )

    axis.set_title(title)
    axis.set_xlabel(x_label)
    axis.set_ylabel("Equal Error Rate")
    axis.ticklabel_format(axis="y", style="plain", useOffset=False)
    axis.xaxis.set_major_locator(MaxNLocator(nbins=12, integer=True))
    axis.tick_params(axis="x", labelsize=8)
    axis.set_ylim(bottom=0.0)
    axis.grid(True, which="both", alpha=0.3)
    if len(specs) > 1:
        axis.legend(loc="best")
    add_figure_metadata(figure, metadata or {})
    figure.tight_layout(rect=(0, 0.06, 1, 1))

    output = Path(figure_path).expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=300, bbox_inches="tight")
    plt.close(figure)
    return output


def main():
    parser = ArgumentParser(
        description="Sweep rotation-consistency EER across horizontal offset ranges.",
        allow_abbrev=False,
    )
    parser.add_argument(
        "--axis",
        choices=["horizontal", "vertical"],
        required=True,
        help="Which offset family to sweep.",
    )
    parser.add_argument(
        "--dataset",
        dest="dataset_format",
        default="casia-v3-lamp",
        choices=DATASET_CHOICES,
        help="Dataset layout.",
    )
    parser.add_argument("--max-id", dest="max_identities", type=int, default=None)
    parser.add_argument(
        "--max-img-per-id",
        dest="max_images_per_identity",
        type=int,
        default=2,
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--filters",
        dest="filters",
        default=None,
        help="Optional Python filters file containing a 'filters' list. Defaults to project filters.py.",
    )
    parser.add_argument(
        "--rotation",
        type=int,
        default=None,
        help="Rotation count to test. Example: 201 means offset range 100.",
    )
    parser.add_argument(
        "--max-offset-range",
        type=int,
        default=40,
        help="Maximum absolute offset range in pixels to test. Ignored when --rotation is provided.",
    )
    parser.add_argument(
        "--horizontal-range",
        type=int,
        default=21,
        help="When sweeping vertical offsets, keep horizontal search fixed to this absolute range.",
    )
    parser.add_argument(
        "--output-name",
        default=None,
        help="Base output name without extension. Defaults to <axis>_<dataset>.",
    )
    parser.add_argument(
        "--score",
        choices=["hd", "match-rotation"],
        default="hd",
        help="Rotation-consistency EER score to sweep: selected-part HD or matching rotation count.",
    )
    parser.add_argument(
        "--compare-methods",
        action="store_true",
        help=(
            "Plot normal whole-iriscode HD against the rotation-consistency parts average-HD method. "
            "Use with --score hd."
        ),
    )
    parser.add_argument("--parts", type=int, default=4, help="Parts for rotation-consistency EER.")
    parser.add_argument("--eliminate", type=int, default=0, help="Outlier parts to eliminate for rotation-consistency EER.")
    parser.add_argument(
        "--tolerance-offset",
        type=int,
        default=None,
        help=(
            "Keep only parts whose best rotation is within this offset from the anchor, "
            "and count rotations within this offset as matching. Overrides --eliminate for part selection."
        ),
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.3955,
        help="Fixed selected-part HD threshold for rotation-consistency HD prediction stats.",
    )
    parser.add_argument(
        "--match-parts",
        type=int,
        default=1,
        help=(
            "Fixed matching-parts threshold for match-rotation prediction stats. "
            "With --tolerance-offset, rotations within that offset count as matches."
        ),
    )
    args = parser.parse_args()

    if args.rotation is not None:
        if args.rotation < 1 or args.rotation % 2 == 0:
            raise ValueError("--rotation must be an odd positive integer, for example 201")
        args.max_offset_range = args.rotation // 2
    if args.max_offset_range < 0:
        raise ValueError("--max-offset-range must be non-negative")
    if args.horizontal_range < 0:
        raise ValueError("--horizontal-range must be non-negative")
    if args.axis != "horizontal":
        raise ValueError("rotation_EER.py with --score currently requires --axis horizontal")
    if args.compare_methods and args.score != "hd":
        raise ValueError("--compare-methods compares baseline HD against parts average-HD, so use --score hd")
    if args.parts < 1:
        raise ValueError("--parts must be at least 1")
    if args.eliminate < 0:
        raise ValueError("--eliminate cannot be negative")
    if args.eliminate >= args.parts:
        raise ValueError("--eliminate must be smaller than --parts")
    if args.match_parts < 1:
        raise ValueError("--match-parts must be at least 1")
    if args.tolerance_offset is None and args.match_parts > max(1, args.parts - args.eliminate):
        raise ValueError("--match-parts cannot be larger than the maximum kept parts after --eliminate")
    if args.match_parts > args.parts:
        raise ValueError("--match-parts cannot be larger than --parts")
    if args.tolerance_offset is not None and args.tolerance_offset < 0:
        raise ValueError("--tolerance-offset cannot be negative")
    min_valid_bits = 1

    dataset_path, dataset_format = resolve_dataset(None, args.dataset_format)
    dataset_name = dataset_output_slug(dataset_format)
    images, labels, image_names = load_dataset(dataset_path, dataset_format)
    images, labels, image_names = sample_dataset(
        images,
        labels,
        image_names,
        max_samples=None,
        max_identities=args.max_identities,
        max_images_per_identity=args.max_images_per_identity,
        seed=args.seed,
    )
    summary = summarize_label_pairs(labels)
    if summary["mated_pairs"] == 0 or summary["non_mated_pairs"] == 0:
        raise ValueError("Need both mated and non-mated pairs in the sampled subset.")

    selected_filters, filters_source = load_filter_bank(args.filters)
    print(f"Filters in use: {len(selected_filters)}")
    print(f"Filters source: {filters_source}")
    classifier = IrisClassifier(selected_filters)
    segmented_samples = segment_samples(images, image_names)
    if args.axis == "horizontal":
        precomputed = precompute_horizontal_candidates(segmented_samples, classifier, args.max_offset_range)
    else:
        precomputed = precompute_vertical_candidates(
            segmented_samples,
            classifier,
            args.horizontal_range,
            args.max_offset_range,
        )

    baseline_curve = None
    curve_specs = []
    if args.compare_methods:
        baseline_curve = sweep_eer(labels, precomputed)
        curve_specs.append({"label": "Baseline whole iriscode HD", "curve": baseline_curve})

    score = args.score
    curve = sweep_rotation_consistency_eer(
        labels,
        precomputed,
        args.parts,
        args.threshold,
        args.eliminate,
        args.tolerance_offset,
        min_valid_bits,
        score,
        args.match_parts,
    )
    part_selection = (
        f"tolerance={args.tolerance_offset}"
        if args.tolerance_offset is not None
        else f"eliminate={args.eliminate}"
    )
    label = f"Parts average HD, parts={args.parts}, {part_selection}"
    if score == "match-rotation":
        label = (
            f"Rotation consistency match-parts={args.match_parts}, "
            f"parts={args.parts}, {part_selection}"
        )
    curve_specs.append({"label": label, "curve": curve})
    best_points = {
        spec["label"]: min(spec["curve"], key=lambda item: item["eer"])
        for spec in curve_specs
    }

    output_dir = DEFAULT_OUTPUT_DIR.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    base_name = args.output_name or f"{args.axis}_{dataset_name}"
    figure_path = output_dir / f"{base_name}.png"

    plot_metadata = {
        "dataset": dataset_format,
        "seg_path": os.environ.get("SEG_PATH"),
        "axis": args.axis,
        "max_offset_range": args.max_offset_range,
        "rotation": args.rotation,
        "horizontal_range": args.horizontal_range if args.axis == "vertical" else None,
        "samples": summary["sample_count"],
        "classes": summary["class_count"],
        "mated_pairs": summary["mated_pairs"],
        "non_mated_pairs": summary["non_mated_pairs"],
        "max_identities": args.max_identities,
        "max_images_per_identity": args.max_images_per_identity,
        "seed": args.seed,
        "filters_source": filters_source,
        "filter_count": len(selected_filters),
        "score": args.score,
        "parts": args.parts,
        "eliminate": args.eliminate if args.tolerance_offset is None else None,
        "tolerance_offset": args.tolerance_offset,
        "match_parts": args.match_parts if args.score == "match-rotation" else None,
    }
    plot_curve(curve_specs, args.axis, dataset_format, figure_path, metadata=plot_metadata)

    for label, point in best_points.items():
        print(f"Best {label}: offset range {point['offset_range']} px")
        print(f"  EER: {point['eer']:.6f} ({point['eer_percent']:.4f}%)")
    print(f"Saved sweep figure to {figure_path}")


if __name__ == "__main__":
    main()
