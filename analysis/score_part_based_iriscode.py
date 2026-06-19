from argparse import ArgumentParser
import os
from pathlib import Path
import sys
import time

import numpy as np

ANALYSIS_ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = ANALYSIS_ROOT.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
DEFAULT_OUTPUT_DIR = ANALYSIS_ROOT / "output" / "score_part_based_iriscode"
DEFAULT_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
MATPLOTLIB_CONFIG_DIR = DEFAULT_OUTPUT_DIR / "matplotlib"
MATPLOTLIB_CONFIG_DIR.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(MATPLOTLIB_CONFIG_DIR))

from dataset_loaders import DATASET_CHOICES, dataset_output_slug, load_dataset, resolve_dataset, sample_dataset
from filter_loader import load_filter_bank
from iris import IrisClassifier
from pairwise_iris_analysis import precompute_codes, summarize_label_pairs


def parse_int_range(text):
    values = []
    for chunk in str(text).split(","):
        chunk = chunk.strip()
        if not chunk:
            continue
        if "-" in chunk:
            start_text, end_text = chunk.split("-", 1)
            start = int(start_text)
            end = int(end_text)
            if end < start:
                raise ValueError(f"Invalid descending range: {chunk}")
            values.extend(range(start, end + 1))
        else:
            values.append(int(chunk))
    values = sorted(set(values))
    if not values:
        raise ValueError("Range cannot be empty")
    return values


def safe_output_name(text):
    return Path(str(text)).stem.replace("/", "_").replace(" ", "_")


def load_rotation_consistency_classifier():
    try:
        from rotation_part_scoring import (
            compute_pairwise_rotation_classifier,
            evaluate_eer,
            summarize_predictions,
        )
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "rotation_part_scoring.py is required for score_part_based_iriscode.py."
        ) from exc
    return compute_pairwise_rotation_classifier, evaluate_eer, summarize_predictions


def plot_results(path, rows, score, match_parts_values):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    parts_values = sorted({row["parts"] for row in rows})
    group_keys = sorted({(row["eliminate"], row["match_parts"]) for row in rows})
    x_positions = np.arange(len(parts_values), dtype=np.float64)
    figure, axis = plt.subplots(figsize=(max(11, len(parts_values) * 0.7), 6.5))
    color_cycle = plt.rcParams["axes.prop_cycle"].by_key()["color"]

    for index, (group_value, match_parts) in enumerate(group_keys):
        series = []
        for parts in parts_values:
            match = next(
                (
                    row
                    for row in rows
                    if row["parts"] == parts
                    and row["eliminate"] == group_value
                    and row["match_parts"] == match_parts
                ),
                None,
            )
            series.append(np.nan if match is None else match["eer"])
        label = f"eliminate={group_value}"
        if match_parts is not None:
            label += f", match={match_parts}"
        axis.plot(
            x_positions,
            series,
            marker="o",
            linewidth=1.8,
            markersize=5,
            color=color_cycle[index % len(color_cycle)],
            label=label,
        )

    best_row = min(rows, key=lambda row: row["eer"])
    best_x = parts_values.index(best_row["parts"])
    axis.scatter(
        [best_x],
        [best_row["eer"]],
        marker="*",
        s=180,
        color="#111111",
        zorder=5,
        label=(
            f"best: p={best_row['parts']} e={best_row['eliminate']}"
            if best_row["match_parts"] is None
            else f"best: p={best_row['parts']} e={best_row['eliminate']} m={best_row['match_parts']}"
        ),
    )

    title = f"Iriscode Part Strategy EER ({score})"
    if match_parts_values is not None:
        title += f" | fixed match_parts={match_parts_values}"
    axis.set_title(title)
    axis.set_xlabel("parts")
    axis.set_ylabel("EER")
    if len(parts_values) <= 14:
        axis.set_xticks(x_positions)
        axis.set_xticklabels([str(value) for value in parts_values])
    else:
        tick_indices = np.unique(np.linspace(0, len(parts_values) - 1, 12, dtype=int))
        axis.set_xticks(x_positions[tick_indices])
        axis.set_xticklabels([str(parts_values[index]) for index in tick_indices])
    axis.tick_params(axis="x", labelrotation=45 if len(parts_values) > 14 else 0, labelsize=8)
    axis.set_ylim(bottom=0.0)
    axis.grid(True, alpha=0.25)
    axis.legend(loc="best", fontsize=8)

    figure.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(figure)


def rows_with_match_parts(classifier_rows, match_parts):
    rows = []
    for row in classifier_rows:
        updated = dict(row)
        predicted_mated = bool(row["rotation_match_count"] >= match_parts)
        updated["predicted_mated"] = predicted_mated
        updated["correct"] = bool(predicted_mated == row["same_class"])
        updated["prediction_score"] = float(row["rotation_match_count"])
        rows.append(updated)
    return rows


def evaluate_grid(
    labels,
    base_codes,
    base_masks,
    rotated_codes,
    rotated_masks,
    offsets,
    dataset_format,
    parts_values,
    eliminate_values,
    score,
    threshold,
    match_parts_values,
    tolerance_offset,
    min_valid_bits,
):
    compute_pairwise_rotation_classifier, evaluate_eer, summarize_predictions = (
        load_rotation_consistency_classifier()
    )
    rows = []
    for parts in parts_values:
        for eliminate in eliminate_values:
            kept_parts_max = parts if tolerance_offset is not None else max(1, parts - eliminate)
            if eliminate >= parts:
                print(f"Skipping parts={parts}, eliminate={eliminate}: eliminate must be smaller than parts")
                continue

            selection_text = (
                f"tolerance_offset={tolerance_offset}"
                if tolerance_offset is not None
                else f"eliminate={eliminate}"
            )
            print(f"Testing parts={parts}, {selection_text}")
            started = time.perf_counter()
            base_classifier_rows = compute_pairwise_rotation_classifier(
                labels,
                base_codes,
                base_masks,
                rotated_codes,
                rotated_masks,
                offsets,
                parts,
                threshold,
                eliminate,
                tolerance_offset,
                min_valid_bits,
                1 if score == "match-rotation" else None,
            )
            elapsed = time.perf_counter() - started
            current_match_parts_values = match_parts_values if score == "match-rotation" else [None]
            for match_parts in current_match_parts_values:
                if score == "match-rotation" and match_parts > kept_parts_max:
                    print(
                        f"  Skipping match_parts={match_parts}: larger than kept parts ({kept_parts_max})"
                    )
                    continue
                classifier_rows = (
                    rows_with_match_parts(base_classifier_rows, match_parts)
                    if score == "match-rotation"
                    else base_classifier_rows
                )
                summary = summarize_predictions(classifier_rows)
                summary.update(evaluate_eer(classifier_rows))
                row = {
                    "dataset": dataset_format,
                    "parts": int(parts),
                    "eliminate": int(eliminate),
                    "tolerance_offset": None if tolerance_offset is None else int(tolerance_offset),
                    "kept_parts_max": int(kept_parts_max),
                    "match_parts": None if score == "hd" else int(match_parts),
                    "score": score,
                    "eer": float(summary["eer"]),
                    "eer_hd_threshold": summary["eer_hd_threshold"],
                    "eer_match_parts_threshold": summary["eer_match_parts_threshold"],
                    "accuracy": float(summary["accuracy"]),
                    "false_accept_rate": float(summary["false_accept_rate"]),
                    "false_reject_rate": float(summary["false_reject_rate"]),
                    "total_pairs": int(summary["total_pairs"]),
                    "seconds": float(elapsed),
                }
                rows.append(row)
                print(
                    f"  match_parts={row['match_parts']} "
                    f"EER={row['eer']:.6f} ACC={row['accuracy']:.6f} in {elapsed:.1f}s"
                )
    return rows


def main():
    parser = ArgumentParser(
        description=(
            "Benchmark iriscode part strategy settings: number of parts, forced elimination, "
            "tolerance offset, and HD versus match-rotation scoring."
        ),
        allow_abbrev=False,
    )
    parser.add_argument("--dataset", dest="dataset_format", default="auto", choices=DATASET_CHOICES)
    parser.add_argument(
        "--filters",
        dest="filters",
        default=None,
        help="Optional Python filters file containing a 'filters' list.",
    )
    parser.add_argument("--rotation", type=int, default=21)
    parser.add_argument("--parts-range", dest="parts_range", required=True, help="Example: 3-8 or 3,5,7")
    parser.add_argument(
        "--eliminate-range",
        dest="eliminate_range",
        default="0",
        help="Example: 0-2 or 0,1,2",
    )
    parser.add_argument(
        "--tolerance-offset",
        type=int,
        default=None,
        help=(
            "Keep only parts whose best rotation is within this offset from the anchor, "
            "and count rotations within this offset as matching. Overrides --eliminate."
        ),
    )
    parser.add_argument(
        "--score",
        choices=["match-rotation", "hd"],
        default="hd",
        help="EER score to sweep. match-rotation uses how many kept parts match the anchor rotation.",
    )
    parser.add_argument(
        "--match-parts",
        dest="match_parts",
        default=None,
        help=(
            "Fixed prediction threshold range for match-rotation accuracy/FAR/FRR. Example: 1-4 or 2,3,4. "
            "EER still sweeps the match-count threshold."
        ),
    )
    parser.add_argument("--threshold", type=float, default=0.3955, help="Only used for fixed accuracy in hd mode.")
    parser.add_argument("--min-valid-bits", type=int, default=1)
    parser.add_argument("--max-id", dest="max_identities", type=int, default=100)
    parser.add_argument("--max-img-per-id", dest="max_images_per_identity", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-name", default=None)
    parser.set_defaults(
        dataset_path=None,
        max_samples=None,
        output_dir=str(DEFAULT_OUTPUT_DIR),
    )
    args = parser.parse_args()

    if args.rotation < 1:
        raise ValueError("--rotation must be at least 1")
    if args.min_valid_bits < 1:
        raise ValueError("--min-valid-bits must be at least 1")
    if args.max_identities is not None and args.max_identities < 1:
        raise ValueError("--max-id must be at least 1")
    if args.max_images_per_identity is not None and args.max_images_per_identity < 1:
        raise ValueError("--max-img-per-id must be at least 1")
    if args.max_samples is not None and args.max_samples < 1:
        raise ValueError("--max-samples must be at least 1")
    if args.tolerance_offset is not None and args.tolerance_offset < 0:
        raise ValueError("--tolerance-offset cannot be negative")
    if args.score == "hd" and args.match_parts is not None:
        raise ValueError("--match-parts only applies when --score match-rotation")

    parts_values = parse_int_range(args.parts_range)
    eliminate_values = [0] if args.tolerance_offset is not None else parse_int_range(args.eliminate_range)
    match_parts_values = parse_int_range(args.match_parts) if args.match_parts is not None else [1]
    if min(parts_values) < 1:
        raise ValueError("--parts-range values must be at least 1")
    if min(eliminate_values) < 0:
        raise ValueError("--eliminate-range values cannot be negative")
    if min(match_parts_values) < 1:
        raise ValueError("--match-parts values must be at least 1")

    dataset_path, dataset_format = resolve_dataset(args.dataset_path, args.dataset_format)
    output_name = args.output_name or (
        f"{dataset_output_slug(dataset_format)}_parts{parts_values[0]}-{parts_values[-1]}"
        f"_elim{eliminate_values[0]}-{eliminate_values[-1]}"
        f"{'' if args.tolerance_offset is None else f'_tol{args.tolerance_offset}'}"
        f"_match{match_parts_values[0]}-{match_parts_values[-1]}_{args.score}"
    )
    output_name = safe_output_name(output_name)
    output_dir = Path(args.output_dir).expanduser().resolve()
    figure_path = output_dir / f"{output_name}.png"

    print(f"Using dataset format: {dataset_format}")
    print(f"Using dataset path: {dataset_path}")
    print(f"Parts values: {parts_values}")
    print(f"Eliminate values: {eliminate_values}")
    if args.tolerance_offset is not None:
        print(f"Tolerance offset: {args.tolerance_offset} (--eliminate is ignored)")
    print(f"Score: {args.score}")
    if args.score == "match-rotation":
        print(f"Fixed match-parts thresholds for prediction stats: {match_parts_values}")

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
    pre_summary = summarize_label_pairs(labels)
    if pre_summary["mated_pairs"] == 0 or pre_summary["non_mated_pairs"] == 0:
        raise ValueError("The sampled subset needs both mated and non-mated pairs.")

    selected_filters, filters_source = load_filter_bank(args.filters)
    print(f"Filters in use: {len(selected_filters)}")
    print(f"Filters source: {filters_source}")
    classifier = IrisClassifier(selected_filters)

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
    if skipped:
        print(f"Skipped {len(skipped)} images due to segmentation failure.")

    summary = summarize_label_pairs(labels)
    print(f"Usable samples: {summary['sample_count']}")
    print(f"Usable mated pairs: {summary['mated_pairs']}")
    print(f"Usable non-mated pairs: {summary['non_mated_pairs']}")
    if summary["mated_pairs"] == 0 or summary["non_mated_pairs"] == 0:
        raise ValueError("After segmentation, the subset needs both mated and non-mated pairs.")

    result_rows = evaluate_grid(
        labels,
        base_codes,
        base_masks,
        rotated_codes,
        rotated_masks,
        offsets,
        dataset_format,
        parts_values,
        eliminate_values,
        args.score,
        args.threshold,
        match_parts_values,
        args.tolerance_offset,
        args.min_valid_bits,
    )
    if not result_rows:
        raise ValueError("No valid parts/eliminate combinations were evaluated.")

    plot_results(
        figure_path,
        result_rows,
        args.score,
        None if args.score == "hd" else match_parts_values,
    )

    best = min(result_rows, key=lambda row: row["eer"])
    print(
        "Best: "
        f"parts={best['parts']} eliminate={best['eliminate']} "
        f"EER={best['eer']:.6f}"
    )
    print(f"Saved plot to {figure_path}")


if __name__ == "__main__":
    main()
