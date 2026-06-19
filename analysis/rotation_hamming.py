from argparse import ArgumentParser
from pathlib import Path
import csv
import sys

import cv2 as cv
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import MaxNLocator


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from filter_loader import load_filter_bank
from iris import (
    IrisClassifier,
    UNET_ONNX_PATH,
    get_iris_band,
    get_segmentation_backend_name,
    hamming_distances,
)


DEFAULT_OUTPUT_DIR = Path(__file__).resolve().parent / "output" / "rotation_hamming"


def safe_output_name(name):
    return "".join(char if char.isalnum() or char in ("-", "_") else "_" for char in str(name))


def load_image(path):
    image_path = Path(path).expanduser().resolve()
    image = cv.imread(str(image_path), cv.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(f"Failed to load image: {image_path}")
    return image_path, image


def encode_image(classifier, image, image_name):
    try:
        iris_band, iris_mask = get_iris_band(image)
    except Exception as exc:
        raise RuntimeError(f"Segmentation failed for {image_name}: {exc}") from exc
    if iris_band is None or iris_mask is None:
        raise RuntimeError(f"Segmentation failed for {image_name}")
    iris_code, iris_code_mask, _ = classifier.get_iris_code(iris_band, iris_mask, offset=0)
    return iris_band, iris_mask, iris_code, iris_code_mask


def rotation_offsets(rotation):
    if rotation < 1:
        raise ValueError("--rotation must be at least 1")
    return np.arange(rotation, dtype=np.int64) - rotation // 2


def profile_against_reference(
    classifier,
    reference_band,
    reference_code,
    reference_code_mask,
    probe_band,
    probe_mask,
    offsets,
):
    rotated_codes, rotated_masks, _ = classifier.get_iris_codes(probe_band, probe_mask, offsets=offsets)
    scores = hamming_distances(
        rotated_codes,
        reference_code,
        rotated_masks,
        reference_code_mask,
    )
    valid_bits = np.sum(np.bitwise_and(rotated_masks, reference_code_mask), axis=1)
    return scores, valid_bits


def write_csv(path, rows):
    output = Path(path).expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["rotation_offset", "hamming_distance", "valid_bits"])
        writer.writeheader()
        writer.writerows(rows)
    return output


def plot_profile(
    output_path,
    image_path,
    compare_image_path,
    rotation,
    offsets,
    scores,
    valid_bits,
    filter_count,
    filters_source,
):
    best_index = int(np.argmin(scores))
    best_offset = int(offsets[best_index])
    best_score = float(scores[best_index])
    mode = "self comparison" if compare_image_path is None else "image pair comparison"

    figure, axis = plt.subplots(figsize=(10, 6))
    axis.plot(offsets, scores, color="#1f77b4", marker="o", markersize=4, lw=1.8)
    axis.scatter(
        [best_offset],
        [best_score],
        color="#d62728",
        s=55,
        zorder=3,
        label=f"best: offset {best_offset}, HD {best_score:.4f}",
    )
    axis.set_title("Hamming Distance Across Rotation Offsets")
    axis.set_xlabel("Rotation offset")
    axis.set_ylabel("Hamming distance")
    axis.xaxis.set_major_locator(MaxNLocator(nbins=12, integer=True))
    axis.tick_params(axis="x", labelsize=8)
    axis.grid(True, alpha=0.3)
    axis.legend(loc="best")

    min_score = float(np.min(scores))
    max_score = float(np.max(scores))
    score_pad = max(0.01, (max_score - min_score) * 0.12)
    axis.set_ylim(max(0.0, min_score - score_pad), min(1.0, max_score + score_pad))

    parameter_text = "\n".join(
        [
            f"image: {image_path.name}",
            f"compare image: {compare_image_path.name if compare_image_path is not None else image_path.name}",
            f"mode: {mode}",
            f"rotation: {rotation}",
            f"offset range: {int(offsets[0])}..{int(offsets[-1])}",
            f"filters: {filter_count}",
            f"filters file: {filters_source}",
            f"segmentation: {get_segmentation_backend_name()}",
            f"model: {Path(UNET_ONNX_PATH).name}",
            f"best offset: {best_offset}",
            f"best HD: {best_score:.6f}",
            f"valid bits at best: {int(valid_bits[best_index])}",
        ]
    )
    axis.text(
        0.015,
        0.985,
        parameter_text,
        transform=axis.transAxes,
        va="top",
        ha="left",
        fontsize=8.5,
        bbox={"boxstyle": "round,pad=0.35", "facecolor": "white", "edgecolor": "#cccccc", "alpha": 0.9},
    )

    output = Path(output_path).expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=300, bbox_inches="tight")
    plt.close(figure)
    return output


def main():
    parser = ArgumentParser(
        description="Plot iris-code Hamming distance across rotation offsets for one image or an image pair."
    )
    parser.add_argument("--img", required=True, help="Reference image path.")
    parser.add_argument(
        "--compare-img",
        default=None,
        help="Optional probe image path. If omitted, the reference image is compared against itself.",
    )
    parser.add_argument("--rotation", type=int, required=True, help="Number of offsets to evaluate around zero.")
    parser.add_argument("--output-name", required=True, help="Base output filename without extension.")
    parser.add_argument("--figure-output", default=None, help="Optional explicit PNG output path.")
    parser.add_argument("--csv-output", default=None, help="Optional CSV output path.")
    parser.add_argument(
        "--filters",
        dest="filters",
        default=None,
        help="Optional Python filters file containing a 'filters' list. Defaults to project filters.py.",
    )
    args = parser.parse_args()

    offsets = rotation_offsets(args.rotation)
    image_path, image = load_image(args.img)
    selected_filters, filters_source = load_filter_bank(args.filters)
    print(f"Filters in use: {len(selected_filters)}")
    print(f"Filters source: {filters_source}")
    classifier = IrisClassifier(selected_filters)
    reference_band, reference_mask, reference_code, reference_code_mask = encode_image(
        classifier,
        image,
        image_path.name,
    )
    compare_image_path = None
    probe_band = reference_band
    probe_mask = reference_mask
    if args.compare_img:
        compare_image_path, compare_image = load_image(args.compare_img)
        probe_band, probe_mask, _, _ = encode_image(classifier, compare_image, compare_image_path.name)

    scores, valid_bits = profile_against_reference(
        classifier,
        reference_band,
        reference_code,
        reference_code_mask,
        probe_band,
        probe_mask,
        offsets,
    )

    output_name = safe_output_name(args.output_name)
    figure_output = args.figure_output or DEFAULT_OUTPUT_DIR / f"{output_name}.png"
    figure_path = plot_profile(
        figure_output,
        image_path,
        compare_image_path,
        args.rotation,
        offsets,
        scores,
        valid_bits,
        len(selected_filters),
        filters_source,
    )
    print(f"Wrote plot: {figure_path}")

    if args.csv_output:
        rows = [
            {
                "rotation_offset": int(offset),
                "hamming_distance": float(score),
                "valid_bits": int(bits),
            }
            for offset, score, bits in zip(offsets, scores, valid_bits)
        ]
        csv_path = write_csv(args.csv_output, rows)
        print(f"Wrote CSV: {csv_path}")


if __name__ == "__main__":
    main()
