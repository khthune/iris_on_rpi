# benchmark_cli

from argparse import ArgumentParser
import os
from pathlib import Path
import sys
import tempfile
import time

import cv2 as cv
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

DEFAULT_OUTPUT_DIR = Path(__file__).resolve().parent / "output" / Path(__file__).stem
MATPLOTLIB_CONFIG_DIR = DEFAULT_OUTPUT_DIR / "matplotlib"
MATPLOTLIB_CONFIG_DIR.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(MATPLOTLIB_CONFIG_DIR))
os.environ.setdefault("XDG_CACHE_HOME", str(MATPLOTLIB_CONFIG_DIR / "cache"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from filter_loader import load_filter_bank
from iris import IrisClassifier, get_iris_band, hamming_distance, hamming_distances
from pairwise_iris_analysis import (
    MATCHER_IRISCODE,
)
from rotation_part_scoring import part_scores_for_offsets, select_parts, split_code_slices


def add_figure_metadata(figure, metadata):
    if not metadata:
        return
    text = " | ".join(f"{key}={value}" for key, value in metadata.items() if value is not None)
    if text:
        figure.text(0.01, 0.01, text, ha="left", va="bottom", fontsize=7, family="monospace", wrap=True)


def load_image(path):
    image_path = Path(path).expanduser().resolve()
    image = cv.imread(str(image_path), cv.IMREAD_GRAYSCALE)
    if image is None:
        raise FileNotFoundError(f"Failed to load image '{image_path}'.")
    return image


def segment_image(image):
    iris_band, iris_mask = get_iris_band(image)
    if iris_band is None or iris_mask is None:
        raise RuntimeError("Iris segmentation failed.")
    return iris_band, iris_mask


def benchmark(name, runs, func):
    times = np.empty(runs, dtype=np.float64)
    result = None
    for index in range(runs):
        start = time.perf_counter()
        result = func()
        times[index] = time.perf_counter() - start

    print(name)
    print(f"  runs: {runs}")
    print(f"  mean: {times.mean():.8f} s")
    print(f"  median: {np.median(times):.8f} s")
    print(f"  min: {times.min():.8f} s")
    print(f"  max: {times.max():.8f} s")
    print(f"  last_result: {format_result(result)}")
    return {
        "name": name,
        "runs": runs,
        "mean": float(times.mean()),
        "median": float(np.median(times)),
        "min": float(times.min()),
        "max": float(times.max()),
        "last_result": format_result(result),
    }


def format_result(result):
    if isinstance(result, np.generic):
        return result.item()
    if isinstance(result, tuple):
        return tuple(format_result(value) for value in result)
    if isinstance(result, list):
        return [format_result(value) for value in result]
    return result


def build_database(classifier, image_paths):
    codes = []
    for image_path in image_paths:
        image = load_image(image_path)
        iris_band, iris_mask = segment_image(image)
        iris_code, mask_code, _ = classifier.get_iris_code(iris_band, iris_mask)
        codes.append(
            np.stack(
                (
                    np.asarray(iris_code, dtype=bool),
                    np.asarray(mask_code, dtype=bool),
                ),
                axis=0,
            )
        )
    return np.stack(codes, axis=1)


def enroll_operation(classifier, image):
    iris_band, iris_mask = segment_image(image)
    iris_code, mask_code, _ = classifier.get_iris_code(iris_band, iris_mask)
    code = np.stack(
        (
            np.asarray(iris_code, dtype=bool),
            np.asarray(mask_code, dtype=bool),
        ),
        axis=0,
    )

    with tempfile.NamedTemporaryFile(suffix=".npy") as handle:
        np.save(handle.name, code[:, np.newaxis, :])
    return code.shape


def rotation_offsets(rotation, rotation_step=1):
    search_rotation = rotation if rotation and rotation > 1 else 1
    offsets = np.arange(search_rotation, dtype=np.int64) - search_rotation // 2
    rotation_step = int(rotation_step)
    if rotation_step < 1:
        raise ValueError("--rotation-step must be at least 1")
    if rotation_step == 1:
        return offsets
    stepped_offsets = offsets[offsets % rotation_step == 0]
    if stepped_offsets.size == 0:
        return np.array([0], dtype=np.int64)
    return stepped_offsets


def compare_iris_code_parts(classifier, iris_band, iris_mask, stored_code, stored_mask, offsets, parts):
    iris_codes, mask_codes, _ = classifier.get_iris_codes(
        iris_band,
        iris_mask,
        offsets=offsets,
    )
    slices = split_code_slices(stored_code.shape[0], parts)
    part_scores, part_offsets = part_scores_for_offsets(
        stored_code,
        stored_mask,
        iris_codes,
        mask_codes,
        offsets,
        slices,
        min_valid_bits=1,
    )
    selected = select_parts(part_scores, part_offsets, eliminate=0)
    return selected["avg_hd"], selected["anchor_offset"]


def compare_iris_code_operation(classifier, image, stored_code, offsets, parts=None):
    iris_band, iris_mask = segment_image(image)
    if parts is not None:
        return compare_iris_code_parts(
            classifier,
            iris_band,
            iris_mask,
            stored_code[0, 0],
            stored_code[1, 0],
            offsets,
            parts,
        )
    iris_codes, mask_codes, _ = classifier.get_iris_codes(
        iris_band,
        iris_mask,
        offsets=offsets,
    )
    scores = hamming_distances(iris_codes, stored_code[0, 0], mask_codes, stored_code[1, 0])
    best_index = int(np.argmin(scores))
    return float(scores[best_index]), int(offsets[best_index])


def compare_image_operation(classifier, image1, image2, offsets, parts=None):
    iris1, mask1 = segment_image(image1)
    iris2, mask2 = segment_image(image2)
    code1, code_mask1, _ = classifier.get_iris_code(iris1, mask1)
    if parts is not None:
        return compare_iris_code_parts(
            classifier,
            iris2,
            mask2,
            np.asarray(code1, dtype=bool),
            np.asarray(code_mask1, dtype=bool),
            offsets,
            parts,
        )
    iris_codes, mask_codes, _ = classifier.get_iris_codes(iris2, mask2, offsets=offsets)
    scores = hamming_distances(iris_codes, np.asarray(code1, dtype=bool), mask_codes, np.asarray(code_mask1, dtype=bool))
    best_index = int(np.argmin(scores))
    return float(scores[best_index]), int(offsets[best_index])


def find_operation(classifier, query_image, codes, offsets, parts=None):
    iris_band, iris_mask = segment_image(query_image)
    iris_codes, mask_codes, _ = classifier.get_iris_codes(
        iris_band,
        iris_mask,
        offsets=offsets,
    )
    slices = split_code_slices(codes.shape[2], parts) if parts is not None else None
    best_match = None
    best_score = float("inf")
    for index in range(codes.shape[1]):
        if parts is None:
            curr_scores = [
                hamming_distance(
                    np.asarray(code, dtype=bool),
                    codes[0, index],
                    np.asarray(mask, dtype=bool),
                    codes[1, index],
                )
                for code, mask in zip(iris_codes, mask_codes)
            ]
            curr_score = float(np.min(curr_scores))
        else:
            part_scores, part_offsets = part_scores_for_offsets(
                codes[0, index],
                codes[1, index],
                iris_codes,
                mask_codes,
                offsets,
                slices,
                min_valid_bits=1,
            )
            curr_score = float(select_parts(part_scores, part_offsets, eliminate=0)["avg_hd"])
        if curr_score < best_score:
            best_score = curr_score
            best_match = index

    return best_match, best_score


def plot_benchmark_results(results, output_name, title, metadata=None):
    labels = [
        "Enroll",
        "Compare Iris Code",
        "Compare Image",
        "Find",
    ]
    means = [result["mean"] for result in results]

    if output_name:
        output = Path(output_name).expanduser()
        if not output.is_absolute():
            output = DEFAULT_OUTPUT_DIR / output
        output = output.resolve()
        output.parent.mkdir(parents=True, exist_ok=True)
        figure, axis = plt.subplots(figsize=(9, 5))
        bars = axis.bar(labels, means, color="#7c8aa5")
        axis.set_title(title)
        axis.set_ylabel("Mean Time (seconds)")
        axis.set_ylim(0, max(means) * 1.15 if means else 1.0)
        for bar, mean in zip(bars, means):
            axis.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height(),
                f"{mean:.2f}s",
                ha="center",
                va="bottom",
            )
        add_figure_metadata(figure, metadata or {})
        figure.tight_layout(rect=(0, 0.06, 1, 1))
        figure.savefig(output, dpi=200, bbox_inches="tight")
        plt.close(figure)
        print(f"Saved benchmark plot to {output}")


def main():
    parser = ArgumentParser(description="Benchmark CLI-style operations for the active matcher and rotation setting.")
    parser.add_argument("query_image", help="Query image path")
    parser.add_argument("compare_image", help="Second image path for comparison")
    parser.add_argument(
        "--database-images",
        nargs="+",
        help="Images to enroll into an in-memory database for the find benchmark",
    )
    parser.add_argument(
        "--rotation",
        type=int,
        default=1,
        help="Rotation count used for comparison and find operations. The default 1 keeps the no-rotation behavior.",
    )
    parser.add_argument(
        "--rotation-step",
        type=int,
        default=1,
        help="Only test offsets divisible by this step. Example: --rotation 21 --rotation-step 4 tests -8,-4,0,4,8.",
    )
    parser.add_argument(
        "--parts",
        type=int,
        default=None,
        help="Split the iriscode into this many parts and compare by average per-part best HD.",
    )
    parser.add_argument("--runs", type=int, default=100, help="Number of runs per benchmark")
    parser.add_argument(
        "--output-name",
        "--figure-output",
        dest="output_name",
        default=f"{Path(__file__).stem}.png",
        help="Output filename for the benchmark bar chart inside the default output directory, or an absolute path.",
    )
    parser.add_argument(
        "--filters",
        "--filters-file",
        dest="filters",
        default=None,
        help="Optional Python filters file containing a 'filters' list. Defaults to project filters.py.",
    )
    args = parser.parse_args()

    if args.runs < 1:
        raise ValueError("--runs must be at least 1")
    if args.rotation < 1:
        raise ValueError("--rotation must be at least 1")
    if args.rotation_step < 1:
        raise ValueError("--rotation-step must be at least 1")
    if args.parts is not None and args.parts < 1:
        raise ValueError("--parts must be at least 1")

    query_image = load_image(args.query_image)
    compare_image = load_image(args.compare_image)
    database_images = args.database_images if args.database_images else [args.query_image, args.compare_image]
    selected_filters, filters_source = load_filter_bank(args.filters)
    offsets = rotation_offsets(args.rotation, args.rotation_step)

    print(f"Filters in use: {len(selected_filters)}")
    print(f"Filters source: {filters_source}")
    print(f"Rotation offsets: {','.join(str(int(offset)) for offset in offsets)}")
    classifier = IrisClassifier(selected_filters)
    stored_database = build_database(
        classifier,
        database_images,
    )
    stored_template = stored_database[:, :1, :]

    results = []
    results.append(benchmark(
        "enroll",
        args.runs,
        lambda: enroll_operation(classifier, query_image),
    ))
    results.append(benchmark(
        "compare_template",
        args.runs,
        lambda: compare_iris_code_operation(classifier, query_image, stored_template, offsets, args.parts),
    ))
    results.append(benchmark(
        "compare_image",
        args.runs,
        lambda: compare_image_operation(classifier, query_image, compare_image, offsets, args.parts),
    ))
    results.append(benchmark(
        "find",
        args.runs,
        lambda: find_operation(classifier, query_image, stored_database, offsets, args.parts),
    ))

    rotation_label = "No Rotation" if args.rotation <= 1 else f"Rotation {args.rotation}"
    if args.rotation_step > 1:
        rotation_label = f"{rotation_label}, step {args.rotation_step}"
    parts_label = "whole iriscode" if args.parts is None else f"{args.parts} parts"
    title = f"CLI Benchmark ({MATCHER_IRISCODE}, {rotation_label}, {parts_label})"
    plot_benchmark_results(
        results,
        args.output_name,
        title,
        metadata={
            "query_image": Path(args.query_image).name,
            "compare_image": Path(args.compare_image).name,
            "database_images": len(database_images),
            "rotation": args.rotation,
            "rotation_step": args.rotation_step,
            "rotation_offsets": ",".join(str(int(offset)) for offset in offsets),
            "parts": args.parts,
            "runs": args.runs,
            "matcher": MATCHER_IRISCODE,
            "seg_path": os.environ.get("SEG_PATH"),
            "filter_count": len(selected_filters),
            "filters_source": filters_source,
            "output_name": args.output_name,
        },
    )


if __name__ == "__main__":
    main()
