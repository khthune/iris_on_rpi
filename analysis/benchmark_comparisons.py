# benchmark_comparisons.py

from argparse import ArgumentParser
from pathlib import Path
import sys
import time

import cv2 as cv
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from filter_loader import load_filter_bank
from iris import IrisClassifier, get_iris_band, hamming_distance


filters, _ = load_filter_bank(None)


def parse_rotation(value):
    if isinstance(value, int):
        return value

    normalized = str(value).strip().lower()
    if normalized == "none":
        return None

    rotation = int(normalized)
    if rotation < 0:
        raise ValueError("rotation must be non-negative or 'none'")
    return rotation


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


def format_result(result):
    if isinstance(result, np.generic):
        return result.item()
    if isinstance(result, tuple):
        return tuple(format_result(value) for value in result)
    if isinstance(result, list):
        return [format_result(value) for value in result]
    return result


def compare_irises_full(classifier, iris1, mask1, iris2, mask2):
    code1, code_mask1, _ = classifier.get_iris_code(iris1, mask1, offset=0)
    code2, code_mask2, _ = classifier.get_iris_code(iris2, mask2, offset=0)
    score = hamming_distance(
        np.asarray(code1, dtype=bool),
        np.asarray(code2, dtype=bool),
        np.asarray(code_mask1, dtype=bool),
        np.asarray(code_mask2, dtype=bool),
    )
    return score


def main():
    parser = ArgumentParser(description="Benchmark iris comparison operations.")
    parser.add_argument("image1", help="Path to the first iris image")
    parser.add_argument("image2", help="Path to the second iris image")
    parser.add_argument("--runs", type=int, default=100, help="Number of benchmark runs per operation")
    parser.add_argument(
        "--rotation",
        type=parse_rotation,
        default=21,
        help="Rotation count for iris-vs-code comparisons, or 'none' to disable rotations",
    )
    args = parser.parse_args()

    if args.runs < 1:
        raise ValueError("--runs must be at least 1")
    if args.rotation == 0:
        args.rotation = None

    print(f"Filters in use: {len(filters)}")
    classifier = IrisClassifier(filters)

    raw_image1 = load_image(args.image1)
    raw_image2 = load_image(args.image2)

    iris1, mask1 = segment_image(raw_image1)
    iris2, mask2 = segment_image(raw_image2)

    code1, code_mask1, _ = classifier.get_iris_code(iris1, mask1)
    code2, code_mask2, _ = classifier.get_iris_code(iris2, mask2)

    code1 = np.asarray(code1, dtype=bool)
    code2 = np.asarray(code2, dtype=bool)
    code_mask1 = np.asarray(code_mask1, dtype=bool)
    code_mask2 = np.asarray(code_mask2, dtype=bool)

    benchmark(
        "iris_to_code",
        args.runs,
        lambda: classifier.get_iris_code(iris1, mask1),
    )
    benchmark(
        "code_vs_code",
        args.runs,
        lambda: hamming_distance(code1, code2, code_mask1, code_mask2),
    )
    benchmark(
        "iris_vs_code",
        args.runs,
        lambda: classifier.compare_iris_code_and_iris(
            iris2,
            code1,
            mask2,
            code_mask1,
            rotation=args.rotation,
        ),
    )
    benchmark(
        "iris_vs_iris_full",
        args.runs,
        lambda: compare_irises_full(
            classifier,
            iris1,
            mask1,
            iris2,
            mask2,
        ),
    )


if __name__ == "__main__":
    main()
