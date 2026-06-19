from argparse import ArgumentParser
import csv
import json
import os
from pathlib import Path
import re
import sys
import time

import cv2 as cv
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
ANALYSIS_ROOT = Path(__file__).resolve().parent
if str(ANALYSIS_ROOT) not in sys.path:
    sys.path.insert(0, str(ANALYSIS_ROOT))
MATPLOTLIB_CONFIG_DIR = ANALYSIS_ROOT / "output" / "benchmark_pipeline" / "matplotlib"
MATPLOTLIB_CONFIG_DIR.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(MATPLOTLIB_CONFIG_DIR))

from dataset_loaders import DATASET_CHOICES, load_dataset, resolve_dataset, sample_dataset
from filter_loader import load_filter_bank
from iris import (
    IrisClassifier,
    UNET_BAND_SHAPE,
    UNET_ONNX_PATH,
    fit_boundary_from_mask,
    fit_polar_boundary_from_mask,
    get_iris_band,
    get_segmentation_backend_name,
    normalize_iris_from_boundaries,
)
from pairwise_iris_analysis import (
    MATCHER_IRISCODE,
    compute_pairwise_scores_iriscode,
    evaluate_zero_false_accept_threshold,
    evaluate_scores,
    precompute_codes,
    summarize_label_pairs,
)

DEFAULT_OUTPUT_DIR = Path(__file__).resolve().parent / "output" / "benchmark_pipeline"
DEFAULT_GT_MANIFEST_ROOT = Path(__file__).resolve().parent / "output" / "gt_manifests"
DEFAULT_GT_ANNOTATION_ROOT = PROJECT_ROOT / "GT" / "dataset"
TRUE_ENV_VALUES = {"1", "true", "yes", "on"}
GT_MANIFEST_SLUGS = {
    "casia-v3-interval": "casia_v3_interval",
    "casia-v4-interval": "casia_v4_interval",
    "iitd": "iitd",
}
GT_ANNOTATION_DIRS = {
    "casia-v3-interval": "casia3i/OperatorA",
    "casia-v4-interval": "casia4i/OperatorA",
    "iitd": "iitd/OperatorA",
}
DEFAULT_PAIRWISE_CONFIGS = {
    "casia-v1": {
        "max_samples": None,
        "max_identities": None,
        "max_images_per_identity": None,
        "seed": 0,
    },
    "casia-v3-interval": {
        "max_samples": None,
        "max_identities": 250,
        "max_images_per_identity": 2,
        "seed": 0,
    },
    "casia-v4-interval": {
        "max_samples": None,
        "max_identities": 250,
        "max_images_per_identity": 2,
        "seed": 0,
    },
    "casia-distance": {
        "max_samples": None,
        "max_identities": 250,
        "max_images_per_identity": 2,
        "seed": 0,
    },
    "casia-1000": {
        "max_samples": None,
        "max_identities": 250,
        "max_images_per_identity": 2,
        "seed": 0,
    },
    "casia-v3-lamp": {
        "max_samples": None,
        "max_identities": 250,
        "max_images_per_identity": 2,
        "seed": 0,
    },
    "casia-v3-twins": {
        "max_samples": None,
        "max_identities": 200,
        "max_images_per_identity": 2,
        "seed": 0,
    },
    "iitd": {
        "max_samples": None,
        "max_identities": 224,
        "max_images_per_identity": 2,
        "seed": 0,
    },
    "mmu": {
        "max_samples": None,
        "max_identities": None,
        "max_images_per_identity": 2,
        "seed": 0,
    },
    "mmu2": {
        "max_samples": None,
        "max_identities": None,
        "max_images_per_identity": 2,
        "seed": 0,
    },
}


def format_result(value):
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, tuple):
        return [format_result(item) for item in value]
    if isinstance(value, list):
        return [format_result(item) for item in value]
    if isinstance(value, dict):
        return {key: format_result(item) for key, item in value.items()}
    return value


def env_flag(name):
    return os.environ.get(name, "").strip().lower() in TRUE_ENV_VALUES


def load_rotation_consistency_classifier():
    try:
        from rotation_part_scoring import (
            compute_pairwise_rotation_classifier,
            evaluate_eer,
            summarize_predictions,
        )
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "rotation_part_scoring.py is required for --score, "
            "--tolerance-offset, or --match-parts."
        ) from exc
    return compute_pairwise_rotation_classifier, evaluate_eer, summarize_predictions


def build_rotation_offsets(rotation, rotation_step=1):
    offsets = np.arange(rotation, dtype=np.int64) - rotation // 2
    rotation_step = int(rotation_step)
    if rotation_step < 1:
        raise ValueError("--rotation-step must be at least 1")
    if rotation_step == 1:
        return offsets
    stepped_offsets = offsets[offsets % rotation_step == 0]
    if stepped_offsets.size == 0:
        return np.array([0], dtype=np.int64)
    return stepped_offsets


def segment_image(image):
    try:
        return get_iris_band(image)
    except Exception:
        return None, None


def load_gt_manifest(dataset_format, manifest_root=DEFAULT_GT_MANIFEST_ROOT):
    slug = GT_MANIFEST_SLUGS.get(dataset_format)
    if slug is None:
        supported = ", ".join(sorted(GT_MANIFEST_SLUGS))
        raise ValueError(f"--gt-mask is not available for {dataset_format}. Available GT-mask datasets: {supported}")

    manifest_path = Path(manifest_root).expanduser().resolve() / slug / "manifest.csv"
    if not manifest_path.exists():
        raise FileNotFoundError(f"--gt-mask requested, but GT manifest was not found: {manifest_path}")

    lookup = {}
    with manifest_path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            source_name = row.get("source_name")
            label_path = row.get("label_path")
            if not source_name or not label_path:
                continue
            label_path = Path(label_path).expanduser()
            if not label_path.is_absolute():
                label_path = (manifest_path.parent / label_path).resolve()
            lookup[source_name] = label_path

    if not lookup:
        raise ValueError(f"GT manifest contains no usable rows: {manifest_path}")
    return manifest_path, lookup


def gt_annotation_stem(dataset_format, image_name):
    image_name = str(image_name)
    image_path = Path(image_name)

    if dataset_format in {"casia-v3-interval", "casia-v4-interval"}:
        return image_path.stem

    if dataset_format == "iitd":
        match = re.match(r"(?P<index>\d+)_", image_path.stem, flags=re.IGNORECASE)
        if not match:
            raise ValueError(f"Unexpected IITD image name for GT annotation mapping: {image_name}")
        image_index = int(match.group("index"))
        eye_code = "A" if image_index <= 5 else "B"
        return f"{image_path.parent.name}-{eye_code}_{image_index:02d}"

    raise ValueError(f"--gt-mask is not available for {dataset_format}.")


def load_gt_points(path):
    points = np.loadtxt(path, dtype=np.float32)
    points = np.asarray(points, dtype=np.float32)
    if points.ndim == 1:
        points = points.reshape(1, -1)
    if points.shape[1] != 2 or len(points) < 3:
        raise ValueError(f"Expected at least three x/y GT points in {path}")
    return np.rint(points).astype(np.int32)


def boundary_mask_from_points(points, shape):
    mask = np.zeros(shape, dtype=np.uint8)
    points = np.asarray(points, dtype=np.float32)
    if len(points) >= 5:
        (center_x, center_y), (diameter_x, diameter_y), angle_degrees = cv.fitEllipse(points.reshape(-1, 1, 2))
        cv.ellipse(
            mask,
            (int(round(center_x)), int(round(center_y))),
            (
                max(1, int(round(diameter_x / 2.0))),
                max(1, int(round(diameter_y / 2.0))),
            ),
            float(angle_degrees),
            0,
            360,
            1,
            thickness=-1,
        )
    else:
        (center_x, center_y), radius = cv.minEnclosingCircle(points.reshape(-1, 1, 2))
        cv.circle(
            mask,
            (int(round(center_x)), int(round(center_y))),
            max(1, int(round(radius))),
            1,
            thickness=-1,
        )
    return mask


def load_gt_annotation_masks(dataset_format, image_name, image_shape, annotation_root=DEFAULT_GT_ANNOTATION_ROOT):
    annotation_dir = GT_ANNOTATION_DIRS.get(dataset_format)
    if annotation_dir is None:
        supported = ", ".join(sorted(GT_ANNOTATION_DIRS))
        raise ValueError(f"--gt-mask is not available for {dataset_format}. Available GT-mask datasets: {supported}")

    stem = gt_annotation_stem(dataset_format, image_name)
    base = Path(annotation_root).expanduser().resolve() / annotation_dir
    inner_path = base / f"{stem}.inner.txt"
    outer_path = base / f"{stem}.outer.txt"
    missing = [str(path) for path in (inner_path, outer_path) if not path.exists()]
    if missing:
        raise FileNotFoundError(f"Missing GT boundary annotation for image '{image_name}': {', '.join(missing)}")

    inner_points = load_gt_points(inner_path)
    outer_points = load_gt_points(outer_path)
    return (
        boundary_mask_from_points(inner_points, image_shape),
        boundary_mask_from_points(outer_points, image_shape),
    )


def gt_masks_to_band(image, iris_mask, pupil_mask, valid_mask):
    pupil_ellipse = fit_boundary_from_mask(pupil_mask, prefer_ellipse=True)
    center = (pupil_ellipse.center_x, pupil_ellipse.center_y)
    pupil_boundary = fit_polar_boundary_from_mask(
        pupil_mask,
        center=center,
        num_angles=UNET_BAND_SHAPE[1],
        smooth_kernel=7,
    )
    iris_boundary = fit_polar_boundary_from_mask(
        iris_mask,
        center=center,
        num_angles=UNET_BAND_SHAPE[1],
        smooth_kernel=17,
    )
    if np.mean(iris_boundary.radii) <= np.mean(pupil_boundary.radii):
        raise ValueError("GT iris boundary must be larger than GT pupil boundary.")
    return normalize_iris_from_boundaries(
        image,
        pupil_boundary,
        iris_boundary,
        valid_mask.astype(np.uint8) * 255,
        band_shape=UNET_BAND_SHAPE,
    )


def build_gt_band_getter(
    dataset_format,
    manifest_root=DEFAULT_GT_MANIFEST_ROOT,
    annotation_root=DEFAULT_GT_ANNOTATION_ROOT,
):
    manifest_path, gt_lookup = load_gt_manifest(dataset_format, manifest_root=manifest_root)
    annotation_root = Path(annotation_root).expanduser().resolve()

    def get_gt_iris_band(image, image_name):
        image_name = str(image_name)
        label_path = gt_lookup.get(image_name)
        if label_path is None:
            raise FileNotFoundError(f"No GT mask for image '{image_name}' in {manifest_path}")
        if not label_path.exists():
            raise FileNotFoundError(f"GT mask file does not exist for image '{image_name}': {label_path}")

        gt_mask = cv.imread(str(label_path), cv.IMREAD_GRAYSCALE)
        if gt_mask is None:
            raise FileNotFoundError(f"Failed to read GT mask for image '{image_name}': {label_path}")
        if gt_mask.shape[:2] != image.shape[:2]:
            gt_mask = cv.resize(gt_mask, (image.shape[1], image.shape[0]), interpolation=cv.INTER_NEAREST)

        pupil_mask, iris_mask = load_gt_annotation_masks(
            dataset_format,
            image_name,
            image.shape[:2],
            annotation_root=annotation_root,
        )
        annulus_mask = iris_mask.astype(bool) & ~pupil_mask.astype(bool)
        valid_mask = (gt_mask > 0) & annulus_mask & (image < 254)
        return gt_masks_to_band(
            image,
            iris_mask=iris_mask,
            pupil_mask=pupil_mask,
            valid_mask=valid_mask,
        )

    get_gt_iris_band.available_image_names = set(gt_lookup)
    return manifest_path, get_gt_iris_band


def filter_samples_by_available_names(images, labels, image_names, available_image_names):
    if available_image_names is None:
        return images, labels, image_names

    keep_indices = [index for index, name in enumerate(image_names) if str(name) in available_image_names]
    if not keep_indices:
        raise RuntimeError("No dataset images matched the selected GT mask manifest.")
    return (
        [images[index] for index in keep_indices],
        labels[keep_indices],
        image_names[keep_indices],
    )


def compute_iriscode_bit_count(selected_filters):
    classifier = IrisClassifier(selected_filters)
    dummy_band = np.zeros(UNET_BAND_SHAPE, dtype=np.float32)
    dummy_mask = np.ones(UNET_BAND_SHAPE, dtype=bool)
    iris_code, _mask_code, _filter_ids = classifier.get_iris_code(dummy_band, dummy_mask)
    return int(len(iris_code))


def evaluate_fixed_hd_threshold(same_class, scores, threshold):
    same_class = np.asarray(same_class, dtype=bool)
    scores = np.asarray(scores, dtype=np.float64)
    predicted_mated = scores <= threshold
    mated_total = int(np.sum(same_class))
    non_mated_total = int(np.sum(~same_class))
    mated_wrong = int(np.sum(same_class & ~predicted_mated))
    non_mated_wrong = int(np.sum(~same_class & predicted_mated))
    mated_correct = int(np.sum(same_class & predicted_mated))
    non_mated_correct = int(np.sum(~same_class & ~predicted_mated))
    return {
        "threshold": float(threshold),
        "rule": "score <= threshold",
        "mated_wrongly_classified_non_mated": mated_wrong,
        "non_mated_wrongly_classified_mated": non_mated_wrong,
        "mated_correct": mated_correct,
        "non_mated_correct": non_mated_correct,
        "mated_total": mated_total,
        "non_mated_total": non_mated_total,
        "tpr": None if mated_total == 0 else float(mated_correct / mated_total),
        "false_reject_rate": None if mated_total == 0 else float(mated_wrong / mated_total),
        "false_accept_rate": None if non_mated_total == 0 else float(non_mated_wrong / non_mated_total),
    }


def run_pairwise_benchmark(
    dataset_path,
    dataset_format,
    rotation,
    rotation_offsets,
    output_dir,
    selected_filters,
    filters_source,
    include_rotation_consistency=False,
    rotation_consistency_parts=5,
    rotation_consistency_threshold=0.3955,
    rotation_consistency_eliminate=1,
    rotation_consistency_tolerance_offset=None,
    rotation_consistency_score="hd",
    rotation_consistency_match_parts=None,
    fixed_threshold=None,
    max_identities=None,
    max_images_per_identity=None,
    seed=None,
    band_getter=None,
    segmentation_source="onnx",
    ):
    config = dict(DEFAULT_PAIRWISE_CONFIGS[dataset_format])
    if max_identities is not None:
        config["max_identities"] = max_identities
    if max_images_per_identity is not None:
        config["max_images_per_identity"] = max_images_per_identity
    if seed is not None:
        config["seed"] = seed
    images, labels, image_names = load_dataset(dataset_path, dataset_format)
    images, labels, image_names = filter_samples_by_available_names(
        images,
        labels,
        image_names,
        getattr(band_getter, "available_image_names", None),
    )
    images, labels, image_names = sample_dataset(
        images,
        labels,
        image_names,
        max_samples=config["max_samples"],
        max_identities=config["max_identities"],
        max_images_per_identity=config["max_images_per_identity"],
        seed=config["seed"],
    )

    classifier = IrisClassifier(selected_filters)
    (
        base_codes,
        base_masks,
        rotated_codes,
        rotated_masks,
        offsets,
        kept_labels,
        kept_image_names,
        skipped,
    ) = precompute_codes(
        images,
        labels,
        image_names,
        classifier,
        rotation,
        band_getter=band_getter,
        offsets=rotation_offsets,
    )
    summary = summarize_label_pairs(kept_labels)
    pairwise = compute_pairwise_scores_iriscode(
        kept_labels,
        base_codes,
        base_masks,
        rotated_codes,
        rotated_masks,
        offsets,
    )
    feature_extractor = "gabor"
    evaluation = evaluate_scores(pairwise["same_class"], pairwise["scores"])
    zero_false_accept = evaluate_zero_false_accept_threshold(
        pairwise["same_class"],
        pairwise["scores"],
        lower_is_mated=True,
    )
    fixed_threshold_result = None
    if fixed_threshold is not None:
        fixed_threshold_result = evaluate_fixed_hd_threshold(
            pairwise["same_class"],
            pairwise["scores"],
            fixed_threshold,
        )
    rotation_consistency_result = None
    if include_rotation_consistency:
        (
            compute_pairwise_rotation_classifier,
            evaluate_rotation_consistency_eer,
            summarize_rotation_consistency_predictions,
        ) = load_rotation_consistency_classifier()
        rotation_rows = compute_pairwise_rotation_classifier(
            kept_labels,
            base_codes,
            base_masks,
            rotated_codes,
            rotated_masks,
            offsets,
            rotation_consistency_parts,
            rotation_consistency_threshold,
            rotation_consistency_eliminate,
            rotation_consistency_tolerance_offset,
            1,
            rotation_consistency_match_parts if rotation_consistency_score == "match-rotation" else None,
        )
        rotation_summary = summarize_rotation_consistency_predictions(rotation_rows)
        rotation_summary.update(evaluate_rotation_consistency_eer(rotation_rows))
        rotation_same_class = np.asarray([row["same_class"] for row in rotation_rows], dtype=bool)
        rotation_predicted_mated = np.asarray([row["predicted_mated"] for row in rotation_rows], dtype=bool)
        rotation_mated_wrong = int(np.sum(rotation_same_class & ~rotation_predicted_mated))
        rotation_non_mated_wrong = int(np.sum(~rotation_same_class & rotation_predicted_mated))
        rotation_mated_correct = int(np.sum(rotation_same_class & rotation_predicted_mated))
        rotation_non_mated_correct = int(np.sum(~rotation_same_class & ~rotation_predicted_mated))
        rotation_mated_total = int(np.sum(rotation_same_class))
        rotation_non_mated_total = int(np.sum(~rotation_same_class))
        prediction_mode = "rotation_match_count" if rotation_consistency_score == "match-rotation" else "hd_threshold"
        if rotation_consistency_score == "match-rotation":
            rotation_zero_false_accept = evaluate_zero_false_accept_threshold(
                [row["same_class"] for row in rotation_rows],
                [row["rotation_match_count"] for row in rotation_rows],
                lower_is_mated=False,
            )
        else:
            rotation_zero_false_accept = evaluate_zero_false_accept_threshold(
                [row["same_class"] for row in rotation_rows],
                [row["avg_hd"] for row in rotation_rows],
                lower_is_mated=True,
            )
        rotation_consistency_result = {
            "parts": int(rotation_consistency_parts),
            "score": rotation_consistency_score,
            "threshold": float(rotation_consistency_threshold),
            "eliminate": (
                None
                if rotation_consistency_tolerance_offset is not None
                else int(rotation_consistency_eliminate)
            ),
            "match_parts": (
                int(rotation_consistency_match_parts)
                if rotation_consistency_score == "match-rotation"
                else None
            ),
            "prediction_mode": prediction_mode,
            "tolerance_offset": (
                None
                if rotation_consistency_tolerance_offset is None
                else int(rotation_consistency_tolerance_offset)
            ),
            "eer_hd_threshold": (
                None if rotation_summary["eer_hd_threshold"] is None else float(rotation_summary["eer_hd_threshold"])
            ),
            "eer_match_parts_threshold": (
                None
                if rotation_summary["eer_match_parts_threshold"] is None
                else float(rotation_summary["eer_match_parts_threshold"])
            ),
            "summary": rotation_summary,
            "zero_false_accept": rotation_zero_false_accept,
            "fixed_threshold": {
                "threshold": float(rotation_consistency_threshold),
                "rule": (
                    "rotation_match_count >= threshold"
                    if rotation_consistency_score == "match-rotation"
                    else "avg_hd <= threshold"
                ),
                "mated_wrongly_classified_non_mated": rotation_mated_wrong,
                "non_mated_wrongly_classified_mated": rotation_non_mated_wrong,
                "mated_correct": rotation_mated_correct,
                "non_mated_correct": rotation_non_mated_correct,
                "mated_total": rotation_mated_total,
                "non_mated_total": rotation_non_mated_total,
                "tpr": None if rotation_mated_total == 0 else float(rotation_mated_correct / rotation_mated_total),
                "false_reject_rate": (
                    None if rotation_mated_total == 0 else float(rotation_mated_wrong / rotation_mated_total)
                ),
                "false_accept_rate": (
                    None
                    if rotation_non_mated_total == 0
                    else float(rotation_non_mated_wrong / rotation_non_mated_total)
                ),
            },
        }

    return {
        "dataset_format": dataset_format,
        "matcher": MATCHER_IRISCODE,
        "feature_extractor": feature_extractor,
        "filters_source": filters_source,
        "filters_count": int(len(selected_filters)),
        "segmentation_backend": get_segmentation_backend_name(),
        "segmentation_model_path": str(UNET_ONNX_PATH),
        "segmentation_source": segmentation_source,
        "dataset_path": str(dataset_path),
        "sample_summary": summary,
        "sampling": config,
        "rotation": int(rotation),
        "rotation_step": (
            None
            if len(offsets) < 2
            else int(np.min(np.diff(np.sort(offsets.astype(np.int64)))))
        ),
        "rotation_offsets": [int(offset) for offset in offsets],
        "kept_sample_count": int(len(kept_labels)),
        "skipped_sample_count": int(len(skipped)),
        "eer": float(evaluation["eer"]),
        "eer_std": float(evaluation["eer_std"]),
        "eer_fpr": float(evaluation["eer_fpr"]),
        "eer_fnr": float(evaluation["eer_fnr"]),
        "eer_threshold": float(evaluation["eer_threshold"]),
        "eer_hd_threshold": float(-evaluation["eer_threshold"]),
        "roc_auc": float(evaluation["roc_auc"]),
        "zero_false_accept": zero_false_accept,
        "fixed_threshold": fixed_threshold_result,
        "rotation_consistency_classifier": rotation_consistency_result,
    }


def main():
    parser = ArgumentParser(
        description=(
            "Evaluate the U-Net + Gabor iris pipeline for pairwise discrimination."
        )
    )
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=[
            "casia-v1",
            "casia-v3-interval",
            "casia-v4-interval",
            "casia-v3-lamp",
            "casia-v3-twins",
            "iitd",
            "mmu",
            "mmu2",
        ],
        choices=[dataset for dataset in DATASET_CHOICES if dataset != "auto"],
        help="Datasets to include in the evaluation.",
    )
    parser.add_argument("--rotation", type=int, default=21, help="Rotation count used for scoring.")
    parser.add_argument(
        "--rotation-step",
        type=int,
        default=1,
        help=(
            "Evaluate every Nth rotation offset inside the centered --rotation range. "
            "For example, --rotation 21 --rotation-step 4 tests -8,-4,0,4,8."
        ),
    )
    parser.add_argument(
        "--gt-mask",
        dest="gtmask",
        action="store_true",
        help=(
            "Use ground-truth binary masks from analysis/output/gt_manifests instead of the ONNX segmenter. "
            "Fails if any selected dataset has no GT manifest."
        ),
    )
    parser.add_argument(
        "--parts",
        dest="rotation_consistency_parts",
        type=int,
        default=None,
        help=(
            "Enable part-split average HD with this many iriscode parts. "
            "Defaults to --eliminate 0 and --score hd."
        ),
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.335,
        help=(
            "Also evaluate the normal pairwise HD benchmark at this fixed threshold. "
            "Mated is predicted when HD <= threshold."
        ),
    )
    parser.add_argument(
        "--score",
        dest="rotation_consistency_score",
        choices=["hd", "match-rotation"],
        default=None,
        help="Rotation-consistency score to evaluate. Supplying this enables the rotation-consistency classifier.",
    )
    parser.add_argument(
        "--eliminate",
        dest="rotation_consistency_eliminate",
        type=int,
        default=0,
        help="Remove this many parts whose best rotation is furthest from the lowest-HD part.",
    )
    parser.add_argument(
        "--match-parts",
        dest="rotation_consistency_match_parts",
        type=int,
        default=None,
        help=(
            "For the rotation-consistency classifier, predict mated when at least this many kept parts "
            "match the lowest-HD part's rotation. Only used with --score match-rotation."
        ),
    )
    parser.add_argument(
        "--tolerance-offset",
        dest="rotation_consistency_tolerance_offset",
        type=int,
        default=None,
        help=(
            "Keep only parts whose best rotation is within this offset from the anchor, "
            "and count rotations within this offset as matching. Overrides --eliminate."
        ),
    )
    parser.add_argument("--max-id", dest="max_identities", type=int, default=None)
    parser.add_argument(
        "--max-img-per-id",
        dest="max_images_per_identity",
        type=int,
        default=None,
    )
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument(
        "--filters",
        dest="filters",
        default=None,
        help="Optional Python filters file containing a 'filters' list. Defaults to project filters.py.",
    )
    parser.add_argument(
        "--output-name",
        default="benchmark_results.json",
        help="Filename for the JSON results inside the default output directory, or an absolute JSON path.",
    )
    parser.set_defaults(
        gt_manifest_root=str(DEFAULT_GT_MANIFEST_ROOT),
        gt_annotation_root=str(DEFAULT_GT_ANNOTATION_ROOT),
        rotation_consistency_threshold=0.3955,
    )
    args = parser.parse_args()

    if args.rotation < 1:
        raise ValueError("--rotation must be at least 1")
    if args.rotation_step < 1:
        raise ValueError("--rotation-step must be at least 1")
    rotation_offsets = build_rotation_offsets(args.rotation, args.rotation_step)
    rotation_consistency_parts = args.rotation_consistency_parts or 5
    if rotation_consistency_parts < 1:
        raise ValueError("--parts must be at least 1")
    if args.rotation_consistency_eliminate < 0:
        raise ValueError("--eliminate cannot be negative")
    if args.rotation_consistency_tolerance_offset is not None and args.rotation_consistency_tolerance_offset < 0:
        raise ValueError("--tolerance-offset cannot be negative")
    rotation_consistency_score = args.rotation_consistency_score or "hd"
    if rotation_consistency_score == "hd" and args.rotation_consistency_match_parts is not None:
        raise ValueError("--match-parts only applies when --score match-rotation")
    if args.rotation_consistency_match_parts is not None and args.rotation_consistency_match_parts < 1:
        raise ValueError("--match-parts must be at least 1")
    if (
        args.rotation_consistency_match_parts is not None
        and args.rotation_consistency_match_parts > rotation_consistency_parts
    ):
        raise ValueError("--match-parts cannot be larger than --parts")
    if (
        args.rotation_consistency_match_parts is not None
        and args.rotation_consistency_tolerance_offset is None
        and args.rotation_consistency_match_parts
        > max(1, rotation_consistency_parts - args.rotation_consistency_eliminate)
    ):
        raise ValueError("--match-parts cannot be larger than the maximum kept parts after --eliminate")
    if args.max_identities is not None and args.max_identities < 1:
        raise ValueError("--max-id must be at least 1")
    if args.max_images_per_identity is not None and args.max_images_per_identity < 1:
        raise ValueError("--max-img-per-id must be at least 1")
    include_rotation_consistency_classifier = (
        env_flag("ROTATION_CONSISTENCY_CLASSIFIER")
        or args.rotation_consistency_parts is not None
        or args.rotation_consistency_score is not None
        or args.rotation_consistency_tolerance_offset is not None
        or args.rotation_consistency_match_parts is not None
    )
    gt_manifest_root = Path(args.gt_manifest_root).expanduser().resolve()
    gt_annotation_root = Path(args.gt_annotation_root).expanduser().resolve()
    gt_manifest_paths = {}
    if args.gtmask:
        for dataset_format in args.datasets:
            manifest_path, _lookup = load_gt_manifest(dataset_format, manifest_root=gt_manifest_root)
            annotation_dir = GT_ANNOTATION_DIRS.get(dataset_format)
            if annotation_dir is None:
                supported = ", ".join(sorted(GT_ANNOTATION_DIRS))
                raise ValueError(f"--gt-mask is not available for {dataset_format}. Available GT-mask datasets: {supported}")
            expected_annotation_dir = gt_annotation_root / annotation_dir
            if not expected_annotation_dir.exists():
                raise FileNotFoundError(
                    f"--gt-mask requested, but GT annotation directory was not found: {expected_annotation_dir}"
                )
            gt_manifest_paths[dataset_format] = str(manifest_path)

    output_name = Path(args.output_name).expanduser()
    if output_name.is_absolute():
        output_path = output_name.resolve()
        output_dir = output_path.parent
    else:
        output_dir = DEFAULT_OUTPUT_DIR.expanduser().resolve()
        output_path = output_dir / output_name
    output_dir.mkdir(parents=True, exist_ok=True)
    selected_filters, filters_source = load_filter_bank(args.filters)
    iriscode_bits = compute_iriscode_bit_count(selected_filters)
    print(f"Filters in use: {len(selected_filters)}")
    print(f"Filters source: {filters_source}")
    print(f"Total iriscode bits from filters: {iriscode_bits}")
    results = {
        "filters_count": int(len(selected_filters)),
        "filters_source": filters_source,
        "iriscode_bits": iriscode_bits,
        "matcher": MATCHER_IRISCODE,
        "feature_extractor": "gabor",
        "segmentation_backend": get_segmentation_backend_name(),
        "segmentation_model_path": str(UNET_ONNX_PATH),
        "segmentation_source": "gtmask" if args.gtmask else "onnx",
        "gtmask": {
            "enabled": bool(args.gtmask),
            "manifest_root": str(gt_manifest_root) if args.gtmask else None,
            "annotation_root": str(gt_annotation_root) if args.gtmask else None,
            "manifests": gt_manifest_paths,
        },
        "rotation": int(args.rotation),
        "rotation_step": int(args.rotation_step),
        "rotation_offsets": [int(offset) for offset in rotation_offsets],
        "pairwise": [],
        "rotation_consistency_classifier": {
            "enabled": bool(include_rotation_consistency_classifier),
            "enabled_by_env": bool(env_flag("ROTATION_CONSISTENCY_CLASSIFIER")),
            "parts": int(rotation_consistency_parts),
            "score": rotation_consistency_score,
            "threshold": float(args.rotation_consistency_threshold),
            "eliminate": (
                None
                if args.rotation_consistency_tolerance_offset is not None
                else int(args.rotation_consistency_eliminate)
            ),
            "match_parts": (
                int(args.rotation_consistency_match_parts)
                if rotation_consistency_score == "match-rotation"
                and args.rotation_consistency_match_parts is not None
                else None
            ),
            "prediction_mode": (
                "rotation_match_count" if rotation_consistency_score == "match-rotation" else "hd_threshold"
            ),
            "tolerance_offset": (
                None
                if args.rotation_consistency_tolerance_offset is None
                else int(args.rotation_consistency_tolerance_offset)
            ),
        },
    }

    for dataset_format in args.datasets:
        dataset_path, dataset_format = resolve_dataset(None, dataset_format)
        print(f"Evaluating dataset: {dataset_format}")
        band_getter = None
        segmentation_source = "onnx"
        if args.gtmask:
            manifest_path, band_getter = build_gt_band_getter(
                dataset_format,
                manifest_root=gt_manifest_root,
                annotation_root=gt_annotation_root,
            )
            segmentation_source = f"gtmask:{manifest_path}"
            print(f"  using GT masks: {manifest_path}")
        pairwise_result = run_pairwise_benchmark(
            dataset_path,
            dataset_format,
            args.rotation,
            rotation_offsets,
            output_dir,
            selected_filters,
            filters_source,
            include_rotation_consistency=include_rotation_consistency_classifier,
            rotation_consistency_parts=rotation_consistency_parts,
            rotation_consistency_threshold=args.rotation_consistency_threshold,
            rotation_consistency_eliminate=args.rotation_consistency_eliminate,
            rotation_consistency_tolerance_offset=args.rotation_consistency_tolerance_offset,
            rotation_consistency_score=rotation_consistency_score,
            rotation_consistency_match_parts=args.rotation_consistency_match_parts,
            fixed_threshold=args.threshold,
            max_identities=args.max_identities,
            max_images_per_identity=args.max_images_per_identity,
            seed=args.seed,
            band_getter=band_getter,
            segmentation_source=segmentation_source,
        )
        results["pairwise"].append(pairwise_result)
        print(f"  pairwise: AUC={pairwise_result['roc_auc']:.4f} EER={pairwise_result['eer']:.4f}")
        fixed_threshold = pairwise_result["fixed_threshold"]
        if fixed_threshold is not None:
            print(
                "  fixed threshold: "
                f"T={fixed_threshold['threshold']:.6f} "
                f"mated classified as non-mated (FNMR/FNR)="
                f"{fixed_threshold['mated_wrongly_classified_non_mated']}/{fixed_threshold['mated_total']} "
                f"rate={fixed_threshold['false_reject_rate']:.4f}; "
                f"non-mated classified as mated (FMR/FAR)="
                f"{fixed_threshold['non_mated_wrongly_classified_mated']}/{fixed_threshold['non_mated_total']} "
                f"rate={fixed_threshold['false_accept_rate']:.4f}; "
                f"TPR={fixed_threshold['tpr']:.4f}"
            )
        rotation_result = pairwise_result["rotation_consistency_classifier"]
        if rotation_result is not None:
            rotation_summary = rotation_result["summary"]
            print(
                "  rotation consistency classifier: "
                f"AUC={rotation_summary['roc_auc']:.4f} "
                f"EER={rotation_summary['eer']:.4f} "
                f"ACC={rotation_summary['accuracy']:.4f}"
            )
        output_path.write_text(json.dumps(format_result(results), indent=2))

    output_path.write_text(json.dumps(format_result(results), indent=2))
    print(f"Saved pipeline results to {output_path}")


if __name__ == "__main__":
    main()
