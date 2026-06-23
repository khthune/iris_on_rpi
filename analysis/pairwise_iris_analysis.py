# pairwise_iris_analysis.py

from argparse import ArgumentParser
from itertools import combinations
import os
from pathlib import Path
import shutil
import sys
import time

import numpy as np
from sklearn.metrics import auc, roc_curve

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
ANALYSIS_ROOT = Path(__file__).resolve().parent
MATPLOTLIB_CONFIG_DIR = ANALYSIS_ROOT / "output" / "pairwise_iris_analysis" / "matplotlib"
MATPLOTLIB_CONFIG_DIR.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(MATPLOTLIB_CONFIG_DIR))
XDG_CACHE_DIR = ANALYSIS_ROOT / "output" / "pairwise_iris_analysis" / "cache"
XDG_CACHE_DIR.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("XDG_CACHE_HOME", str(XDG_CACHE_DIR))

from dataset_loaders import DATASET_CHOICES, dataset_output_slug, load_dataset, resolve_dataset, sample_dataset
from filter_loader import load_filter_bank
from iris import IrisClassifier as CurrentIrisClassifier, get_iris_band
from rotation_part_scoring import compute_pairwise_rotation_classifier

import matplotlib

if matplotlib.get_backend().lower() == "agg" or __name__ == "__main__":
    matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns


DEFAULT_DATASET_FORMAT = "auto"
DEFAULT_OUTPUT_DIR = Path(__file__).resolve().parent / "output" / "pairwise_iris_analysis"
MATCHER_IRISCODE = "iriscode"


def load_iris_classifier_class(iris_engine):
    if iris_engine == "current":
        return CurrentIrisClassifier
    if iris_engine == "legacy":
        from legacy_iris import IrisClassifier as LegacyIrisClassifier

        return LegacyIrisClassifier
    raise ValueError(f"Unknown iris engine: {iris_engine}")


def build_wahet_band_getter():
    local_wahet = ANALYSIS_ROOT / "wahet"
    wahet_executable = local_wahet if local_wahet.exists() else shutil.which("wahet")
    if wahet_executable is None:
        raise FileNotFoundError(
            f"--segmenter wahet requested, but WAHET was not found at {local_wahet} or in PATH."
        )
    from legacy_iris import get_iris_band as get_wahet_iris_band

    def get_wahet_band(image, _image_name):
        return get_wahet_iris_band(image, wahet_executable=wahet_executable)

    return get_wahet_band


def add_figure_metadata(figure, metadata):
    if not metadata:
        return
    text = " | ".join(f"{key}={value}" for key, value in metadata.items() if value is not None)
    if text:
        figure.text(0.01, 0.01, text, ha="left", va="bottom", fontsize=7, family="monospace", wrap=True)


def precompute_codes(
    images,
    labels,
    image_names,
    classifier,
    rotation,
    band_getter=None,
    offsets=None,
    min_valid_iris_pixels=None,
    rotation_method="recompute",
):
    if band_getter is None:
        band_getter = lambda image, _image_name: get_iris_band(image)
    if offsets is None:
        offsets = np.arange(rotation) - rotation // 2
    else:
        offsets = np.asarray(offsets, dtype=np.int64)
        if offsets.size == 0:
            raise ValueError("offsets cannot be empty")
    sample_count = len(images)

    base_codes = []
    base_masks = []
    rotated_codes = []
    rotated_masks = []
    kept_labels = []
    kept_image_names = []
    skipped = []
    for index, image in enumerate(images, start=1):
        if index == 1 or index % 250 == 0 or index == sample_count:
            print(f"Precomputing iris codes: {index}/{sample_count}")

        try:
            iris_band, iris_mask = band_getter(image, image_names[index - 1])
        except Exception as exc:
            skipped.append((index - 1, str(image_names[index - 1]), str(exc)))
            continue
        if iris_band is None or iris_mask is None:
            skipped.append((index - 1, str(image_names[index - 1]), "segmentation returned None"))
            continue
        base_code, base_mask, _ = classifier.get_iris_code(iris_band, iris_mask, offset=0)
        base_mask = np.asarray(base_mask, dtype=bool)
        if min_valid_iris_pixels is not None:
            valid_iriscode_bits = int(np.sum(base_mask))
            if valid_iriscode_bits < min_valid_iris_pixels:
                skipped.append(
                    (
                        index - 1,
                        str(image_names[index - 1]),
                        f"valid iriscode bits {valid_iriscode_bits} < {min_valid_iris_pixels}",
                    )
                )
                continue

        base_codes.append(np.asarray(base_code, dtype=bool))
        base_masks.append(base_mask)
        kept_labels.append(labels[index - 1])
        kept_image_names.append(image_names[index - 1])

        if rotation_method == "recompute":
            image_rotated_codes = []
            image_rotated_masks = []
            for offset in offsets:
                code, code_mask, _ = classifier.get_iris_code(iris_band, iris_mask, offset=int(offset))
                image_rotated_codes.append(np.asarray(code, dtype=bool))
                image_rotated_masks.append(np.asarray(code_mask, dtype=bool))
            rotated_codes.append(np.stack(image_rotated_codes, axis=0))
            rotated_masks.append(np.stack(image_rotated_masks, axis=0))
        elif rotation_method == "roll":
            if not hasattr(classifier, "_roll_iris_code_offsets"):
                raise ValueError("--rotation-method roll is only available for the current iris engine.")
            rolled_codes, rolled_masks = classifier._roll_iris_code_offsets(
                np.asarray(base_code, dtype=bool),
                base_mask,
                iris_band.shape,
                offsets,
            )
            rotated_codes.append(rolled_codes)
            rotated_masks.append(rolled_masks)
        else:
            raise ValueError(f"Unknown rotation method: {rotation_method}")

    if not base_codes:
        raise RuntimeError("Segmentation failed for every sampled image.")

    return (
        np.stack(base_codes, axis=0),
        np.stack(base_masks, axis=0),
        np.stack(rotated_codes, axis=0),
        np.stack(rotated_masks, axis=0),
        offsets,
        np.array(kept_labels),
        np.array(kept_image_names),
        skipped,
    )


def best_score_against_rotations(base_code, base_mask, candidate_codes, candidate_masks):
    diff = np.bitwise_xor(candidate_codes, base_code)
    combined_mask = np.bitwise_and(candidate_masks, base_mask)
    valid_bits = np.sum(combined_mask, axis=1)
    mismatch_bits = np.sum(np.bitwise_and(diff, combined_mask), axis=1)

    scores = np.full(candidate_codes.shape[0], 2.0, dtype=np.float64)
    valid_rows = valid_bits > 0
    scores[valid_rows] = mismatch_bits[valid_rows] / valid_bits[valid_rows]

    best_index = int(np.argmin(scores))
    return float(scores[best_index]), best_index, int(valid_bits[best_index])


def compute_pairwise_scores_iriscode(
    labels,
    base_codes,
    base_masks,
    rotated_codes,
    rotated_masks,
    offsets,
    min_valid_bits=None,
):
    idx1_list = []
    idx2_list = []
    score_list = []
    same_class_list = []
    best_offset_list = []
    direction_list = []
    valid_bits_list = []
    skipped_low_valid_bits = 0

    pair_count = len(labels) * (len(labels) - 1) // 2
    started = time.perf_counter()

    for pair_index, (idx1, idx2) in enumerate(combinations(range(len(labels)), 2), start=1):
        score_12, offset_index_12, valid_bits_12 = best_score_against_rotations(
            base_codes[idx1],
            base_masks[idx1],
            rotated_codes[idx2],
            rotated_masks[idx2],
        )
        score_21, offset_index_21, valid_bits_21 = best_score_against_rotations(
            base_codes[idx2],
            base_masks[idx2],
            rotated_codes[idx1],
            rotated_masks[idx1],
        )

        if score_12 <= score_21:
            best_score = score_12
            best_offset = int(offsets[offset_index_12])
            direction = 1
            valid_bits = valid_bits_12
        else:
            best_score = score_21
            best_offset = int(offsets[offset_index_21])
            direction = -1
            valid_bits = valid_bits_21

        if min_valid_bits is not None and valid_bits < min_valid_bits:
            skipped_low_valid_bits += 1
            if pair_index == 1 or pair_index % 25000 == 0 or pair_index == pair_count:
                elapsed = time.perf_counter() - started
                print(
                    f"Scored pairs: {pair_index}/{pair_count} in {elapsed:.1f}s "
                    f"(skipped low valid bits: {skipped_low_valid_bits})"
                )
            continue

        idx1_list.append(idx1)
        idx2_list.append(idx2)
        score_list.append(best_score)
        same_class_list.append(labels[idx1] == labels[idx2])
        best_offset_list.append(best_offset)
        direction_list.append(direction)
        valid_bits_list.append(valid_bits)

        if pair_index == 1 or pair_index % 25000 == 0 or pair_index == pair_count:
            elapsed = time.perf_counter() - started
            suffix = ""
            if min_valid_bits is not None:
                suffix = f" (skipped low valid bits: {skipped_low_valid_bits})"
            print(f"Scored pairs: {pair_index}/{pair_count} in {elapsed:.1f}s{suffix}")

    return {
        "idx1": np.array(idx1_list, dtype=np.int32),
        "idx2": np.array(idx2_list, dtype=np.int32),
        "scores": np.array(score_list, dtype=np.float32),
        "same_class": np.array(same_class_list, dtype=bool),
        "best_offset": np.array(best_offset_list, dtype=np.int16),
        "direction": np.array(direction_list, dtype=np.int8),
        "valid_bits": np.array(valid_bits_list, dtype=np.int32),
        "skipped_low_valid_bits": int(skipped_low_valid_bits),
        "total_candidate_pairs": int(pair_count),
    }


compute_pairwise_scores = compute_pairwise_scores_iriscode


def summarize_label_pairs(labels):
    unique_labels, counts = np.unique(labels, return_counts=True)
    sample_count = int(len(labels))
    class_count = int(len(unique_labels))
    total_pairs = sample_count * (sample_count - 1) // 2
    mated_pairs = int(sum(count * (count - 1) // 2 for count in counts))
    non_mated_pairs = int(total_pairs - mated_pairs)

    return {
        "sample_count": sample_count,
        "class_count": class_count,
        "total_pairs": total_pairs,
        "mated_pairs": mated_pairs,
        "non_mated_pairs": non_mated_pairs,
    }


def evaluate_scores(same_class, scores):
    same_class = np.asarray(same_class, dtype=bool)
    fpr, tpr, thresholds = roc_curve(same_class, -scores)
    fnr = 1.0 - tpr
    eer_index = int(np.nanargmin(np.abs(fnr - fpr)))
    eer = float((fpr[eer_index] + fnr[eer_index]) / 2.0)
    roc_auc = float(auc(fpr, tpr))
    mated_total = int(np.sum(same_class))
    non_mated_total = int(np.sum(~same_class))
    eer_fpr = float(fpr[eer_index])
    eer_fnr = float(fnr[eer_index])
    fpr_std = 0.0 if non_mated_total == 0 else np.sqrt(eer_fpr * (1.0 - eer_fpr) / non_mated_total)
    fnr_std = 0.0 if mated_total == 0 else np.sqrt(eer_fnr * (1.0 - eer_fnr) / mated_total)
    eer_std = float(0.5 * np.sqrt((fpr_std ** 2) + (fnr_std ** 2)))

    return {
        "fpr": fpr,
        "tpr": tpr,
        "thresholds": thresholds,
        "eer": eer,
        "eer_std": eer_std,
        "eer_fpr": eer_fpr,
        "eer_fnr": eer_fnr,
        "eer_threshold": float(thresholds[eer_index]),
        "roc_auc": roc_auc,
    }


def evaluate_zero_false_accept_threshold(same_class, scores, lower_is_mated=True):
    same_class = np.asarray(same_class, dtype=bool)
    scores = np.asarray(scores, dtype=np.float64)
    mated_scores = scores[same_class]
    non_mated_scores = scores[~same_class]
    if mated_scores.size == 0 or non_mated_scores.size == 0:
        return {
            "threshold": None,
            "boundary_threshold": None,
            "tpr": None,
            "genuine_accept_rate": None,
            "false_reject_rate": None,
            "false_accept_rate": None,
            "mated_accepted": 0,
            "mated_total": int(mated_scores.size),
            "non_mated_total": int(non_mated_scores.size),
            "rule": "score <= threshold" if lower_is_mated else "score >= threshold",
        }

    if lower_is_mated:
        boundary_threshold = float(np.min(non_mated_scores))
        threshold = float(np.nextafter(boundary_threshold, -np.inf))
        accepted_mated = mated_scores <= threshold
        accepted_non_mated = non_mated_scores <= threshold
        rule = "score <= threshold"
    else:
        boundary_threshold = float(np.max(non_mated_scores))
        threshold = float(np.nextafter(boundary_threshold, np.inf))
        accepted_mated = mated_scores >= threshold
        accepted_non_mated = non_mated_scores >= threshold
        rule = "score >= threshold"

    tpr = float(np.mean(accepted_mated))
    far = float(np.mean(accepted_non_mated))
    return {
        "threshold": threshold,
        "boundary_threshold": boundary_threshold,
        "tpr": tpr,
        "genuine_accept_rate": tpr,
        "false_reject_rate": float(1.0 - tpr),
        "false_accept_rate": far,
        "mated_accepted": int(np.sum(accepted_mated)),
        "mated_total": int(mated_scores.size),
        "non_mated_total": int(non_mated_scores.size),
        "rule": rule,
    }


def evaluate_fmr_threshold(same_class, scores, target_fmr, lower_is_mated=True):
    same_class = np.asarray(same_class, dtype=bool)
    scores = np.asarray(scores, dtype=np.float64)
    mated_scores = scores[same_class]
    non_mated_scores = scores[~same_class]
    if mated_scores.size == 0 or non_mated_scores.size == 0:
        return {
            "threshold": None,
            "target_fmr": float(target_fmr),
            "actual_fmr": None,
            "fnmr": None,
            "mated_accepted": 0,
            "mated_total": int(mated_scores.size),
            "non_mated_accepted": 0,
            "non_mated_total": int(non_mated_scores.size),
            "rule": "score <= threshold" if lower_is_mated else "score >= threshold",
        }

    target_fmr = float(target_fmr)
    if not 0.0 <= target_fmr <= 1.0:
        raise ValueError("target_fmr must be between 0 and 1")

    non_mated_total = int(non_mated_scores.size)
    allowed_false_accepts = int(np.floor(target_fmr * non_mated_total))
    if lower_is_mated:
        sorted_non_mated = np.sort(non_mated_scores)
        if allowed_false_accepts <= 0:
            threshold = float(np.nextafter(sorted_non_mated[0], -np.inf))
        elif allowed_false_accepts >= non_mated_total:
            threshold = float(np.inf)
        else:
            threshold = float(np.nextafter(sorted_non_mated[allowed_false_accepts], -np.inf))
        accepted_mated = mated_scores <= threshold
        accepted_non_mated = non_mated_scores <= threshold
        rule = "score <= threshold"
    else:
        sorted_non_mated = np.sort(non_mated_scores)[::-1]
        if allowed_false_accepts <= 0:
            threshold = float(np.nextafter(sorted_non_mated[0], np.inf))
        elif allowed_false_accepts >= non_mated_total:
            threshold = float(-np.inf)
        else:
            threshold = float(np.nextafter(sorted_non_mated[allowed_false_accepts], np.inf))
        accepted_mated = mated_scores >= threshold
        accepted_non_mated = non_mated_scores >= threshold
        rule = "score >= threshold"

    non_mated_accepted = int(np.sum(accepted_non_mated))
    mated_accepted = int(np.sum(accepted_mated))
    mated_total = int(mated_scores.size)

    return {
        "threshold": threshold,
        "target_fmr": target_fmr,
        "actual_fmr": float(non_mated_accepted / non_mated_total),
        "fnmr": float(1.0 - (mated_accepted / mated_total)),
        "mated_accepted": mated_accepted,
        "mated_total": mated_total,
        "non_mated_accepted": non_mated_accepted,
        "non_mated_total": non_mated_total,
        "allowed_false_accepts": allowed_false_accepts,
        "rule": rule,
    }


def evaluate_zero_false_reject_threshold(same_class, scores, lower_is_mated=True):
    same_class = np.asarray(same_class, dtype=bool)
    scores = np.asarray(scores, dtype=np.float64)
    mated_scores = scores[same_class]
    non_mated_scores = scores[~same_class]
    if mated_scores.size == 0 or non_mated_scores.size == 0:
        return {
            "threshold": None,
            "boundary_threshold": None,
            "non_mated_rejected_rate": None,
            "false_accept_rate": None,
            "false_reject_rate": None,
            "non_mated_rejected": 0,
            "mated_total": int(mated_scores.size),
            "non_mated_total": int(non_mated_scores.size),
            "rule": "score <= threshold" if lower_is_mated else "score >= threshold",
        }

    if lower_is_mated:
        boundary_threshold = float(np.max(mated_scores))
        threshold = float(np.nextafter(boundary_threshold, np.inf))
        accepted_mated = mated_scores <= threshold
        accepted_non_mated = non_mated_scores <= threshold
        rule = "score <= threshold"
    else:
        boundary_threshold = float(np.min(mated_scores))
        threshold = float(np.nextafter(boundary_threshold, -np.inf))
        accepted_mated = mated_scores >= threshold
        accepted_non_mated = non_mated_scores >= threshold
        rule = "score >= threshold"

    non_mated_rejected = ~accepted_non_mated
    return {
        "threshold": threshold,
        "boundary_threshold": boundary_threshold,
        "non_mated_rejected_rate": float(np.mean(non_mated_rejected)),
        "false_accept_rate": float(np.mean(accepted_non_mated)),
        "false_reject_rate": float(1.0 - np.mean(accepted_mated)),
        "non_mated_rejected": int(np.sum(non_mated_rejected)),
        "mated_total": int(mated_scores.size),
        "non_mated_total": int(non_mated_scores.size),
        "rule": rule,
    }


def plot_results(
    scores,
    same_class,
    evaluation,
    zero_false_accept=None,
    zero_false_reject=None,
    figure_path=None,
    matcher=MATCHER_IRISCODE,
    metadata=None,
    scale="linear",
):
    mated_scores = scores[same_class]
    non_mated_scores = scores[~same_class]
    fpr = evaluation["fpr"]
    distribution_title = "Hamming Distance Distribution"
    x_label = "Hamming Distance"
    x_limit = (0.0, 0.6)

    sns.set_theme(style="whitegrid")
    figure, axes = plt.subplots(1, 2, figsize=(14, 6))

    if scale == "log":
        bins = np.linspace(x_limit[0], x_limit[1], 80)
        axes[0].hist(
            mated_scores,
            bins=bins,
            label="Mated",
            color="#3b5bff",
            alpha=0.45,
            log=True,
        )
        axes[0].hist(
            non_mated_scores,
            bins=bins,
            label="Non-Mated",
            color="#ff4d4f",
            alpha=0.45,
            log=True,
        )
    else:
        sns.kdeplot(mated_scores, ax=axes[0], label="Mated", color="#3b5bff", fill=True, alpha=0.55)
        sns.kdeplot(
            non_mated_scores,
            ax=axes[0],
            label="Non-Mated",
            color="#ff4d4f",
            fill=True,
            alpha=0.55,
        )
    axes[0].set_title(distribution_title)
    axes[0].set_xlabel(x_label)
    if scale == "log":
        axes[0].set_ylabel("Pair Count (log scale)")
        axes[0].set_ylim(bottom=0.8)
    else:
        axes[0].set_ylabel("Density")
    if x_limit is not None:
        axes[0].set_xlim(*x_limit)
    if zero_false_accept and zero_false_accept.get("threshold") is not None:
        axes[0].axvline(
            zero_false_accept["threshold"],
            color="#111111",
            linestyle="--",
            linewidth=1.4,
            label=(
                "Zero false accepts "
                f"(T={zero_false_accept['threshold']:.4f})"
            ),
        )
    if zero_false_reject and zero_false_reject.get("threshold") is not None:
        axes[0].axvline(
            zero_false_reject["threshold"],
            color="#00897b",
            linestyle=":",
            linewidth=1.8,
            label=(
                "Zero false rejects "
                f"(T={zero_false_reject['threshold']:.4f})"
            ),
        )
    axes[0].legend(loc="upper right")

    plot_fpr = fpr
    chance_fpr = np.linspace(0.0, 1.0, 200)
    x_label_roc = "False Positive Rate"

    axes[1].plot(
        plot_fpr,
        evaluation["tpr"],
        color="#ff8c00",
        lw=2,
        label=f"ROC (AUC = {evaluation['roc_auc']:.4f}, EER = {evaluation['eer']:.4f})",
    )
    axes[1].plot(chance_fpr, chance_fpr, linestyle="--", color="#6c757d", lw=1)
    axes[1].set_title("ROC Curve")
    axes[1].set_xlabel(x_label_roc)
    axes[1].set_ylabel("True Positive Rate")
    axes[1].set_xlim(0.0, 1.0)
    axes[1].set_ylim(0.0, 1.0)
    axes[1].legend(loc="lower right")

    add_figure_metadata(figure, metadata or {})
    figure.tight_layout(rect=(0, 0.04, 1, 1))

    if figure_path:
        output = Path(figure_path).expanduser().resolve()
        output.parent.mkdir(parents=True, exist_ok=True)
        figure.savefig(output, dpi=300, bbox_inches="tight")

    plt.close(figure)


def main():
    parser = ArgumentParser(
        description="Compute pairwise iris comparison scores and plot distribution/ROC/EER."
    )
    parser.add_argument(
        "--dataset-path",
        help="Path to the dataset directory. If omitted, a known default path is used.",
    )
    parser.add_argument(
        "--dataset",
        dest="dataset_format",
        default=DEFAULT_DATASET_FORMAT,
        choices=DATASET_CHOICES,
        help="Dataset folder layout to load",
    )
    parser.add_argument(
        "--output-name",
        dest="output_name",
        default=None,
        help="Output filename for the figure inside the default output directory. Example: my_run.png",
    )
    parser.add_argument(
        "--rotation",
        type=int,
        default=21,
        help="Number of horizontal offsets to evaluate around zero",
    )
    parser.add_argument(
        "--filters",
        dest="filters",
        default=None,
        help="Optional Python filters file containing a 'filters' list. Defaults to project filters.py.",
    )
    parser.add_argument(
        "--iris-engine",
        choices=["current", "legacy"],
        default="current",
        help="Recognition engine to use for iriscode generation and comparison.",
    )
    parser.add_argument(
        "--segmenter",
        choices=["unet", "wahet"],
        default="unet",
        help="Segmentation/normalization method to use.",
    )
    parser.add_argument(
        "--parts",
        type=int,
        default=None,
        help="Use part-split average HD with this many iriscode parts. If omitted, use normal whole-iriscode HD.",
    )
    parser.add_argument(
        "--eliminate",
        type=int,
        default=0,
        help="For --parts, remove this many parts furthest from the lowest-HD part's rotation. Default: 0.",
    )
    parser.add_argument(
        "--score",
        choices=["hd"],
        default="hd",
        help="Part-split score. Currently only hd is supported here.",
    )
    parser.add_argument(
        "--max-id",
        dest="max_identities",
        type=int,
        default=None,
        help="Randomly sample at most this many identities.",
    )
    parser.add_argument(
        "--max-img-per-id",
        dest="max_images_per_identity",
        type=int,
        default=20,
        help="Randomly sample at most this many images per identity.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Random seed for deterministic subset sampling.",
    )
    parser.add_argument(
        "--scale",
        choices=["linear", "log"],
        default="linear",
        help="HD distribution scale. Use log for a logarithmic count histogram.",
    )
    parser.add_argument(
        "--fmr",
        type=float,
        default=1e-4,
        help="Target false match rate for printing an operating threshold. Default: 1e-4.",
    )
    args = parser.parse_args()

    if args.rotation < 1:
        raise ValueError("--rotation must be at least 1")
    if not 0.0 <= args.fmr <= 1.0:
        raise ValueError("--fmr must be between 0 and 1")
    if args.parts is not None and args.parts < 1:
        raise ValueError("--parts must be at least 1")
    if args.eliminate < 0:
        raise ValueError("--eliminate cannot be negative")
    if args.parts is not None and args.eliminate >= args.parts:
        raise ValueError("--eliminate must be smaller than --parts")

    dataset_path, dataset_format = resolve_dataset(args.dataset_path, args.dataset_format)
    dataset_name = dataset_output_slug(dataset_format)
    figure_output = None
    if args.output_name is not None:
        output_name = args.output_name
        if Path(output_name).suffix == "":
            output_name = f"{output_name}.png"
        figure_output = str(DEFAULT_OUTPUT_DIR / output_name)
    if figure_output is None:
        default_name = f"fltr_ana_{dataset_name}.png"
        figure_output = str(DEFAULT_OUTPUT_DIR / default_name)

    print(f"Using dataset format: {dataset_format}")
    print(f"Dataset name: {dataset_name}")
    print(f"Using dataset path: {dataset_path}")

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
    pre_summary = summarize_label_pairs(labels)
    print(f"Samples: {pre_summary['sample_count']}")
    print(f"Classes: {pre_summary['class_count']}")
    print(f"Total unordered pairs: {pre_summary['total_pairs']}")
    print(f"Mated pairs: {pre_summary['mated_pairs']}")
    print(f"Non-mated pairs: {pre_summary['non_mated_pairs']}")
    if pre_summary["mated_pairs"] == 0 or pre_summary["non_mated_pairs"] == 0:
        raise ValueError(
            "The sampled subset does not contain both mated and non-mated pairs. "
            "Use a larger subset and prefer --max-img-per-id 2 or more."
        )
    selected_filters, filters_source = load_filter_bank(args.filters)
    print(f"Filters in use: {len(selected_filters)}")
    print(f"Filters source: {filters_source}")
    print(f"Iris engine: {args.iris_engine}")
    print(f"Segmenter: {args.segmenter}")
    classifier_class = load_iris_classifier_class(args.iris_engine)
    classifier = classifier_class(selected_filters)
    band_getter = None
    if args.segmenter == "wahet":
        band_getter = build_wahet_band_getter()
    (
        base_codes,
        base_masks,
        rotated_codes,
        rotated_masks,
        offsets,
        labels,
        image_names,
        skipped,
    ) = precompute_codes(
        images,
        labels,
        image_names,
        classifier,
        args.rotation,
        band_getter=band_getter,
    )
    if skipped:
        print(f"Skipped {len(skipped)} images due to segmentation failure.")
        for skipped_index, skipped_name, reason in skipped[:5]:
            print(f"  skipped[{skipped_index}] {skipped_name}: {reason}")
        if len(skipped) > 5:
            print(f"  ... {len(skipped) - 5} more skipped images")

    summary = summarize_label_pairs(labels)
    print(f"Usable after segmentation: {summary['sample_count']}")
    print(f"Usable classes: {summary['class_count']}")
    print(f"Usable unordered pairs: {summary['total_pairs']}")
    print(f"Usable mated pairs: {summary['mated_pairs']}")
    print(f"Usable non-mated pairs: {summary['non_mated_pairs']}")
    if summary["mated_pairs"] == 0 or summary["non_mated_pairs"] == 0:
        raise ValueError(
            "After removing segmentation failures, the subset no longer contains both mated and non-mated pairs."
        )
    matcher_name = MATCHER_IRISCODE
    if args.parts is None:
        pairwise = compute_pairwise_scores_iriscode(
            labels,
            base_codes,
            base_masks,
            rotated_codes,
            rotated_masks,
            offsets,
        )
    else:
        print(
            "Using part-split average HD: "
            f"parts={args.parts} eliminate={args.eliminate}"
        )
        part_rows = compute_pairwise_rotation_classifier(
            labels,
            base_codes,
            base_masks,
            rotated_codes,
            rotated_masks,
            offsets,
            args.parts,
            0.3955,
            args.eliminate,
            None,
            1,
            None,
        )
        pairwise = {
            "scores": np.asarray([row["avg_hd"] for row in part_rows], dtype=np.float32),
            "same_class": np.asarray([row["same_class"] for row in part_rows], dtype=bool),
        }
        matcher_name = f"{MATCHER_IRISCODE}_parts{args.parts}_hd"
    evaluation = evaluate_scores(pairwise["same_class"], pairwise["scores"])
    zero_false_accept = evaluate_zero_false_accept_threshold(
        pairwise["same_class"],
        pairwise["scores"],
        lower_is_mated=True,
    )
    fmr_threshold = evaluate_fmr_threshold(
        pairwise["same_class"],
        pairwise["scores"],
        args.fmr,
        lower_is_mated=True,
    )
    zero_false_reject = evaluate_zero_false_reject_threshold(
        pairwise["same_class"],
        pairwise["scores"],
        lower_is_mated=True,
    )

    print(f"EER: {evaluation['eer']:.6f}")
    print(f"EER HD threshold: {-evaluation['eer_threshold']:.6f}")
    print(f"ROC AUC: {evaluation['roc_auc']:.6f}")
    if zero_false_accept["threshold"] is not None:
        print(
            "Zero non-mated classified as mated (FMR/FAR=0): "
            f"HD threshold={zero_false_accept['threshold']:.6f} "
            f"TPR={zero_false_accept['tpr']:.4f} "
            f"FNMR/FNR={zero_false_accept['false_reject_rate']:.4f} "
            f"({zero_false_accept['mated_accepted']}/{zero_false_accept['mated_total']} mated accepted)"
        )
    if fmr_threshold["threshold"] is not None:
        print(
            f"Target FMR/FAR={args.fmr:g}: "
            f"HD threshold={fmr_threshold['threshold']:.6f} "
            f"actual FMR/FAR={fmr_threshold['actual_fmr']:.8f} "
            f"FNMR/FNR={fmr_threshold['fnmr']:.4f} "
            f"({fmr_threshold['non_mated_accepted']}/{fmr_threshold['non_mated_total']} non-mated accepted, "
            f"{fmr_threshold['mated_accepted']}/{fmr_threshold['mated_total']} mated accepted)"
        )
    if zero_false_reject["threshold"] is not None:
        print(
            "Zero mated classified as non-mated (FNMR/FNR=0): "
            f"HD threshold={zero_false_reject['threshold']:.6f} "
            f"FMR/FAR={zero_false_reject['false_accept_rate']:.4f} "
            f"({zero_false_reject['non_mated_rejected']}/{zero_false_reject['non_mated_total']} non-mated rejected)"
        )

    plot_metadata = {
        "dataset": dataset_format,
        "dataset_path": dataset_path,
        "seg_path": os.environ.get("SEG_PATH"),
        "iris_engine": args.iris_engine,
        "segmenter": args.segmenter,
        "output_name": args.output_name,
        "rotation": args.rotation,
        "samples": summary["sample_count"],
        "classes": summary["class_count"],
        "mated_pairs": summary["mated_pairs"],
        "non_mated_pairs": summary["non_mated_pairs"],
        "max_identities": args.max_identities,
        "max_images_per_identity": args.max_images_per_identity,
        "seed": args.seed,
        "matcher": MATCHER_IRISCODE,
        "parts": args.parts,
        "eliminate": args.eliminate if args.parts is not None else None,
        "score": args.score if args.parts is not None else None,
        "target_fmr": args.fmr,
        "target_fmr_threshold": fmr_threshold["threshold"],
        "target_fmr_actual_fmr": fmr_threshold["actual_fmr"],
        "target_fmr_fnmr": fmr_threshold["fnmr"],
        "filter_count": len(selected_filters),
        "filters": filters_source,
        "skipped": len(skipped),
        "scale": args.scale,
    }
    plot_results(
        pairwise["scores"],
        pairwise["same_class"],
        evaluation,
        zero_false_accept=zero_false_accept,
        zero_false_reject=zero_false_reject,
        figure_path=figure_output,
        matcher=matcher_name,
        metadata=plot_metadata,
        scale=args.scale,
    )
    print(f"Saved analysis figure to {Path(figure_output).expanduser().resolve()}")


if __name__ == "__main__":
    main()
