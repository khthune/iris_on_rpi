from itertools import combinations
import time

import numpy as np
from sklearn.metrics import auc, roc_curve


def split_code_slices(code_length, parts):
    parts = int(parts)
    if parts < 1:
        raise ValueError("--parts must be at least 1")
    if parts > code_length:
        raise ValueError(f"--parts cannot be larger than iriscode length ({code_length})")

    boundaries = np.linspace(0, code_length, parts + 1, dtype=int)
    return [slice(int(boundaries[index]), int(boundaries[index + 1])) for index in range(parts)]


def best_score_against_rotations(base_code, base_mask, candidate_codes, candidate_masks, min_valid_bits=1):
    base_code = np.asarray(base_code, dtype=bool)
    base_mask = np.asarray(base_mask, dtype=bool)
    candidate_codes = np.asarray(candidate_codes, dtype=bool)
    candidate_masks = np.asarray(candidate_masks, dtype=bool)

    diff = np.bitwise_xor(candidate_codes, base_code)
    combined_mask = np.bitwise_and(candidate_masks, base_mask)
    valid_bits = np.sum(combined_mask, axis=1)
    mismatch_bits = np.sum(np.bitwise_and(diff, combined_mask), axis=1)

    scores = np.full(candidate_codes.shape[0], np.nan, dtype=np.float64)
    valid = valid_bits >= int(min_valid_bits)
    scores[valid] = mismatch_bits[valid] / valid_bits[valid]
    if not np.any(np.isfinite(scores)):
        return float("inf"), 0

    best_score = float(np.nanmin(scores))
    tied_indices = np.flatnonzero(np.isclose(scores, best_score, rtol=0.0, atol=0.0))
    return best_score, int(tied_indices[0])


def part_scores_for_offsets(base_code, base_mask, candidate_codes, candidate_masks, offsets, slices, min_valid_bits=1):
    offsets = np.asarray(offsets)
    part_scores = np.empty(len(slices), dtype=np.float64)
    part_offsets = np.empty(len(slices), dtype=np.int16)

    for part_index, code_slice in enumerate(slices):
        score, offset_index = best_score_against_rotations(
            np.asarray(base_code, dtype=bool)[code_slice],
            np.asarray(base_mask, dtype=bool)[code_slice],
            np.asarray(candidate_codes, dtype=bool)[:, code_slice],
            np.asarray(candidate_masks, dtype=bool)[:, code_slice],
            min_valid_bits=min_valid_bits,
        )
        part_scores[part_index] = score
        part_offsets[part_index] = int(offsets[offset_index]) if offsets.size else 0

    return part_scores, part_offsets


def select_parts(part_scores, part_offsets, eliminate=0, tolerance_offset=None):
    part_scores = np.asarray(part_scores, dtype=np.float64)
    part_offsets = np.asarray(part_offsets, dtype=np.int16)
    if part_scores.size == 0:
        raise ValueError("part_scores cannot be empty")
    if part_scores.shape != part_offsets.shape:
        raise ValueError("part_scores and part_offsets must have the same shape")

    finite = np.isfinite(part_scores)
    if not np.any(finite):
        return {
            "avg_hd": float("inf"),
            "anchor_offset": 0,
            "selected_indices": [],
            "selected_offsets": [],
            "selected_scores": [],
            "rotation_match_count": 0,
            "kept_parts": 0,
        }

    finite_indices = np.flatnonzero(finite)
    finite_scores = part_scores[finite_indices]
    best_score = float(np.min(finite_scores))
    tied = finite_indices[np.isclose(part_scores[finite_indices], best_score, rtol=0.0, atol=0.0)]
    anchor_index = int(tied[np.argmin(np.abs(part_offsets[tied]))])
    anchor_offset = int(part_offsets[anchor_index])

    if tolerance_offset is not None:
        tolerance = int(tolerance_offset)
        selected_indices = finite_indices[np.abs(part_offsets[finite_indices] - anchor_offset) <= tolerance]
    else:
        selected_indices = finite_indices.copy()
        eliminate = int(eliminate)
        if eliminate < 0:
            raise ValueError("--eliminate cannot be negative")
        if eliminate:
            if eliminate >= selected_indices.size:
                keep_count = 1
            else:
                keep_count = selected_indices.size - eliminate
            distance = np.abs(part_offsets[selected_indices] - anchor_offset)
            order = np.lexsort((part_scores[selected_indices], distance))
            selected_indices = selected_indices[order[:keep_count]]

    if selected_indices.size == 0:
        selected_indices = np.array([anchor_index], dtype=np.int64)

    selected_scores = part_scores[selected_indices]
    selected_offsets = part_offsets[selected_indices]
    if tolerance_offset is None:
        rotation_match_count = int(np.sum(selected_offsets == anchor_offset))
    else:
        tolerance = int(tolerance_offset)
        rotation_match_count = int(np.sum(np.abs(selected_offsets - anchor_offset) <= tolerance))

    return {
        "avg_hd": float(np.mean(selected_scores)),
        "anchor_offset": anchor_offset,
        "selected_indices": [int(index) for index in selected_indices],
        "selected_offsets": [int(offset) for offset in selected_offsets],
        "selected_scores": [float(score) for score in selected_scores],
        "rotation_match_count": rotation_match_count,
        "kept_parts": int(selected_indices.size),
    }


def compute_pairwise_rotation_classifier(
    labels,
    base_codes,
    base_masks,
    rotated_codes,
    rotated_masks,
    offsets,
    parts,
    threshold,
    eliminate=0,
    tolerance_offset=None,
    min_valid_bits=1,
    match_parts=None,
):
    slices = split_code_slices(base_codes.shape[1], parts)
    rows = []
    pair_count = len(labels) * (len(labels) - 1) // 2
    started = time.perf_counter()

    for pair_index, (idx1, idx2) in enumerate(combinations(range(len(labels)), 2), start=1):
        scores_12, offsets_12 = part_scores_for_offsets(
            base_codes[idx1],
            base_masks[idx1],
            rotated_codes[idx2],
            rotated_masks[idx2],
            offsets,
            slices,
            min_valid_bits=min_valid_bits,
        )
        scores_21, offsets_21 = part_scores_for_offsets(
            base_codes[idx2],
            base_masks[idx2],
            rotated_codes[idx1],
            rotated_masks[idx1],
            offsets,
            slices,
            min_valid_bits=min_valid_bits,
        )
        result_12 = select_parts(scores_12, offsets_12, eliminate, tolerance_offset)
        result_21 = select_parts(scores_21, offsets_21, eliminate, tolerance_offset)

        if result_12["avg_hd"] <= result_21["avg_hd"]:
            chosen = result_12
            direction = 1
        else:
            chosen = result_21
            direction = -1

        same_class = bool(labels[idx1] == labels[idx2])
        if match_parts is None:
            predicted_mated = bool(chosen["avg_hd"] <= threshold)
            prediction_mode = "hd_threshold"
            prediction_score = float(chosen["avg_hd"])
        else:
            predicted_mated = bool(chosen["rotation_match_count"] >= int(match_parts))
            prediction_mode = "rotation_match_count"
            prediction_score = float(chosen["rotation_match_count"])

        rows.append(
            {
                "idx1": int(idx1),
                "idx2": int(idx2),
                "same_class": same_class,
                "predicted_mated": predicted_mated,
                "correct": bool(predicted_mated == same_class),
                "prediction_mode": prediction_mode,
                "prediction_score": prediction_score,
                "avg_hd": float(chosen["avg_hd"]),
                "anchor_offset": int(chosen["anchor_offset"]),
                "rotation_match_count": int(chosen["rotation_match_count"]),
                "kept_parts": int(chosen["kept_parts"]),
                "selected_indices": chosen["selected_indices"],
                "selected_offsets": chosen["selected_offsets"],
                "selected_scores": chosen["selected_scores"],
                "direction": int(direction),
            }
        )

        if pair_index == 1 or pair_index % 25000 == 0 or pair_index == pair_count:
            elapsed = time.perf_counter() - started
            print(f"Rotation consistency scored pairs: {pair_index}/{pair_count} in {elapsed:.1f}s")

    return rows


def summarize_predictions(rows):
    if not rows:
        return {
            "total_pairs": 0,
            "mated_total": 0,
            "non_mated_total": 0,
            "accuracy": 0.0,
            "false_accept_rate": 0.0,
            "false_reject_rate": 0.0,
        }

    same_class = np.asarray([row["same_class"] for row in rows], dtype=bool)
    predicted_mated = np.asarray([row["predicted_mated"] for row in rows], dtype=bool)
    mated_total = int(np.sum(same_class))
    non_mated_total = int(np.sum(~same_class))
    correct = int(np.sum(predicted_mated == same_class))
    false_accepts = int(np.sum(~same_class & predicted_mated))
    false_rejects = int(np.sum(same_class & ~predicted_mated))

    return {
        "total_pairs": int(len(rows)),
        "mated_total": mated_total,
        "non_mated_total": non_mated_total,
        "accuracy": float(correct / len(rows)),
        "false_accepts": false_accepts,
        "false_rejects": false_rejects,
        "false_accept_rate": 0.0 if non_mated_total == 0 else float(false_accepts / non_mated_total),
        "false_reject_rate": 0.0 if mated_total == 0 else float(false_rejects / mated_total),
    }


def _eer_from_scores(same_class, scores, lower_is_mated=True):
    same_class = np.asarray(same_class, dtype=bool)
    scores = np.asarray(scores, dtype=np.float64)
    finite = np.isfinite(scores)
    same_class = same_class[finite]
    scores = scores[finite]
    if scores.size == 0 or np.unique(same_class).size < 2:
        return {
            "eer": 1.0,
            "roc_auc": 0.0,
            "threshold": None,
            "eer_fpr": 0.0,
            "eer_fnr": 0.0,
            "eer_std": 0.0,
        }

    ranking_scores = -scores if lower_is_mated else scores
    fpr, tpr, thresholds = roc_curve(same_class, ranking_scores)
    fnr = 1.0 - tpr
    eer_index = int(np.nanargmin(np.abs(fnr - fpr)))
    threshold = float(thresholds[eer_index])
    if lower_is_mated:
        threshold = -threshold
    eer_fpr = float(fpr[eer_index])
    eer_fnr = float(fnr[eer_index])
    mated_total = int(np.sum(same_class))
    non_mated_total = int(np.sum(~same_class))
    fpr_std = 0.0 if non_mated_total == 0 else np.sqrt(eer_fpr * (1.0 - eer_fpr) / non_mated_total)
    fnr_std = 0.0 if mated_total == 0 else np.sqrt(eer_fnr * (1.0 - eer_fnr) / mated_total)
    return {
        "eer": float((eer_fpr + eer_fnr) / 2.0),
        "roc_auc": float(auc(fpr, tpr)),
        "threshold": threshold,
        "eer_fpr": eer_fpr,
        "eer_fnr": eer_fnr,
        "eer_std": float(0.5 * np.sqrt((fpr_std ** 2) + (fnr_std ** 2))),
    }


def evaluate_eer(rows):
    if not rows:
        return {
            "eer": 1.0,
            "eer_std": 0.0,
            "eer_fpr": 0.0,
            "eer_fnr": 0.0,
            "roc_auc": 0.0,
            "eer_hd_threshold": None,
            "eer_match_parts_threshold": None,
        }

    same_class = [row["same_class"] for row in rows]
    hd_scores = [row["avg_hd"] for row in rows]
    match_scores = [row["rotation_match_count"] for row in rows]
    hd_eer = _eer_from_scores(same_class, hd_scores, lower_is_mated=True)
    match_eer = _eer_from_scores(same_class, match_scores, lower_is_mated=False)

    first_mode = rows[0].get("prediction_mode")
    if first_mode == "rotation_match_count":
        return {
            "eer": match_eer["eer"],
            "eer_std": match_eer["eer_std"],
            "eer_fpr": match_eer["eer_fpr"],
            "eer_fnr": match_eer["eer_fnr"],
            "roc_auc": match_eer["roc_auc"],
            "eer_hd_threshold": hd_eer["threshold"],
            "eer_match_parts_threshold": match_eer["threshold"],
        }
    return {
        "eer": hd_eer["eer"],
        "eer_std": hd_eer["eer_std"],
        "eer_fpr": hd_eer["eer_fpr"],
        "eer_fnr": hd_eer["eer_fnr"],
        "roc_auc": hd_eer["roc_auc"],
        "eer_hd_threshold": hd_eer["threshold"],
        "eer_match_parts_threshold": match_eer["threshold"],
    }
