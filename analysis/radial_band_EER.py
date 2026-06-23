from argparse import ArgumentParser
import os
from pathlib import Path
import sys

DEFAULT_OUTPUT_DIR = Path(__file__).resolve().parent / "output" / "radial_band_EER"
MATPLOTLIB_CONFIG_DIR = DEFAULT_OUTPUT_DIR / "matplotlib"
MATPLOTLIB_CONFIG_DIR.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(MATPLOTLIB_CONFIG_DIR))
os.environ.setdefault("XDG_CACHE_HOME", str(MATPLOTLIB_CONFIG_DIR))

import cv2 as cv
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from dataset_loaders import DATASET_CHOICES, dataset_output_slug, load_dataset, resolve_dataset, sample_dataset
from filter_loader import load_filter_bank
from iris import (
    IrisClassifier,
    UNET_BAND_SHAPE,
    build_valid_source_mask,
    clean_component_mask,
    fit_boundary_from_mask,
    fit_polar_boundary_from_mask,
    get_iris_band,
    predict_unet_masks,
)
from pairwise_iris_analysis import (
    compute_pairwise_scores_iriscode,
    evaluate_scores,
    summarize_label_pairs,
)

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator


def bit_center_rows(classifier, band_shape):
    iris_h, iris_w = band_shape
    rows = []

    for filter_settings in classifier._filter_settings:
        x_stride, y_stride = filter_settings["stride"]
        _, start_y = filter_settings["start_position"]
        num_x = iris_w // x_stride
        num_y = iris_h // y_stride
        y_positions = start_y + y_stride * np.arange(num_y)

        filter_rows = np.tile(y_positions, num_x)
        rows.append(np.repeat(filter_rows, 2))

    return np.concatenate(rows).astype(np.int32)


def bit_center_columns(classifier, band_shape):
    iris_h, iris_w = band_shape
    columns = []

    for filter_settings in classifier._filter_settings:
        x_stride, y_stride = filter_settings["stride"]
        start_x, _ = filter_settings["start_position"]
        num_x = iris_w // x_stride
        num_y = iris_h // y_stride
        x_positions = (start_x + x_stride * np.arange(num_x)) % iris_w

        filter_columns = np.repeat(x_positions, num_y)
        columns.append(np.repeat(filter_columns, 2))

    return np.concatenate(columns).astype(np.int32)


def bit_support_rows(classifier, band_shape):
    iris_h, iris_w = band_shape
    rows = []

    for filter_settings, (filter_real, _) in zip(classifier._filter_settings, classifier._filters):
        x_stride, y_stride = filter_settings["stride"]
        _, start_y = filter_settings["start_position"]
        filter_h = filter_real.shape[0]
        y_half = filter_h // 2
        num_x = iris_w // x_stride
        num_y = iris_h // y_stride
        y_positions = start_y + y_stride * np.arange(num_y)
        window_tops = y_positions - y_half
        window_bottoms = window_tops + filter_h
        filter_rows = np.column_stack((window_tops, window_bottoms))
        rows.append(np.repeat(np.tile(filter_rows, (num_x, 1)), 2, axis=0))

    return np.concatenate(rows).astype(np.int32)


def radial_eligible_indices(center_rows, band_height, discard_outer=0.0, discard_inner=0.0, support_rows=None):
    if discard_outer or discard_inner:
        if support_rows is None:
            center_positions = center_rows.astype(np.float32) / max(band_height - 1, 1)
            support_rows = np.column_stack((center_positions, center_positions))
            inner_limit = discard_inner
            outer_limit = 1.0 - discard_outer
        else:
            inner_limit = discard_inner * band_height
            outer_limit = (1.0 - discard_outer) * band_height
        eligible_indices = np.flatnonzero(
            (support_rows[:, 0] >= inner_limit) & (support_rows[:, 1] <= outer_limit)
        )
    else:
        eligible_indices = np.arange(center_rows.shape[0])
    return eligible_indices


def band_selector(center_positions, band_index, band_count, band_extent, eligible_indices=None):
    if eligible_indices is None:
        eligible_indices = np.arange(center_positions.shape[0])
    order = eligible_indices[np.argsort(center_positions[eligible_indices], kind="stable")]
    split_indices = np.array_split(order, band_count)
    selected_indices = split_indices[band_index - 1]
    selector = np.zeros(center_positions.shape[0], dtype=bool)
    selector[selected_indices] = True

    selected_positions = center_positions[selected_indices]
    start = int(np.min(selected_positions)) if selected_positions.size else None
    end = int(np.max(selected_positions)) + 1 if selected_positions.size else None
    center = float(np.mean(selected_positions) / max(band_extent - 1, 1)) if selected_positions.size else None
    return selector, start, end, center


def precompute_codes(images, labels, image_names, classifier, rotation):
    offsets = np.arange(rotation) - rotation // 2
    sample_count = len(images)

    base_codes = []
    base_masks = []
    rotated_codes = []
    rotated_masks = []
    kept_labels = []
    kept_image_names = []
    skipped = []

    for index, image in enumerate(images, start=1):
        if index == 1 or index % 100 == 0 or index == sample_count:
            print(f"Precomputing iris codes: {index}/{sample_count}")

        try:
            iris_band, iris_mask = get_iris_band(image)
        except Exception as exc:
            skipped.append((index - 1, str(image_names[index - 1]), str(exc)))
            continue
        if iris_band is None or iris_mask is None:
            skipped.append((index - 1, str(image_names[index - 1]), "segmentation returned None"))
            continue

        base_code, base_mask, _ = classifier.get_iris_code(iris_band, iris_mask, offset=0)
        base_codes.append(np.asarray(base_code, dtype=bool))
        base_masks.append(np.asarray(base_mask, dtype=bool))
        kept_labels.append(labels[index - 1])
        kept_image_names.append(image_names[index - 1])

        image_rotated_codes = []
        image_rotated_masks = []
        for offset in offsets:
            code, code_mask, _ = classifier.get_iris_code(iris_band, iris_mask, offset=int(offset))
            image_rotated_codes.append(np.asarray(code, dtype=bool))
            image_rotated_masks.append(np.asarray(code_mask, dtype=bool))

        rotated_codes.append(np.stack(image_rotated_codes, axis=0))
        rotated_masks.append(np.stack(image_rotated_masks, axis=0))

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


def evaluate_band(labels, base_codes, base_masks, rotated_codes, rotated_masks, offsets, selector):
    selected_base_masks = base_masks & selector[None, :]
    selected_rotated_masks = rotated_masks & selector[None, None, :]
    valid_bits = np.sum(selected_base_masks, axis=1)

    pairwise = compute_pairwise_scores_iriscode(
        labels,
        base_codes,
        selected_base_masks,
        rotated_codes,
        selected_rotated_masks,
        offsets,
    )
    evaluation = evaluate_scores(pairwise["same_class"], pairwise["scores"])

    return evaluation, pairwise, valid_bits


def best_single_pair_band_score(base_code, base_mask, candidate_codes, candidate_masks, offsets, selector):
    selected_base_mask = np.asarray(base_mask, dtype=bool) & selector
    selected_candidate_masks = np.asarray(candidate_masks, dtype=bool) & selector[None, :]
    diff = np.bitwise_xor(candidate_codes, base_code)
    combined_mask = np.bitwise_and(selected_candidate_masks, selected_base_mask)
    valid_bits = np.sum(combined_mask, axis=1)
    mismatch_bits = np.sum(np.bitwise_and(diff, combined_mask), axis=1)

    scores = np.full(candidate_codes.shape[0], 2.0, dtype=np.float64)
    valid_rows = valid_bits > 0
    scores[valid_rows] = mismatch_bits[valid_rows] / valid_bits[valid_rows]
    best_index = int(np.argmin(scores))
    return float(scores[best_index]), int(offsets[best_index]), int(valid_bits[best_index])


def display_image_for_plot(image):
    if image.ndim == 2:
        return image, "gray"
    if image.shape[2] == 4:
        return cv.cvtColor(image, cv.COLOR_BGRA2RGBA), None
    return cv.cvtColor(image, cv.COLOR_BGR2RGB), None


def plot_single_pair_rows(rows, figure_path, metadata, image_a=None, image_b=None):
    bands = np.array([row["band"] for row in rows], dtype=np.int32)
    scores = np.array([row["hamming_distance"] for row in rows], dtype=np.float64)
    valid_fraction = np.array([row["valid_fraction"] for row in rows], dtype=np.float64)
    offsets = np.array([row["best_offset"] for row in rows], dtype=np.int32)
    band_axis = rows[0].get("band_axis", "radial") if rows else "radial"
    axis_label = "Band (1 = pupil side)" if band_axis == "radial" else "Band (angular sector)"
    title_suffix = "Radial Band" if band_axis == "radial" else "Horizontal Band"

    figure_width = max(14, min(22, len(bands) * 0.7))
    if image_a is not None and image_b is not None:
        figure, axes_grid = plt.subplots(
            2,
            3,
            figsize=(figure_width, 9.2),
            height_ratios=(1.2, 1.0),
        )
        image_a_display, image_a_cmap = display_image_for_plot(image_a)
        image_b_display, image_b_cmap = display_image_for_plot(image_b)
        axes_grid[0, 0].imshow(image_a_display, cmap=image_a_cmap)
        axes_grid[0, 0].set_title(f"Image A: {metadata.get('image_a', '')}")
        axes_grid[0, 0].axis("off")
        axes_grid[0, 1].imshow(image_b_display, cmap=image_b_cmap)
        axes_grid[0, 1].set_title(f"Image B: {metadata.get('image_b', '')}")
        axes_grid[0, 1].axis("off")
        axes_grid[0, 2].axis("off")
        axes = axes_grid[1]
    else:
        figure, axes = plt.subplots(1, 3, figsize=(figure_width, 5.8))

    axes[0].plot(bands, scores, color="#1f77b4", lw=2)
    axes[0].scatter(bands, scores, color="#1f77b4", s=28)
    axes[0].set_title(f"Single-Pair HD per {title_suffix}")
    axes[0].set_xlabel(axis_label)
    axes[0].set_ylabel("Hamming distance")
    axes[0].set_ylim(0.0, min(1.0, max(0.6, float(np.max(scores)) * 1.15)))
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(bands, offsets, color="#f08c00", lw=2)
    axes[1].scatter(bands, offsets, color="#f08c00", s=28)
    axes[1].set_title("Best Rotation per Band")
    axes[1].set_xlabel(axis_label)
    axes[1].set_ylabel("Rotation offset")
    axes[1].grid(True, alpha=0.3)

    axes[2].plot(bands, valid_fraction, color="#7048e8", lw=2)
    axes[2].scatter(bands, valid_fraction, color="#7048e8", s=28)
    axes[2].set_title("Valid Bits per Band")
    axes[2].set_xlabel(axis_label)
    axes[2].set_ylabel("Valid-bit fraction")
    axes[2].set_ylim(0.0, 1.0)
    axes[2].grid(True, alpha=0.3)

    tick_rotation = 45 if len(bands) > 16 else 0
    for axis in axes:
        axis.xaxis.set_major_locator(MaxNLocator(nbins=min(12, max(2, len(bands))), integer=True))
        axis.tick_params(axis="x", labelrotation=tick_rotation, labelsize=8)

    figure.tight_layout()
    figure_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(figure_path, dpi=300, bbox_inches="tight")
    plt.close(figure)


def plot_rows(rows, figure_path):
    bands = np.array([row["band"] for row in rows], dtype=np.int32)
    eer = np.array([row["eer"] for row in rows], dtype=np.float64)
    valid_fraction = np.array([row["mean_valid_fraction"] for row in rows], dtype=np.float64)
    mated_mode_offsets = [row.get("mated_mode_offset") for row in rows]
    band_axis = rows[0].get("band_axis", "radial") if rows else "radial"
    axis_label = "Band (1 = pupil side)" if band_axis == "radial" else "Band (angular sector)"
    title_suffix = "Radial Band" if band_axis == "radial" else "Horizontal Band"

    figure_width = max(16, min(24, len(bands) * 0.7))
    figure, axes = plt.subplots(1, 2, figsize=(figure_width, 5.8))

    axes[0].plot(bands, eer, color="#1f77b4", lw=2)
    axes[0].scatter(bands, eer, color="#1f77b4", s=28)
    for band, band_eer, offset in zip(bands, eer, mated_mode_offsets):
        if offset is None:
            continue
        axes[0].annotate(
            f"{int(offset)} px",
            xy=(band, band_eer),
            xytext=(0, 8),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=7,
            color="#1f77b4",
        )
    axes[0].set_title(f"EER per {title_suffix}")
    axes[0].set_xlabel(axis_label)
    axes[0].set_ylabel("Equal Error Rate")
    axes[0].set_ylim(bottom=0.0)
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(bands, valid_fraction, color="#7048e8", lw=2)
    axes[1].scatter(bands, valid_fraction, color="#7048e8", s=28)
    axes[1].set_title("Valid Bits per Band")
    axes[1].set_xlabel(axis_label)
    axes[1].set_ylabel("Mean valid-bit fraction")
    axes[1].set_ylim(0.0, 1.0)
    axes[1].grid(True, alpha=0.3)

    tick_rotation = 45 if len(bands) > 16 else 0
    for axis in axes:
        axis.xaxis.set_major_locator(MaxNLocator(nbins=min(12, max(2, len(bands))), integer=True))
        axis.tick_params(axis="x", labelrotation=tick_rotation, labelsize=8)

    figure.tight_layout()
    figure_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(figure_path, dpi=300, bbox_inches="tight")
    plt.close(figure)


def most_common_offset(offsets):
    offsets = np.asarray(offsets, dtype=np.int16)
    if offsets.size == 0:
        return None, 0, 0.0
    values, counts = np.unique(offsets, return_counts=True)
    best_count = int(np.max(counts))
    tied_values = values[counts == best_count]
    best_value = int(tied_values[np.argmin(np.abs(tied_values))])
    return best_value, best_count, float(best_count / offsets.size)


def band_rows_from_centers(classifier, bands, discard_outer=0.0, discard_inner=0.0):
    center_rows = bit_center_rows(classifier, UNET_BAND_SHAPE)
    support_rows = bit_support_rows(classifier, UNET_BAND_SHAPE)
    eligible_indices = radial_eligible_indices(
        center_rows,
        UNET_BAND_SHAPE[0],
        discard_outer=discard_outer,
        discard_inner=discard_inner,
        support_rows=support_rows,
    )
    rows = []
    for band in range(1, bands + 1):
        selector, row_start, row_end, center_position = band_selector(
            center_rows,
            band,
            bands,
            UNET_BAND_SHAPE[0],
            eligible_indices=eligible_indices,
        )
        if not np.any(selector):
            raise RuntimeError(f"Band {band} selected zero iriscode bits. Try fewer bands.")
        rows.append(
            {
                "band": band,
                "row_start": row_start,
                "row_end": row_end,
                "center_position": center_position,
                "selected_bits": int(np.sum(selector)),
            }
        )
    return rows


def band_edges_from_rows(band_rows, band_height, discard_outer=0.0, discard_inner=0.0):
    outer_keep_edge = 1.0 - discard_outer
    edges = [discard_inner]
    for first, second in zip(band_rows, band_rows[1:]):
        edges.append((float(first["row_end"]) + float(second["row_start"])) / 2.0 / max(band_height - 1, 1))
    edges.append(outer_keep_edge)
    return np.clip(np.asarray(edges, dtype=np.float32), 0.0, 1.0)


def radial_band_overlay_original(raw_image, classifier, bands, alpha=0.35, discard_outer=0.0, discard_inner=0.0, band_axis="radial"):
    gray, iris_mask, pupil_mask, eyelash_mask = predict_unet_masks(raw_image)
    iris_mask = clean_component_mask(iris_mask)
    pupil_mask = clean_component_mask(pupil_mask)
    eyelash_mask = clean_component_mask(eyelash_mask, kernel_size=3)

    valid_source_mask = build_valid_source_mask(
        iris_mask,
        pupil_mask,
        eyelash_mask,
        source_image=gray,
        oversat_threshold=254,
    )
    pupil_ellipse = fit_boundary_from_mask(pupil_mask, prefer_ellipse=True)
    center = (pupil_ellipse.center_x, pupil_ellipse.center_y)
    pupil_boundary = fit_polar_boundary_from_mask(pupil_mask, center=center, num_angles=UNET_BAND_SHAPE[1], smooth_kernel=7)
    iris_boundary = fit_polar_boundary_from_mask(iris_mask, center=center, num_angles=UNET_BAND_SHAPE[1], smooth_kernel=17)

    theta = np.linspace(0.0, 2.0 * np.pi, UNET_BAND_SHAPE[1], endpoint=False, dtype=np.float32)
    pupil_x, pupil_y = pupil_boundary.sample(theta)
    iris_x, iris_y = iris_boundary.sample(theta)

    overlay = cv.cvtColor(gray, cv.COLOR_GRAY2RGB).astype(np.float32)
    cmap = plt.get_cmap("tab20", bands)

    if band_axis == "horizontal":
        edges = np.linspace(0, UNET_BAND_SHAPE[1], bands + 1)
        for band_index in range(bands):
            start = int(round(edges[band_index]))
            end = int(round(edges[band_index + 1]))
            if end <= start:
                continue
            outer_points = np.column_stack((iris_x[start:end], iris_y[start:end]))
            inner_points = np.column_stack((pupil_x[start:end], pupil_y[start:end]))
            if outer_points.shape[0] < 2:
                continue
            polygon = np.vstack((outer_points, inner_points[::-1])).round().astype(np.int32)
            band_mask = np.zeros(gray.shape, dtype=np.uint8)
            cv.fillPoly(band_mask, [polygon], 255)
            band_mask = cv.bitwise_and(band_mask, valid_source_mask)
            color = np.array(cmap(band_index)[:3], dtype=np.float32) * 255.0
            valid = band_mask > 0
            overlay[valid] = (1.0 - alpha) * overlay[valid] + alpha * color

        for edge in edges:
            index = int(round(edge)) % UNET_BAND_SHAPE[1]
            points = np.array(
                [[pupil_x[index], pupil_y[index]], [iris_x[index], iris_y[index]]],
                dtype=np.int32,
            )
            cv.polylines(overlay, [points], isClosed=False, color=(255, 255, 255), thickness=1, lineType=cv.LINE_AA)
        return np.clip(overlay, 0, 255).astype(np.uint8)

    band_rows = band_rows_from_centers(
        classifier,
        bands,
        discard_outer=discard_outer,
        discard_inner=discard_inner,
    )
    edges = band_edges_from_rows(
        band_rows,
        UNET_BAND_SHAPE[0],
        discard_outer=discard_outer,
        discard_inner=discard_inner,
    )
    for band_index in range(bands):
        inner = float(edges[band_index])
        outer = float(edges[band_index + 1])
        inner_points = np.column_stack(
            (
                (1.0 - inner) * pupil_x + inner * iris_x,
                (1.0 - inner) * pupil_y + inner * iris_y,
            )
        )
        outer_points = np.column_stack(
            (
                (1.0 - outer) * pupil_x + outer * iris_x,
                (1.0 - outer) * pupil_y + outer * iris_y,
            )
        )
        polygon = np.vstack((outer_points, inner_points[::-1])).round().astype(np.int32)
        band_mask = np.zeros(gray.shape, dtype=np.uint8)
        cv.fillPoly(band_mask, [polygon], 255)
        band_mask = cv.bitwise_and(band_mask, valid_source_mask)
        color = np.array(cmap(band_index)[:3], dtype=np.float32) * 255.0
        valid = band_mask > 0
        overlay[valid] = (1.0 - alpha) * overlay[valid] + alpha * color

    discard_color = np.array([255.0, 64.0, 64.0], dtype=np.float32)
    if discard_inner:
        inner = 0.0
        outer = discard_inner
        inner_points = np.column_stack(
            (
                (1.0 - inner) * pupil_x + inner * iris_x,
                (1.0 - inner) * pupil_y + inner * iris_y,
            )
        )
        outer_points = np.column_stack(
            (
                (1.0 - outer) * pupil_x + outer * iris_x,
                (1.0 - outer) * pupil_y + outer * iris_y,
            )
        )
        polygon = np.vstack((outer_points, inner_points[::-1])).round().astype(np.int32)
        discard_mask = np.zeros(gray.shape, dtype=np.uint8)
        cv.fillPoly(discard_mask, [polygon], 255)
        discard_mask = cv.bitwise_and(discard_mask, valid_source_mask)
        valid = discard_mask > 0
        overlay[valid] = (1.0 - alpha) * overlay[valid] + alpha * discard_color

    if discard_outer:
        inner = 1.0 - discard_outer
        inner_points = np.column_stack(
            (
                (1.0 - inner) * pupil_x + inner * iris_x,
                (1.0 - inner) * pupil_y + inner * iris_y,
            )
        )
        outer_points = np.column_stack((iris_x, iris_y))
        polygon = np.vstack((outer_points, inner_points[::-1])).round().astype(np.int32)
        discard_mask = np.zeros(gray.shape, dtype=np.uint8)
        cv.fillPoly(discard_mask, [polygon], 255)
        discard_mask = cv.bitwise_and(discard_mask, valid_source_mask)
        valid = discard_mask > 0
        overlay[valid] = (1.0 - alpha) * overlay[valid] + alpha * discard_color

    boundary_edges = list(edges)
    if discard_inner:
        boundary_edges.append(0.0)
    if discard_outer:
        boundary_edges.append(1.0)
    for edge in boundary_edges:
        edge_x = (1.0 - edge) * pupil_x + edge * iris_x
        edge_y = (1.0 - edge) * pupil_y + edge * iris_y
        points = np.column_stack((edge_x, edge_y)).round().astype(np.int32)
        cv.polylines(overlay, [points], isClosed=True, color=(255, 255, 255), thickness=1, lineType=cv.LINE_AA)

    return np.clip(overlay, 0, 255).astype(np.uint8)


def save_original_band_overlay(raw_image, output_path, classifier, bands, alpha, discard_outer=0.0, discard_inner=0.0, band_axis="radial"):
    overlay = radial_band_overlay_original(
        raw_image,
        classifier,
        bands,
        alpha=alpha,
        discard_outer=discard_outer,
        discard_inner=discard_inner,
        band_axis=band_axis,
    )
    figure, axes = plt.subplots(1, 2, figsize=(14, 7))
    axes[0].imshow(raw_image, cmap="gray", vmin=0, vmax=255)
    axes[0].set_title("Original Image")
    axes[0].axis("off")

    axes[1].imshow(overlay)
    overlay_title = "Radial Band Overlay (band 1 = pupil side)" if band_axis == "radial" else "Horizontal Band Overlay"
    axes[1].set_title(overlay_title)
    axes[1].axis("off")

    figure.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(figure)
    return output_path


def normalized_band_overlay(raw_image, bands, alpha=0.35, discard_outer=0.0, discard_inner=0.0, band_axis="radial"):
    iris_band, iris_mask = get_iris_band(raw_image)
    if iris_band is None or iris_mask is None:
        raise RuntimeError("Segmentation returned None for preview image.")

    if iris_band.ndim == 2:
        base = cv.cvtColor(iris_band, cv.COLOR_GRAY2RGB)
    else:
        base = cv.cvtColor(iris_band, cv.COLOR_BGR2RGB)
    overlay = base.astype(np.float32)

    band_h, band_w = iris_band.shape[:2]
    if band_axis == "radial":
        keep_start = int(round(discard_inner * band_h))
        keep_end = int(round((1.0 - discard_outer) * band_h))
        extent = band_h
    else:
        keep_start = 0
        keep_end = band_w
        extent = band_w
    keep_start = int(np.clip(keep_start, 0, extent))
    keep_end = int(np.clip(keep_end, keep_start + 1, extent))
    edges = np.linspace(keep_start, keep_end, bands + 1)
    cmap = plt.get_cmap("tab20", bands)

    for band_index in range(bands):
        start = int(round(edges[band_index]))
        end = int(round(edges[band_index + 1]))
        if end <= start:
            continue
        color = np.array(cmap(band_index)[:3], dtype=np.float32) * 255.0
        if band_axis == "radial":
            overlay[start:end, :] = (1.0 - alpha) * overlay[start:end, :] + alpha * color
        else:
            overlay[:, start:end] = (1.0 - alpha) * overlay[:, start:end] + alpha * color

    if band_axis == "radial" and discard_inner:
        overlay[:keep_start, :] = (0.6 * overlay[:keep_start, :] + 0.4 * np.array([255.0, 64.0, 64.0]))
    if band_axis == "radial" and discard_outer:
        overlay[keep_end:, :] = (0.6 * overlay[keep_end:, :] + 0.4 * np.array([255.0, 64.0, 64.0]))

    for edge in edges:
        position = int(round(edge))
        if band_axis == "radial":
            cv.line(overlay, (0, position), (band_w - 1, position), (255, 255, 255), 1, lineType=cv.LINE_AA)
        else:
            cv.line(overlay, (position, 0), (position, band_h - 1), (255, 255, 255), 1, lineType=cv.LINE_AA)

    invalid = iris_mask == 0
    if np.any(invalid):
        overlay[invalid] *= 0.35

    return np.clip(overlay, 0, 255).astype(np.uint8), iris_band, iris_mask


def save_combined_band_preview(raw_image, output_path, classifier, bands, alpha, discard_outer=0.0, discard_inner=0.0, band_axis="radial"):
    normalized_overlay, iris_band, iris_mask = normalized_band_overlay(
        raw_image,
        bands,
        alpha=alpha,
        discard_outer=discard_outer,
        discard_inner=discard_inner,
        band_axis=band_axis,
    )
    original_overlay = radial_band_overlay_original(
        raw_image,
        classifier,
        bands,
        alpha=alpha,
        discard_outer=discard_outer,
        discard_inner=discard_inner,
        band_axis=band_axis,
    )

    figure, axes = plt.subplots(3, 1, figsize=(14, 10), height_ratios=(1.0, 1.0, 2.0))
    axes[0].imshow(iris_band, cmap="gray", vmin=0, vmax=255, aspect="auto")
    axes[0].set_title("Normalized Iris Band")
    axes[0].axis("off")

    axes[1].imshow(normalized_overlay, aspect="auto")
    normalized_title = "Normalized Iris Band with Equal-Height Bands" if band_axis == "radial" else "Normalized Iris Band with Horizontal Bands"
    axes[1].set_title(normalized_title)
    if band_axis == "horizontal":
        band_width = normalized_overlay.shape[1]
        edges = np.linspace(0, band_width, bands + 1)
        centers = (edges[:-1] + edges[1:]) / 2.0
        axes[1].set_xticks(centers)
        axes[1].set_xticklabels([str(index) for index in range(1, bands + 1)], fontsize=8)
        axes[1].set_yticks([])
        axes[1].tick_params(axis="x", length=0, pad=4)
        for spine in axes[1].spines.values():
            spine.set_visible(False)
    else:
        axes[1].axis("off")

    axes[2].imshow(original_overlay)
    axes[2].set_title("Same Bands Projected onto Original Iris Shape")
    axes[2].axis("off")

    figure.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(figure)
    return output_path


def save_sample_overlay(
    raw_image,
    output_dir,
    output_name,
    classifier,
    bands,
    alpha,
    discard_outer=0.0,
    discard_inner=0.0,
    band_axis="radial",
):
    preview_path = output_dir / f"{output_name}_band_preview.png"
    save_combined_band_preview(
        raw_image,
        preview_path,
        classifier,
        bands,
        alpha,
        discard_outer=discard_outer,
        discard_inner=discard_inner,
        band_axis=band_axis,
    )
    return preview_path


def load_single_image(path):
    image_path = Path(path).expanduser().resolve()
    image = cv.imread(str(image_path), cv.IMREAD_UNCHANGED)
    if image is None:
        raise FileNotFoundError(f"Could not read image: {image_path}")
    return image_path, image


def encode_single_image(image, classifier, offsets):
    iris_band, iris_mask = get_iris_band(image)
    if iris_band is None or iris_mask is None:
        raise RuntimeError("Segmentation returned None.")

    base_code, base_mask, _ = classifier.get_iris_code(iris_band, iris_mask, offset=0)
    rotated_codes = []
    rotated_masks = []
    for offset in offsets:
        code, mask, _ = classifier.get_iris_code(iris_band, iris_mask, offset=int(offset))
        rotated_codes.append(np.asarray(code, dtype=bool))
        rotated_masks.append(np.asarray(mask, dtype=bool))

    return (
        np.asarray(base_code, dtype=bool),
        np.asarray(base_mask, dtype=bool),
        np.stack(rotated_codes, axis=0),
        np.stack(rotated_masks, axis=0),
    )


def run_single_pair(args):
    if args.rotation < 1:
        raise ValueError("--rotation must be at least 1")
    if args.bands < 2:
        raise ValueError("--bands must be at least 2")
    if not args.image_a or not args.image_b:
        raise ValueError("Single-pair mode needs both --image-a and --image-b.")
    if not 0.0 <= args.discard_outer < 1.0:
        raise ValueError("discard_outer must be in the range [0.0, 1.0)")
    if not 0.0 <= args.discard_inner < 1.0:
        raise ValueError("discard_inner must be in the range [0.0, 1.0)")
    if args.discard_inner + args.discard_outer >= 1.0:
        raise ValueError("discard_inner + discard_outer must be less than 1.0")
    if args.band_axis == "horizontal" and (args.discard_inner or args.discard_outer):
        print("Note: radial discard settings still remove pupil/limbus bits before horizontal-band analysis.")

    image_a_path, image_a = load_single_image(args.image_a)
    image_b_path, image_b = load_single_image(args.image_b)
    selected_filters, filters_source = load_filter_bank(args.filters)
    print(f"Filters in use: {len(selected_filters)}")
    print(f"Filters source: {filters_source}")
    classifier = IrisClassifier(selected_filters)
    offsets = np.arange(args.rotation) - args.rotation // 2

    code_a, mask_a, rotated_codes_a, rotated_masks_a = encode_single_image(image_a, classifier, offsets)
    code_b, mask_b, rotated_codes_b, rotated_masks_b = encode_single_image(image_b, classifier, offsets)

    center_rows = bit_center_rows(classifier, UNET_BAND_SHAPE)
    center_columns = bit_center_columns(classifier, UNET_BAND_SHAPE)
    support_rows = bit_support_rows(classifier, UNET_BAND_SHAPE)
    if center_rows.shape[0] != code_a.shape[0]:
        raise RuntimeError(f"Band selector length {center_rows.shape[0]} does not match iriscode length {code_a.shape[0]}.")
    if center_columns.shape[0] != code_a.shape[0]:
        raise RuntimeError(f"Column band selector length {center_columns.shape[0]} does not match iriscode length {code_a.shape[0]}.")

    eligible_indices = radial_eligible_indices(
        center_rows,
        UNET_BAND_SHAPE[0],
        discard_outer=args.discard_outer,
        discard_inner=args.discard_inner,
        support_rows=support_rows,
    )
    if args.band_axis == "radial":
        band_positions = center_rows
        band_extent = UNET_BAND_SHAPE[0]
    else:
        band_positions = center_columns
        band_extent = UNET_BAND_SHAPE[1]

    rows = []
    for band in range(1, args.bands + 1):
        selector, position_start, position_end, _center_position = band_selector(
            band_positions,
            band,
            args.bands,
            band_extent,
            eligible_indices=eligible_indices,
        )
        if not np.any(selector):
            raise RuntimeError(f"Band {band} selected zero iriscode bits. Try fewer bands.")

        score_ab, offset_ab, valid_ab = best_single_pair_band_score(
            code_a,
            mask_a,
            rotated_codes_b,
            rotated_masks_b,
            offsets,
            selector,
        )
        score_ba, offset_ba, valid_ba = best_single_pair_band_score(
            code_b,
            mask_b,
            rotated_codes_a,
            rotated_masks_a,
            offsets,
            selector,
        )
        if score_ab <= score_ba:
            score = score_ab
            best_offset = offset_ab
            valid_bits = valid_ab
            direction = "a_to_b"
        else:
            score = score_ba
            best_offset = offset_ba
            valid_bits = valid_ba
            direction = "b_to_a"

        selected_bits = int(np.sum(selector))
        row = {
            "band_axis": args.band_axis,
            "band": band,
            "position_start": position_start,
            "position_end": position_end,
            "row_start": position_start if args.band_axis == "radial" else None,
            "row_end": position_end if args.band_axis == "radial" else None,
            "column_start": position_start if args.band_axis == "horizontal" else None,
            "column_end": position_end if args.band_axis == "horizontal" else None,
            "selected_bits": selected_bits,
            "hamming_distance": score,
            "best_offset": best_offset,
            "direction": direction,
            "valid_bits": valid_bits,
            "valid_fraction": float(valid_bits / selected_bits) if selected_bits else 0.0,
        }
        rows.append(row)
        print(
            f"Band {band}/{args.bands}: HD={score:.6f} "
            f"offset={best_offset} valid_bits={valid_bits}/{selected_bits}"
        )

    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    output_name = args.output_name or f"single_pair_{args.bands}_{args.band_axis}_bands"
    figure_path = output_dir / f"{output_name}.png"
    metadata = {
        "image_a": image_a_path.name,
        "image_b": image_b_path.name,
    }
    plot_single_pair_rows(rows, figure_path, metadata, image_a=image_a, image_b=image_b)

    best = min(rows, key=lambda item: item["hamming_distance"])
    print(f"Best band: {best['band']} HD={best['hamming_distance']:.6f} offset={best['best_offset']}")
    print("Note: single-pair mode reports HD per band, not EER, because EER requires mated and non-mated score distributions.")
    print(f"Saved figure to {figure_path}")


def run_eer(args):
    if args.rotation < 1:
        raise ValueError("--rotation must be at least 1")
    if args.bands < 2:
        raise ValueError("--bands must be at least 2")
    if not 0.0 <= args.discard_outer < 1.0:
        raise ValueError("discard_outer must be in the range [0.0, 1.0)")
    if not 0.0 <= args.discard_inner < 1.0:
        raise ValueError("discard_inner must be in the range [0.0, 1.0)")
    if args.discard_inner + args.discard_outer >= 1.0:
        raise ValueError("discard_inner + discard_outer must be less than 1.0")
    if args.band_axis == "horizontal" and (args.discard_inner or args.discard_outer):
        print("Note: radial discard settings still remove pupil/limbus bits before horizontal-band analysis.")

    dataset_path, dataset_format = resolve_dataset(args.dataset_path, args.dataset_format)
    dataset_name = dataset_output_slug(dataset_format)
    output_name = args.output_name or f"{dataset_name}_{args.bands}_{args.band_axis}_bands"
    output_dir = Path(args.output_dir).expanduser().resolve()
    figure_path = output_dir / f"{output_name}.png"

    print(f"Using dataset format: {dataset_format}")
    print(f"Using dataset path: {dataset_path}")
    if args.discard_outer:
        print(f"Discarding outer iris fraction: {args.discard_outer:.4f}")
    if args.discard_inner:
        print(f"Discarding inner iris fraction: {args.discard_inner:.4f}")

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
        for skipped_index, skipped_name, reason in skipped[:5]:
            print(f"  skipped[{skipped_index}] {skipped_name}: {reason}")

    summary = summarize_label_pairs(labels)
    print(f"Usable samples: {summary['sample_count']}")
    print(f"Usable classes: {summary['class_count']}")
    print(f"Usable mated pairs: {summary['mated_pairs']}")
    print(f"Usable non-mated pairs: {summary['non_mated_pairs']}")
    if summary["mated_pairs"] == 0 or summary["non_mated_pairs"] == 0:
        raise ValueError("After segmentation, the subset needs both mated and non-mated pairs.")

    center_rows = bit_center_rows(classifier, UNET_BAND_SHAPE)
    center_columns = bit_center_columns(classifier, UNET_BAND_SHAPE)
    support_rows = bit_support_rows(classifier, UNET_BAND_SHAPE)
    if center_rows.shape[0] != base_codes.shape[1]:
        raise RuntimeError(f"Band selector length {center_rows.shape[0]} does not match iriscode length {base_codes.shape[1]}.")
    if center_columns.shape[0] != base_codes.shape[1]:
        raise RuntimeError(f"Column band selector length {center_columns.shape[0]} does not match iriscode length {base_codes.shape[1]}.")
    if support_rows.shape[0] != base_codes.shape[1]:
        raise RuntimeError(f"Band support length {support_rows.shape[0]} does not match iriscode length {base_codes.shape[1]}.")
    eligible_indices = radial_eligible_indices(
        center_rows,
        UNET_BAND_SHAPE[0],
        discard_outer=args.discard_outer,
        discard_inner=args.discard_inner,
        support_rows=support_rows,
    )
    if args.band_axis == "radial":
        band_positions = center_rows
        band_extent = UNET_BAND_SHAPE[0]
        start_key = "row_start"
        end_key = "row_end"
        position_label = "center_rows"
    else:
        band_positions = center_columns
        band_extent = UNET_BAND_SHAPE[1]
        start_key = "column_start"
        end_key = "column_end"
        position_label = "center_columns"

    rows = []
    for band in range(1, args.bands + 1):
        selector, position_start, position_end, center_position = band_selector(
            band_positions,
            band,
            args.bands,
            band_extent,
            eligible_indices=eligible_indices,
        )
        if not np.any(selector):
            raise RuntimeError(f"Band {band} selected zero iriscode bits. Try fewer bands.")

        print(
            f"Evaluating band {band}/{args.bands} {position_label} {position_start}:{position_end} "
            f"selected_bits={int(np.sum(selector))}"
        )
        evaluation, pairwise, valid_bits = evaluate_band(
            labels,
            base_codes,
            base_masks,
            rotated_codes,
            rotated_masks,
            offsets,
            selector,
        )
        scores = pairwise["scores"]
        same_class = pairwise["same_class"]
        mated_mode_offset, mated_mode_count, mated_mode_fraction = most_common_offset(pairwise["best_offset"][same_class])
        row = {
            "band_axis": args.band_axis,
            "band": band,
            "position_start": position_start,
            "position_end": position_end,
            "row_start": position_start if args.band_axis == "radial" else None,
            "row_end": position_end if args.band_axis == "radial" else None,
            "column_start": position_start if args.band_axis == "horizontal" else None,
            "column_end": position_end if args.band_axis == "horizontal" else None,
            "center_position": center_position,
            "selected_bits": int(np.sum(selector)),
            "mean_valid_bits": float(np.mean(valid_bits)),
            "mean_valid_fraction": float(np.mean(valid_bits) / np.sum(selector)),
            "eer": float(evaluation["eer"]),
            "eer_percent": float(evaluation["eer"] * 100.0),
            "mean_mated_hd": float(np.mean(scores[same_class])),
            "mean_non_mated_hd": float(np.mean(scores[~same_class])),
            "mated_mode_offset": mated_mode_offset,
            "mated_mode_offset_count": mated_mode_count,
            "mated_mode_offset_fraction": mated_mode_fraction,
        }
        rows.append(row)
        print(
            f"  EER={row['eer']:.6f} valid_fraction={row['mean_valid_fraction']:.4f} "
            f"mated_mode_offset={mated_mode_offset} px"
        )

    best = min(rows, key=lambda item: item["eer"])
    overlay_rng = np.random.default_rng(args.seed + 7919)
    overlay_index = int(overlay_rng.integers(0, len(images)))
    plot_rows(rows, figure_path)
    preview_path = save_sample_overlay(
        images[overlay_index],
        output_dir,
        output_name,
        classifier,
        args.bands,
        args.overlay_alpha,
        discard_outer=args.discard_outer,
        discard_inner=args.discard_inner,
        band_axis=args.band_axis,
    )

    print(f"Best band: {best['band']} {args.band_axis} positions {best['position_start']}:{best['position_end']}")
    print(f"Best EER: {best['eer']:.6f} ({best['eer_percent']:.4f}%)")
    print(f"Saved figure to {figure_path}")
    print(f"Saved band preview to {preview_path}")


def build_parser():
    parser = ArgumentParser(description="Measure iris recognition EER for each iris band.")
    parser.add_argument("--dataset", dest="dataset_format", default="auto", choices=DATASET_CHOICES)
    parser.add_argument(
        "--filters",
        dest="filters",
        default=None,
        help="Optional Python filters file containing a 'filters' list.",
    )
    parser.add_argument("--rotation", type=int, default=21, help="Number of offsets to evaluate around zero.")
    parser.add_argument("--bands", type=int, default=16, help="Number of iris bands to evaluate.")
    parser.add_argument(
        "--band-axis",
        choices=["radial", "horizontal"],
        default="radial",
        help="Split bands by normalized iris rows (radial) or normalized iris columns (horizontal).",
    )
    parser.add_argument("--max-id", dest="max_identities", type=int, default=100)
    parser.add_argument("--max-img-per-id", dest="max_images_per_identity", type=int, default=20)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-name", default=None)
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--overlay-alpha", type=float, default=0.35)
    parser.add_argument("--discard-outer", type=float, default=0.0)
    parser.add_argument("--discard-inner", type=float, default=0.0)
    parser.set_defaults(
        dataset_path=None,
        func=run_eer,
    )
    return parser


def main():
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
