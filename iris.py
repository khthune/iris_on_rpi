# iris.py

import cv2 as cv
import numpy as np
import os
from dataclasses import dataclass
from pathlib import Path
from numpy.lib.stride_tricks import sliding_window_view
PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_BAND_SHAPE = (64, 512)
DEFAULT_INVALID_DILATION_KERNEL = max(1, int(os.environ.get("IRIS_UNET_INVALID_DILATION_KERNEL", 1)))
DEFAULT_INVALID_DILATION_ITERATIONS = max(1, int(os.environ.get("IRIS_UNET_INVALID_DILATION_ITERATIONS", 1)))


@dataclass(frozen=True)
class EllipseBoundary:
    center_x: float
    center_y: float
    axis_x: float
    axis_y: float
    angle_radians: float

    def sample(self, theta):
        theta = np.asarray(theta, dtype=np.float32)
        cos_theta = np.cos(theta)
        sin_theta = np.sin(theta)
        local_x = self.axis_x * cos_theta
        local_y = self.axis_y * sin_theta
        cos_angle = np.cos(self.angle_radians)
        sin_angle = np.sin(self.angle_radians)
        x = self.center_x + local_x * cos_angle - local_y * sin_angle
        y = self.center_y + local_x * sin_angle + local_y * cos_angle
        return x, y


@dataclass(frozen=True)
class PolarBoundary:
    center_x: float
    center_y: float
    radii: np.ndarray

    def sample(self, theta):
        theta = np.asarray(theta, dtype=np.float32)
        tau = 2.0 * np.pi
        wrapped = np.mod(theta, tau)
        base_theta = np.linspace(0.0, tau, len(self.radii), endpoint=False, dtype=np.float32)
        extended_theta = np.concatenate((base_theta, [tau]), axis=0)
        extended_radii = np.concatenate((self.radii, [self.radii[0]]), axis=0)
        radii = np.interp(wrapped, extended_theta, extended_radii)
        x = self.center_x + radii * np.cos(wrapped)
        y = self.center_y + radii * np.sin(wrapped)
        return x, y


def _largest_component(mask):
    mask = np.asarray(mask, dtype=np.uint8)
    if mask.ndim != 2:
        raise ValueError("Expected a 2D mask.")
    if not np.any(mask):
        return mask
    num_labels, labels, stats, _ = cv.connectedComponentsWithStats(mask, connectivity=8)
    if num_labels <= 1:
        return mask
    areas = stats[1:, cv.CC_STAT_AREA]
    largest = 1 + int(np.argmax(areas))
    return (labels == largest).astype(np.uint8)


def clean_component_mask(mask, kernel_size=5):
    mask = (np.asarray(mask) > 0).astype(np.uint8)
    if not np.any(mask):
        return mask
    kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (kernel_size, kernel_size))
    closed = cv.morphologyEx(mask, cv.MORPH_CLOSE, kernel)
    opened = cv.morphologyEx(closed, cv.MORPH_OPEN, kernel)
    return _largest_component(opened)


def fit_boundary_from_mask(mask, prefer_ellipse=True):
    component = clean_component_mask(mask)
    if not np.any(component):
        raise ValueError("Cannot fit a boundary to an empty mask.")

    contours, _ = cv.findContours(component, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_NONE)
    if not contours:
        raise ValueError("Failed to extract contours from mask.")
    contour = max(contours, key=cv.contourArea)

    if prefer_ellipse and len(contour) >= 5:
        (center_x, center_y), (diameter_x, diameter_y), angle_degrees = cv.fitEllipse(contour)
        return EllipseBoundary(
            center_x=float(center_x),
            center_y=float(center_y),
            axis_x=max(float(diameter_x) / 2.0, 1.0),
            axis_y=max(float(diameter_y) / 2.0, 1.0),
            angle_radians=np.deg2rad(angle_degrees),
        )

    (center_x, center_y), radius = cv.minEnclosingCircle(contour)
    radius = max(float(radius), 1.0)
    return EllipseBoundary(
        center_x=float(center_x),
        center_y=float(center_y),
        axis_x=radius,
        axis_y=radius,
        angle_radians=0.0,
    )


def _periodic_smooth(values, kernel_size):
    kernel_size = max(int(kernel_size), 1)
    if kernel_size <= 1:
        return values.astype(np.float32)
    kernel = np.ones(kernel_size, dtype=np.float32) / float(kernel_size)
    pad = kernel_size // 2
    extended = np.concatenate((values[-pad:], values, values[:pad]), axis=0)
    return np.convolve(extended, kernel, mode="same")[pad : pad + len(values)].astype(np.float32)


def fit_polar_boundary_from_mask(mask, center, num_angles=DEFAULT_BAND_SHAPE[1], smooth_kernel=9, fallback_to_ellipse=True):
    component = clean_component_mask(mask)
    if not np.any(component):
        raise ValueError("Cannot fit a boundary to an empty mask.")

    contours, _ = cv.findContours(component, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_NONE)
    if not contours:
        raise ValueError("Failed to extract contours from mask.")
    contour = max(contours, key=cv.contourArea).reshape(-1, 2).astype(np.float32)

    center_x, center_y = center
    dx = contour[:, 0] - center_x
    dy = contour[:, 1] - center_y
    angles = np.mod(np.arctan2(dy, dx), 2.0 * np.pi)
    radii = np.hypot(dx, dy)

    angle_to_radius = np.full(num_angles, np.nan, dtype=np.float32)
    angle_indices = np.floor(angles / (2.0 * np.pi) * num_angles).astype(np.int32) % num_angles
    for index, radius in zip(angle_indices, radii):
        if np.isnan(angle_to_radius[index]) or radius > angle_to_radius[index]:
            angle_to_radius[index] = radius

    valid = ~np.isnan(angle_to_radius)
    if valid.sum() < max(num_angles // 4, 16):
        if not fallback_to_ellipse:
            raise ValueError("Not enough contour coverage to fit a polar boundary.")
        ellipse = fit_boundary_from_mask(component, prefer_ellipse=True)
        theta = np.linspace(0.0, 2.0 * np.pi, num_angles, endpoint=False, dtype=np.float32)
        sample_x, sample_y = ellipse.sample(theta)
        radii = np.hypot(sample_x - center_x, sample_y - center_y).astype(np.float32)
        return PolarBoundary(center_x=float(center_x), center_y=float(center_y), radii=radii)

    valid_indices = np.flatnonzero(valid)
    extended_x = np.concatenate(
        (valid_indices.astype(np.float32) - num_angles, valid_indices.astype(np.float32), valid_indices.astype(np.float32) + num_angles)
    )
    extended_y = np.concatenate((angle_to_radius[valid_indices], angle_to_radius[valid_indices], angle_to_radius[valid_indices]))
    filled = np.interp(np.arange(num_angles, dtype=np.float32), extended_x, extended_y)
    return PolarBoundary(center_x=float(center_x), center_y=float(center_y), radii=_periodic_smooth(filled, smooth_kernel))


def dilate_invalid_region(valid_mask, support_mask, kernel_size=DEFAULT_INVALID_DILATION_KERNEL, iterations=DEFAULT_INVALID_DILATION_ITERATIONS):
    kernel_size = max(int(kernel_size), 1)
    iterations = max(int(iterations), 1)
    valid_bool = np.asarray(valid_mask) > 0
    support_bool = np.asarray(support_mask) > 0
    invalid_inside_support = support_bool & ~valid_bool
    if not np.any(invalid_inside_support):
        return valid_bool.astype(np.uint8) * 255

    kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (kernel_size, kernel_size))
    dilated_invalid = cv.dilate(invalid_inside_support.astype(np.uint8), kernel, iterations=iterations) > 0
    dilated_valid = support_bool & ~dilated_invalid
    return dilated_valid.astype(np.uint8) * 255


def build_valid_source_mask(
    iris_mask,
    pupil_mask,
    occlusion_mask=None,
    source_image=None,
    oversat_threshold=254,
    invalid_dilation_kernel=DEFAULT_INVALID_DILATION_KERNEL,
    invalid_dilation_iterations=DEFAULT_INVALID_DILATION_ITERATIONS,
):
    iris_mask = clean_component_mask(iris_mask)
    pupil_mask = clean_component_mask(pupil_mask)
    annulus_mask = iris_mask.astype(bool) & ~pupil_mask.astype(bool)
    valid = annulus_mask.copy()
    if occlusion_mask is not None:
        valid &= ~(np.asarray(occlusion_mask) > 0)
    if source_image is not None:
        source = np.asarray(source_image)
        valid &= source < oversat_threshold
    return dilate_invalid_region(
        valid.astype(np.uint8) * 255,
        annulus_mask.astype(np.uint8) * 255,
        kernel_size=invalid_dilation_kernel,
        iterations=invalid_dilation_iterations,
    )


def normalize_iris_from_boundaries(image, pupil_boundary, iris_boundary, valid_source_mask, band_shape=DEFAULT_BAND_SHAPE):
    if image.ndim != 2:
        raise ValueError("Expected a grayscale source image.")

    band_height, band_width = band_shape
    theta = np.linspace(0.0, 2.0 * np.pi, band_width, endpoint=False, dtype=np.float32)
    radial = np.linspace(0.0, 1.0, band_height, dtype=np.float32)[:, None]

    pupil_x, pupil_y = pupil_boundary.sample(theta)
    iris_x, iris_y = iris_boundary.sample(theta)
    map_x = (1.0 - radial) * pupil_x[None, :] + radial * iris_x[None, :]
    map_y = (1.0 - radial) * pupil_y[None, :] + radial * iris_y[None, :]

    band = cv.remap(
        image.astype(np.float32),
        map_x.astype(np.float32),
        map_y.astype(np.float32),
        interpolation=cv.INTER_LINEAR,
        borderMode=cv.BORDER_REFLECT_101,
    )
    sampled_mask = cv.remap(
        valid_source_mask.astype(np.uint8),
        map_x.astype(np.float32),
        map_y.astype(np.float32),
        interpolation=cv.INTER_NEAREST,
        borderMode=cv.BORDER_CONSTANT,
        borderValue=0,
    )
    return np.clip(band, 0, 255).astype(np.uint8), (sampled_mask > 0).astype(np.uint8) * 255


def semantic_masks_to_band(
    image,
    iris_mask,
    pupil_mask,
    occlusion_mask=None,
    band_shape=DEFAULT_BAND_SHAPE,
    prefer_ellipse=True,
    invalid_dilation_kernel=DEFAULT_INVALID_DILATION_KERNEL,
    invalid_dilation_iterations=DEFAULT_INVALID_DILATION_ITERATIONS,
):
    if occlusion_mask is not None:
        occlusion_mask = clean_component_mask(occlusion_mask, kernel_size=3)
        occlusion_mask = cv.dilate(
            occlusion_mask,
            cv.getStructuringElement(cv.MORPH_ELLIPSE, (5, 5)),
            iterations=1,
        )

    valid_source_mask = build_valid_source_mask(
        iris_mask,
        pupil_mask,
        occlusion_mask,
        source_image=image,
        oversat_threshold=254,
        invalid_dilation_kernel=invalid_dilation_kernel,
        invalid_dilation_iterations=invalid_dilation_iterations,
    )
    pupil_ellipse = fit_boundary_from_mask(pupil_mask, prefer_ellipse=prefer_ellipse)
    center = (pupil_ellipse.center_x, pupil_ellipse.center_y)
    pupil_boundary = fit_polar_boundary_from_mask(pupil_mask, center=center, num_angles=band_shape[1], smooth_kernel=7)
    iris_boundary = fit_polar_boundary_from_mask(iris_mask, center=center, num_angles=band_shape[1], smooth_kernel=17)

    if np.mean(iris_boundary.radii) <= np.mean(pupil_boundary.radii):
        raise ValueError("Iris boundary must be larger than pupil boundary.")

    return normalize_iris_from_boundaries(image, pupil_boundary, iris_boundary, valid_source_mask, band_shape=band_shape)


def _infer_pupil_mask_from_binary_iris(iris_mask):
    iris_component = clean_component_mask(iris_mask)
    if not np.any(iris_component):
        raise ValueError("Cannot infer pupil from an empty binary iris mask.")

    contours, hierarchy = cv.findContours(iris_component, cv.RETR_CCOMP, cv.CHAIN_APPROX_SIMPLE)
    if contours and hierarchy is not None:
        hierarchy = hierarchy[0]
        child_candidates = []
        for index, contour in enumerate(contours):
            parent = hierarchy[index][3]
            if parent < 0:
                continue
            area = cv.contourArea(contour)
            if area > 0:
                child_candidates.append((area, contour))
        if child_candidates:
            _, pupil_contour = max(child_candidates, key=lambda item: item[0])
            pupil_mask = np.zeros_like(iris_component, dtype=np.uint8)
            cv.drawContours(pupil_mask, [pupil_contour], -1, 1, thickness=-1)
            return clean_component_mask(pupil_mask)

    iris_boundary = fit_boundary_from_mask(iris_component, prefer_ellipse=True)
    radius = max(2, int(round(0.35 * min(iris_boundary.axis_x, iris_boundary.axis_y))))
    pupil_mask = np.zeros_like(iris_component, dtype=np.uint8)
    center = (int(round(iris_boundary.center_x)), int(round(iris_boundary.center_y)))
    cv.circle(pupil_mask, center, radius, 1, thickness=-1)
    return pupil_mask


def binary_iris_mask_to_band(
    image,
    iris_or_annulus_mask,
    band_shape=DEFAULT_BAND_SHAPE,
    prefer_ellipse=True,
    invalid_dilation_kernel=DEFAULT_INVALID_DILATION_KERNEL,
    invalid_dilation_iterations=DEFAULT_INVALID_DILATION_ITERATIONS,
):
    annulus_mask = clean_component_mask(iris_or_annulus_mask)
    pupil_mask = _infer_pupil_mask_from_binary_iris(annulus_mask)
    iris_mask = clean_component_mask((annulus_mask.astype(bool) | pupil_mask.astype(bool)).astype(np.uint8))
    return semantic_masks_to_band(
        image,
        iris_mask=iris_mask,
        pupil_mask=pupil_mask,
        occlusion_mask=None,
        band_shape=band_shape,
        prefer_ellipse=prefer_ellipse,
        invalid_dilation_kernel=invalid_dilation_kernel,
        invalid_dilation_iterations=invalid_dilation_iterations,
    )
def _resolve_segmentation_model_path():
    configured = os.environ.get(
        "SEG_PATH",
        os.environ.get(
            "IRIS_SEG_PATH",
            os.environ.get(
                "IRIS_SEGMENTATION_ONNX_PATH",
                os.environ.get(
                    "IRIS_SEGMENTATION_PATH",
                    os.environ.get("IRIS_UNET_ONNX_PATH"),
                ),
            ),
        ),
    )
    if configured is not None:
        path = Path(configured).expanduser()
        if path.is_absolute():
            return path
        cwd_candidate = path.resolve()
        if cwd_candidate.exists():
            return cwd_candidate
        return (PROJECT_ROOT / path).resolve()

    default_candidates = [
        PROJECT_ROOT / "models" / "upp_scse_mobilenetv2.onnx",
        PROJECT_ROOT / "models" / "mysegmenter.onnx",
    ]
    for candidate in default_candidates:
        if candidate.exists():
            return candidate
    return default_candidates[0]


def get_segmentation_backend_name(backend=None):
    if backend is not None:
        return str(backend)
    return "sam-iris" if _is_sam_segmentation_path() else "onnx"


UNET_ONNX_PATH = _resolve_segmentation_model_path()
UNET_INPUT_SIZE = (
    int(os.environ.get("IRIS_UNET_INPUT_WIDTH", 480)),
    int(os.environ.get("IRIS_UNET_INPUT_HEIGHT", 640)),
)
UNET_THRESHOLD = float(os.environ.get("IRIS_UNET_THRESHOLD", 0.5))
UNET_BAND_SHAPE = (
    int(os.environ.get("IRIS_UNET_BAND_HEIGHT", DEFAULT_BAND_SHAPE[0])),
    int(os.environ.get("IRIS_UNET_BAND_WIDTH", DEFAULT_BAND_SHAPE[1])),
)
_UNET_NET = None
_SAM_PREDICTOR = None


def _is_sam_segmentation_path(path=None):
    path = UNET_ONNX_PATH if path is None else Path(path)
    return path.suffix.lower() in {".pt", ".pth"}

def hamming_distance(a,b,mask1, mask2):
    diff = np.bitwise_xor(a,b)
    mask = np.bitwise_and(mask1, mask2)
    total = np.sum(np.bitwise_and(diff, mask))
    n = np.sum(mask)
    if n == 0:
        return 2.0
    return total/n

def hamming_distances(a, b, masks_a, mask_b):
    diff = np.bitwise_xor(a, b)
    mask = np.bitwise_and(masks_a, mask_b)
    scores = np.full(a.shape[0], 2.0, dtype=np.float64)
    total = np.sum(np.bitwise_and(diff, mask), axis=1)
    n = np.sum(mask, axis=1)
    valid = n > 0
    scores[valid] = total[valid] / n[valid]
    return scores


def complex_gabor_kernel(size, sigma, theta, lambd, psi, gamma, rotate_envelope=True):
    """Create a complex Gabor kernel."""
    y_size, x_size = size
    x_half_size = x_size // 2
    y_half_size = y_size // 2
    y, x = np.meshgrid(np.linspace(-y_half_size, y_half_size, y_size),
                       np.linspace(-x_half_size, x_half_size, x_size))
    
    x_theta = x * np.cos(theta) + y * np.sin(theta)
    y_theta = -x * np.sin(theta) + y * np.cos(theta)

    if rotate_envelope:
        gaussian = np.exp(-0.5 * (x_theta**2 + (gamma**2) * y_theta**2) / sigma**2)
    else:
        gaussian = np.exp(-0.5 * (x**2 + (gamma**2) * y**2) / sigma**2)

    complex_sinusoid = np.exp(1j * (2 * np.pi * x_theta / lambd + psi))

    gabor = gaussian * complex_sinusoid
    return gabor

def complex_to_bits(z, mask_bit_list):
    real = (z.real >= 0).astype(np.bool)
    imag = (z.imag >= 0).astype(np.bool)
    result = np.empty(z.size*2, dtype=np.bool)
    mask_bits = np.empty(z.size*2, dtype=np.bool)
    result[0::2] = real
    result[1::2] = imag 
    mask_bits[0::2] = mask_bit_list
    mask_bits[1::2] = mask_bit_list
    return result, mask_bits


def _apply_filter_grid_batch(iris, filter_real, filter_imag, stride, start_positions, mask=None):
    x_stride, y_stride = stride
    iris_h, iris_w = iris.shape
    filter_h, filter_w = filter_real.shape
    num_x = iris_w // x_stride

    start_positions = np.asarray(start_positions, dtype=np.int64)
    x_starts = start_positions[:, 0]
    y_starts = start_positions[:, 1]
    if not np.all(y_starts == y_starts[0]):
        raise ValueError("All start positions must share the same y coordinate.")

    x_half = filter_w // 2
    y_half = filter_h // 2
    y_bottom = filter_h - y_half - 1
    x_right = filter_w - x_half - 1
    num_y = iris_h // y_stride
    y_positions = y_starts[0] + y_stride * np.arange(num_y)
    window_tops = y_positions - y_half
    valid_y = (window_tops >= 0) & ((window_tops + filter_h) <= iris_h)

    result_grid = np.zeros((num_y, len(start_positions), num_x), dtype=np.complex64)
    mask_grid = np.zeros((num_y, len(start_positions), num_x), dtype=bool)
    if not np.any(valid_y) or filter_h > iris_h:
        return (
            result_grid.transpose(1, 2, 0).reshape(len(start_positions), -1),
            mask_grid.transpose(1, 2, 0).reshape(len(start_positions), -1),
        )

    wrapped_iris = np.concatenate(
        (
            iris[:, -x_half:] if x_half else iris[:, :0],
            iris,
            iris[:, :x_right] if x_right else iris[:, :0],
        ),
        axis=1,
    )

    iris_windows = sliding_window_view(wrapped_iris, (filter_h, filter_w))
    sampled_rows = iris_windows[window_tops[valid_y]]
    x_positions = (x_starts[:, None] + x_stride * np.arange(num_x)) % iris_w
    sampled_iris = sampled_rows[:, x_positions, :, :]

    result_real = np.einsum("yoxij,ij->yox", sampled_iris, filter_real, optimize=True)
    result_imag = np.einsum("yoxij,ij->yox", sampled_iris, filter_imag, optimize=True)
    result_grid[valid_y] = result_real + result_imag * 1j

    if mask is None:
        mask_grid[valid_y] = True
        return (
            result_grid.transpose(1, 2, 0).reshape(len(start_positions), -1),
            mask_grid.transpose(1, 2, 0).reshape(len(start_positions), -1),
        )

    wrapped_mask = np.concatenate(
        (
            mask[:, -x_half:] if x_half else mask[:, :0],
            mask,
            mask[:, :x_right] if x_right else mask[:, :0],
        ),
        axis=1,
    )
    mask_windows = sliding_window_view(wrapped_mask, (filter_h, filter_w))
    sampled_mask_rows = mask_windows[window_tops[valid_y]]
    sampled_mask = sampled_mask_rows[:, x_positions, :, :]
    mask_grid[valid_y] = np.all(sampled_mask == 255, axis=(3, 4))
    return (
        result_grid.transpose(1, 2, 0).reshape(len(start_positions), -1),
        mask_grid.transpose(1, 2, 0).reshape(len(start_positions), -1),
    )

def _resolve_dnn_backend(name):
    mapping = {
        "opencv": cv.dnn.DNN_BACKEND_OPENCV,
    }
    if hasattr(cv.dnn, "DNN_BACKEND_CUDA"):
        mapping["cuda"] = cv.dnn.DNN_BACKEND_CUDA
    return mapping.get(name)


def _resolve_dnn_target(name):
    mapping = {
        "cpu": cv.dnn.DNN_TARGET_CPU,
    }
    if hasattr(cv.dnn, "DNN_TARGET_CUDA"):
        mapping["cuda"] = cv.dnn.DNN_TARGET_CUDA
    if hasattr(cv.dnn, "DNN_TARGET_CUDA_FP16"):
        mapping["cuda_fp16"] = cv.dnn.DNN_TARGET_CUDA_FP16
    return mapping.get(name)


def _get_unet_net():
    global _UNET_NET
    if _UNET_NET is not None:
        return _UNET_NET
    if _is_sam_segmentation_path():
        raise ValueError(
            f"SEG_PATH points to a SAM/Iris-SAM PyTorch checkpoint, not ONNX: {UNET_ONNX_PATH}"
        )
    if not UNET_ONNX_PATH.exists():
        raise FileNotFoundError(
            f"Segmentation ONNX model not found at '{UNET_ONNX_PATH}'. "
            "Set IRIS_SEGMENTATION_ONNX_PATH or place the model there."
        )
    net = cv.dnn.readNetFromONNX(str(UNET_ONNX_PATH))
    backend_name = os.environ.get("IRIS_UNET_DNN_BACKEND", "opencv").strip().lower()
    target_name = os.environ.get("IRIS_UNET_DNN_TARGET", "cpu").strip().lower()
    backend = _resolve_dnn_backend(backend_name)
    target = _resolve_dnn_target(target_name)
    if backend is not None:
        net.setPreferableBackend(backend)
    if target is not None:
        net.setPreferableTarget(target)
    _UNET_NET = net
    return net


def _resolve_sam_device(torch_module):
    configured = os.environ.get("IRIS_SAM_DEVICE", "auto").strip().lower()
    if configured and configured != "auto":
        return configured
    if torch_module.cuda.is_available():
        return "cuda"
    if hasattr(torch_module.backends, "mps") and torch_module.backends.mps.is_available():
        return "mps"
    return "cpu"


def _get_sam_predictor():
    global _SAM_PREDICTOR
    if _SAM_PREDICTOR is not None:
        return _SAM_PREDICTOR
    if not UNET_ONNX_PATH.exists():
        raise FileNotFoundError(
            f"Iris-SAM checkpoint not found at '{UNET_ONNX_PATH}'. "
            "Set SEG_PATH to the downloaded Iris-SAM model.pt checkpoint."
        )
    try:
        import torch
        from segment_anything import SamPredictor, sam_model_registry
    except ImportError as exc:
        raise ImportError(
            "Iris-SAM needs PyTorch and Meta's segment-anything package. Install with: "
            "python3 -m pip install git+https://github.com/facebookresearch/segment-anything.git"
        ) from exc

    model_type = os.environ.get("IRIS_SAM_MODEL_TYPE", "vit_h").strip()
    if model_type not in sam_model_registry:
        available = ", ".join(sorted(sam_model_registry))
        raise ValueError(f"Unknown IRIS_SAM_MODEL_TYPE={model_type!r}. Available: {available}")
    device = _resolve_sam_device(torch)
    model = sam_model_registry[model_type](checkpoint=None)
    checkpoint = torch.load(str(UNET_ONNX_PATH), map_location="cpu")
    if isinstance(checkpoint, dict) and "model" in checkpoint:
        checkpoint = checkpoint["model"]
    model.load_state_dict(checkpoint)
    model.to(device=device)
    model.eval()
    _SAM_PREDICTOR = SamPredictor(model)
    return _SAM_PREDICTOR


def _sam_prompt_box(image_shape):
    image_h, image_w = image_shape[:2]
    scale = float(os.environ.get("IRIS_SAM_BOX_SCALE", 1.0))
    scale = min(max(scale, 0.05), 1.0)
    box_w = image_w * scale
    box_h = image_h * scale
    x1 = max(0.0, (image_w - box_w) / 2.0)
    y1 = max(0.0, (image_h - box_h) / 2.0)
    x2 = min(float(image_w - 1), x1 + box_w - 1.0)
    y2 = min(float(image_h - 1), y1 + box_h - 1.0)
    return np.array([x1, y1, x2, y2], dtype=np.float32)


def _predict_sam_iris_annulus_mask(img):
    source_gray = _prepare_segmentation_gray(img)
    if img.ndim == 2:
        rgb = cv.cvtColor(img, cv.COLOR_GRAY2RGB)
    else:
        rgb = cv.cvtColor(img, cv.COLOR_BGR2RGB)

    predictor = _get_sam_predictor()
    predictor.set_image(rgb)
    masks, _scores, _logits = predictor.predict(
        point_coords=None,
        point_labels=None,
        box=_sam_prompt_box(rgb.shape),
        multimask_output=False,
    )
    return source_gray, masks[0].astype(bool)


def _sigmoid_if_needed(output):
    output = np.asarray(output, dtype=np.float32)
    if output.min() < 0.0 or output.max() > 1.0:
        return 1.0 / (1.0 + np.exp(-output))
    return output


def _prepare_segmentation_gray(img):
    return img if img.ndim == 2 else cv.cvtColor(img, cv.COLOR_BGR2GRAY)

def _forward_segmentation_model(img):
    source_gray = _prepare_segmentation_gray(img)
    resized = cv.resize(source_gray, UNET_INPUT_SIZE, interpolation=cv.INTER_LINEAR).astype(np.float32) / 255.0
    rgb = np.repeat(resized[:, :, None], 3, axis=2)
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    normalized = ((rgb - mean) / std).transpose(2, 0, 1)[None, :, :, :]

    net = _get_unet_net()
    net.setInput(normalized)
    output = net.forward()
    return source_gray, output


def predict_unet_masks(img):
    if _is_sam_segmentation_path():
        source_gray, annulus_mask = _predict_sam_iris_annulus_mask(img)
        pupil_mask = _infer_pupil_mask_from_binary_iris(annulus_mask)
        iris_mask = clean_component_mask((annulus_mask.astype(bool) | pupil_mask.astype(bool)).astype(np.uint8))
        eyelash_mask = np.zeros_like(iris_mask, dtype=bool)
        return source_gray, iris_mask.astype(bool), pupil_mask.astype(bool), eyelash_mask

    source_gray, output = _forward_segmentation_model(img)
    if output.ndim != 4:
        raise RuntimeError(f"Unexpected segmentation output shape: {output.shape}")

    if output.shape[1] == 1:
        probability = _sigmoid_if_needed(output[0, 0])
        image_h, image_w = source_gray.shape
        resized_probability = cv.resize(probability, (image_w, image_h), interpolation=cv.INTER_LINEAR)
        annulus_mask = resized_probability >= UNET_THRESHOLD
        pupil_mask = _infer_pupil_mask_from_binary_iris(annulus_mask)
        iris_mask = clean_component_mask((annulus_mask.astype(bool) | pupil_mask.astype(bool)).astype(np.uint8))
        eyelash_mask = np.zeros_like(iris_mask, dtype=bool)
        return source_gray, iris_mask.astype(bool), pupil_mask.astype(bool), eyelash_mask

    if output.shape[1] < 3:
        raise RuntimeError(f"Unexpected multiclass segmentation output shape: {output.shape}")

    probabilities = _sigmoid_if_needed(output[0])
    image_h, image_w = source_gray.shape
    resized_probs = np.stack(
        [
            cv.resize(probabilities[index], (image_w, image_h), interpolation=cv.INTER_LINEAR)
            for index in range(probabilities.shape[0])
        ],
        axis=0,
    )

    # Worldcoin's published model card lists classes: eyeball, iris, pupil, eyelashes.
    iris_mask = resized_probs[1] >= UNET_THRESHOLD
    pupil_mask = resized_probs[2] >= UNET_THRESHOLD
    eyelash_mask = resized_probs[3] >= UNET_THRESHOLD if resized_probs.shape[0] > 3 else np.zeros_like(iris_mask)
    return source_gray, iris_mask, pupil_mask, eyelash_mask


def predict_binary_iris_mask(img):
    if _is_sam_segmentation_path():
        return _predict_sam_iris_annulus_mask(img)

    source_gray, output = _forward_segmentation_model(img)
    if output.ndim != 4 or output.shape[1] != 1:
        raise RuntimeError(f"Unexpected binary segmentation output shape: {output.shape}")

    probability = _sigmoid_if_needed(output[0, 0])
    image_h, image_w = source_gray.shape
    resized_probability = cv.resize(probability, (image_w, image_h), interpolation=cv.INTER_LINEAR)
    iris_mask = resized_probability >= UNET_THRESHOLD
    return source_gray, iris_mask


def _segment_with_unet(img):
    if _is_sam_segmentation_path():
        source_gray, annulus_mask = _predict_sam_iris_annulus_mask(img)
        return binary_iris_mask_to_band(
            source_gray,
            annulus_mask,
            band_shape=UNET_BAND_SHAPE,
            prefer_ellipse=True,
        )

    source_gray, output = _forward_segmentation_model(img)
    if output.ndim != 4:
        raise RuntimeError(f"Unexpected segmentation output shape: {output.shape}")

    if output.shape[1] == 1:
        probability = _sigmoid_if_needed(output[0, 0])
        image_h, image_w = source_gray.shape
        resized_probability = cv.resize(probability, (image_w, image_h), interpolation=cv.INTER_LINEAR)
        return binary_iris_mask_to_band(
            source_gray,
            resized_probability >= UNET_THRESHOLD,
            band_shape=UNET_BAND_SHAPE,
            prefer_ellipse=True,
        )

    if output.shape[1] < 3:
        raise RuntimeError(f"Unexpected multiclass segmentation output shape: {output.shape}")

    probabilities = _sigmoid_if_needed(output[0])
    image_h, image_w = source_gray.shape
    resized_probs = np.stack(
        [
            cv.resize(probabilities[index], (image_w, image_h), interpolation=cv.INTER_LINEAR)
            for index in range(probabilities.shape[0])
        ],
        axis=0,
    )
    iris_mask = resized_probs[1] >= UNET_THRESHOLD
    pupil_mask = resized_probs[2] >= UNET_THRESHOLD
    eyelash_mask = resized_probs[3] >= UNET_THRESHOLD if resized_probs.shape[0] > 3 else np.zeros_like(iris_mask)
    return semantic_masks_to_band(
        source_gray,
        iris_mask=iris_mask,
        pupil_mask=pupil_mask,
        occlusion_mask=eyelash_mask,
        band_shape=UNET_BAND_SHAPE,
        prefer_ellipse=True,
    )


def get_iris_band(img, backend=None):
    get_segmentation_backend_name(backend)
    return _segment_with_unet(img)

class IrisClassifier():
    def __init__(self, filters) -> None:
        self.init_filters(filters)
        
    def init_filters(self, filters):
        self._filters = [] 
        for filter_settings in filters:
            filter = complex_gabor_kernel(**filter_settings["filter"])
            real_filter = np.real(filter)
            imag_filter = np.imag(filter)
            real_filter = real_filter - np.mean(real_filter)
            imag_filter = imag_filter - np.mean(imag_filter)
            self._filters.append((real_filter, imag_filter))
        self._filter_settings = filters

    def __call__(self, iris1, iris2, mask1, mask2, rotation=6, offset=0):
        bits_1, mask_1, _ = self.get_iris_code(iris1, mask1)
        return self.compare_iris_code_and_iris(iris2, bits_1, mask2, mask_1, rotation=rotation, offset=offset)
    
    def _encode_iris_offsets(self, iris, mask=None, offsets=(0,)):
        offsets = np.asarray(offsets, dtype=np.int64)
        bit_chunks = []
        filter_chunks = []
        mask_chunks = []

        for i, (filter_real, filter_imag) in enumerate(self._filters):
            start_x, start_y = self._filter_settings[i]["start_position"]
            start_positions = np.column_stack(
                (
                    start_x + offsets,
                    np.full(offsets.shape, start_y, dtype=np.int64),
                )
            )
            results, mask_bit_lists = _apply_filter_grid_batch(
                iris,
                filter_real,
                filter_imag,
                self._filter_settings[i]["stride"],
                start_positions,
                mask,
            )

            filter_bits = []
            filter_masks = []
            for result, mask_bit_list in zip(results, mask_bit_lists):
                new_bits, mask_bits = complex_to_bits(result, mask_bit_list)
                filter_bits.append(new_bits)
                filter_masks.append(mask_bits)

            bit_chunks.append(np.stack(filter_bits, axis=0))
            mask_chunks.append(np.stack(filter_masks, axis=0))
            filter_ids = np.full((len(offsets), filter_bits[0].shape[0]), i, dtype=np.uint8)
            filter_chunks.append(filter_ids)

        bits = np.concatenate(bit_chunks, axis=1)
        filters = np.concatenate(filter_chunks, axis=1)
        mask_bits = np.concatenate(mask_chunks, axis=1)
        return bits, mask_bits, filters

    def get_iris_code(self, iris, mask=None, offset=0):
        bits, mask_bits, filters = self._encode_iris_offsets(iris, mask, offsets=(offset,))
        return bits[0], mask_bits[0], filters[0]

    def get_iris_codes(self, iris, mask=None, offsets=(0,)):
        bits, mask_bits, filters = self._encode_iris_offsets(iris, mask, offsets=offsets)
        return bits, mask_bits, filters
    
    def compare_iris_code_and_iris(self, iris, iris_code, iris_mask, iris_code_mask, rotation=None, offset=0):
        if rotation is None:
            bits, mask, _ = self.get_iris_code(iris, iris_mask, offset=offset)
            return (hamming_distance(bits, iris_code, mask, iris_code_mask), 0)
        offsets = np.arange(rotation) - rotation // 2
        bits, masks, _ = self.get_iris_codes(iris, iris_mask, offsets=offsets)
        scores = hamming_distances(bits, iris_code, masks, iris_code_mask)
        return (np.min(scores), np.argmin(scores)-rotation//2)
