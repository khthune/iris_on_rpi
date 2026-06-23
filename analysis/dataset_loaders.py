# dataset_loaders.py

from pathlib import Path
import random
import re

import cv2 as cv
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATASETS_ROOT = PROJECT_ROOT.parent / "datasets"

CASIA_V1_PATH = DATASETS_ROOT / "CASIA Version.1" / "CASIA Iris Image Database (version 1.0)"
CASIA_V3_INTERVAL_PATH = DATASETS_ROOT / "CASIA-IrisV3" / "CASIA-IrisV3-Interval"
CASIA_V4_INTERVAL_PATH = DATASETS_ROOT / "CASIA-IrisV4-Interval"
CASIA_DISTANCE_PATH = DATASETS_ROOT / "CASIA-Iris-Distance"
CASIA_1000_PATH = DATASETS_ROOT / "CASIA-Iris-Thousand"
CASIA_V3_LAMP_PATH = DATASETS_ROOT / "CASIA-IrisV3" / "CASIA-IrisV3-Lamp"
CASIA_V3_TWINS_PATH = DATASETS_ROOT / "CASIA-IrisV3" / "CASIA-IrisV3-Twins"
IITD_PATH = DATASETS_ROOT / "IIT Delhi Iris Database"
MMU_PATH = DATASETS_ROOT / "MMU" / "MMU Iris Database"
MMU2_PATH = DATASETS_ROOT / "MMU" / "MMU2 Iris Database" / "MMU2 Iris Database"

DATASET_CHOICES = [
    "auto",
    "casia-v1",
    "casia-v3-interval",
    "casia-v4-interval",
    "casia-distance",
    "casia-1000",
    "casia-v3-lamp",
    "casia-v3-twins",
    "iitd",
    "mmu",
    "mmu2",
]


def dataset_output_slug(dataset_format):
    return str(dataset_format).replace("-", "")


def sample_dataset(
    images,
    labels,
    image_names,
    max_samples=None,
    max_identities=None,
    max_images_per_identity=None,
    seed=0,
):
    if max_samples is None and max_identities is None and max_images_per_identity is None:
        return images, labels, image_names

    rng = random.Random(seed)
    label_to_indices = {}
    for index, label in enumerate(labels):
        label_to_indices.setdefault(label, []).append(index)

    selected_labels = list(label_to_indices.keys())
    if max_identities is not None and max_identities < len(selected_labels):
        selected_labels = rng.sample(selected_labels, max_identities)

    selected_indices = []
    for label in selected_labels:
        indices = list(label_to_indices[label])
        if max_images_per_identity is not None and max_images_per_identity < len(indices):
            indices = rng.sample(indices, max_images_per_identity)
        selected_indices.extend(indices)

    if max_samples is not None and max_samples < len(selected_indices):
        selected_indices = rng.sample(selected_indices, max_samples)

    selected_indices.sort()
    sampled_images = [images[index] for index in selected_indices]
    sampled_labels = labels[selected_indices]
    sampled_image_names = image_names[selected_indices]
    return sampled_images, sampled_labels, sampled_image_names


def _build_index(dataset_dir, image_paths, label_builder):
    paths = []
    labels = []
    names = []
    for image_path in image_paths:
        paths.append(image_path)
        labels.append(label_builder(image_path))
        names.append(str(image_path.relative_to(dataset_dir)))
    return paths, np.array(labels), np.array(names)


def _load_images_with_labels(dataset_dir, image_paths, label_builder, imread_flag=cv.IMREAD_GRAYSCALE):
    paths, labels, names = _build_index(dataset_dir, image_paths, label_builder)
    images = []
    for image_path in paths:
        image = cv.imread(str(image_path), imread_flag)
        if image is None:
            raise FileNotFoundError(f"Failed to load image '{image_path}'.")
        images.append(image)
    return images, labels, names

    return images, np.array(labels), np.array(names)


def load_casia_v1(dataset_path):
    dataset_dir = Path(dataset_path).expanduser().resolve()
    if not dataset_dir.exists():
        raise FileNotFoundError(f"Dataset directory does not exist: {dataset_dir}")

    image_paths = sorted(path for path in dataset_dir.glob("*/*/*.bmp"))
    if not image_paths:
        raise FileNotFoundError(
            "No CASIA V1 images were found. Expected files like XXX/S/XXX_S_Y.bmp."
        )

    return _load_images_with_labels(
        dataset_dir,
        image_paths,
        lambda image_path: f"{image_path.parent.parent.name}_{image_path.parent.name}",
    )


def load_casia_v3_interval(dataset_path):
    dataset_dir = Path(dataset_path).expanduser().resolve()
    if not dataset_dir.exists():
        raise FileNotFoundError(f"Dataset directory does not exist: {dataset_dir}")

    image_paths = sorted(path for path in dataset_dir.glob("*/*/*.jpg") if path.name.lower() != "thumbs.db")
    if not image_paths:
        raise FileNotFoundError(
            "No CASIA-IrisV3 Interval images were found. Expected files like XXX/L/S1XXXL01.jpg."
        )

    return _load_images_with_labels(
        dataset_dir,
        image_paths,
        lambda image_path: f"{image_path.parent.parent.name}_{image_path.parent.name}",
    )


def load_casia_v4_interval(dataset_path):
    dataset_dir = Path(dataset_path).expanduser().resolve()
    if not dataset_dir.exists():
        raise FileNotFoundError(f"Dataset directory does not exist: {dataset_dir}")

    image_paths = sorted(path for path in dataset_dir.glob("*/*/*.jpg") if path.name.lower() != "thumbs.db")
    if not image_paths:
        raise FileNotFoundError(
            "No CASIA-IrisV4 Interval images were found. Expected files like XXX/L/S1XXXL01.jpg."
        )

    return _load_images_with_labels(
        dataset_dir,
        image_paths,
        lambda image_path: f"{image_path.parent.parent.name}_{image_path.parent.name}",
    )


def load_casia_distance(dataset_path):
    dataset_dir = Path(dataset_path).expanduser().resolve()
    if not dataset_dir.exists():
        raise FileNotFoundError(f"Dataset directory does not exist: {dataset_dir}")

    image_paths = sorted(path for path in dataset_dir.glob("*/*.jpg") if path.name.lower() != "thumbs.db")
    if not image_paths:
        raise FileNotFoundError(
            "No CASIA-Iris-Distance images were found. Expected files like YYY/S4YYYDNN.jpg."
        )

    def label_builder(image_path):
        match = re.match(r"S4(?P<subject>\d+)D\d+$", image_path.stem, flags=re.IGNORECASE)
        if not match:
            raise ValueError(f"Unexpected CASIA-Iris-Distance filename format: {image_path.name}")
        subject = match.group("subject")
        return subject

    return _load_images_with_labels(dataset_dir, image_paths, label_builder)


def load_casia_1000(dataset_path):
    dataset_dir = Path(dataset_path).expanduser().resolve()
    if not dataset_dir.exists():
        raise FileNotFoundError(f"Dataset directory does not exist: {dataset_dir}")

    image_paths = sorted(path for path in dataset_dir.glob("*/*/*.jpg") if path.name.lower() != "thumbs.db")
    if not image_paths:
        raise FileNotFoundError(
            "No CASIA-Iris-Thousand images were found. Expected files like YYY/L/S5YYYLNN.jpg."
        )

    def label_builder(image_path):
        side = image_path.parent.name.upper()
        if side not in {"L", "R"}:
            raise ValueError(f"Unexpected CASIA-Iris-Thousand eye directory: {image_path}")
        return f"{image_path.parent.parent.name}_{side}"

    return _load_images_with_labels(dataset_dir, image_paths, label_builder)


def load_casia_v3_lamp(dataset_path):
    dataset_dir = Path(dataset_path).expanduser().resolve()
    if not dataset_dir.exists():
        raise FileNotFoundError(f"Dataset directory does not exist: {dataset_dir}")

    image_paths = sorted(path for path in dataset_dir.glob("*/*/*.jpg") if path.name.lower() != "thumbs.db")
    if not image_paths:
        raise FileNotFoundError(
            "No CASIA-IrisV3 Lamp images were found. Expected files like XXX/L/S2XXXL01.jpg."
        )

    return _load_images_with_labels(
        dataset_dir,
        image_paths,
        lambda image_path: f"{image_path.parent.parent.name}_{image_path.parent.name}",
    )


def load_casia_v3_twins(dataset_path):
    dataset_dir = Path(dataset_path).expanduser().resolve()
    if not dataset_dir.exists():
        raise FileNotFoundError(f"Dataset directory does not exist: {dataset_dir}")

    image_paths = sorted(path for path in dataset_dir.glob("*/*/*.jpg") if path.name.lower() != "thumbs.db")
    if not image_paths:
        raise FileNotFoundError(
            "No CASIA-IrisV3 Twins images were found. Expected files like XX/1L/S3XXXL01.jpg."
        )

    return _load_images_with_labels(
        dataset_dir,
        image_paths,
        lambda image_path: f"{image_path.parent.parent.name}_{image_path.parent.name}",
    )


def load_iitd(dataset_path):
    dataset_dir = Path(dataset_path).expanduser().resolve()
    if not dataset_dir.exists():
        raise FileNotFoundError(f"Dataset directory does not exist: {dataset_dir}")

    image_paths = sorted(
        path
        for path in dataset_dir.glob("[0-9][0-9][0-9]/*.bmp")
        if path.name.lower() not in {"thumbs.db", ".ds_store"}
    )
    if not image_paths:
        raise FileNotFoundError(
            "No IITD images were found. Expected files like 001/01_L.bmp or 014/06_R.bmp."
        )

    def label_builder(image_path):
        match = re.match(r"\d+_([LR])$", image_path.stem, flags=re.IGNORECASE)
        if not match:
            raise ValueError(f"Unexpected IITD filename format: {image_path.name}")
        return f"{image_path.parent.name}_{match.group(1).upper()}"

    return _load_images_with_labels(dataset_dir, image_paths, label_builder)


def load_mmu(dataset_path):
    dataset_dir = Path(dataset_path).expanduser().resolve()
    if not dataset_dir.exists():
        raise FileNotFoundError(f"Dataset directory does not exist: {dataset_dir}")

    image_paths = sorted(
        path for path in dataset_dir.glob("*/*/*.bmp") if path.name.lower() not in {"thumbs.db", ".ds_store"}
    )
    if not image_paths:
        raise FileNotFoundError(
            "No MMU images were found. Expected files like 1/left/aeval1.bmp or 1/right/aevar1.bmp."
        )

    def label_builder(image_path):
        side = image_path.parent.name.lower()
        if side not in {"left", "right"}:
            raise ValueError(f"Unexpected MMU directory structure: {image_path}")
        side_code = "L" if side == "left" else "R"
        return f"{image_path.parent.parent.name}_{side_code}"

    return _load_images_with_labels(dataset_dir, image_paths, label_builder)


def load_mmu2(dataset_path):
    dataset_dir = Path(dataset_path).expanduser().resolve()
    if not dataset_dir.exists():
        raise FileNotFoundError(f"Dataset directory does not exist: {dataset_dir}")

    image_paths = sorted(
        path for path in dataset_dir.glob("*.bmp") if path.name.lower() not in {"thumbs.db", ".ds_store"}
    )
    if not image_paths:
        raise FileNotFoundError(
            "No MMU2 images were found. Expected files like 010101.bmp or 1000205.bmp."
        )

    def label_builder(image_path):
        stem = image_path.stem
        if len(stem) < 6 or not stem.isdigit():
            raise ValueError(f"Unexpected MMU2 filename format: {image_path.name}")
        subject = stem[:-4]
        eye_code = stem[-4:-2]
        if eye_code not in {"01", "02"}:
            raise ValueError(f"Unexpected MMU2 eye code in filename: {image_path.name}")
        side_code = "L" if eye_code == "01" else "R"
        return f"{subject}_{side_code}"

    return _load_images_with_labels(dataset_dir, image_paths, label_builder)


def resolve_dataset(dataset_path, dataset_format):
    if dataset_path:
        resolved_path = Path(dataset_path).expanduser().resolve()
        if not resolved_path.exists():
            raise FileNotFoundError(f"Dataset directory does not exist: {resolved_path}")

        if dataset_format == "auto":
            name = resolved_path.name.lower()
            if any(resolved_path.glob("*/*/*.bmp")):
                if "casia" in name and "version.1" in str(resolved_path).lower():
                    return resolved_path, "casia-v1"
                if "iit" in name:
                    return resolved_path, "iitd"
                if "mmu iris database" in name:
                    return resolved_path, "mmu"
            if any(resolved_path.glob("*/*/*.jpg")):
                if "thousand" in name or "1000" in name:
                    return resolved_path, "casia-1000"
                if "distance" in name:
                    return resolved_path, "casia-distance"
                if "v4" in name and "interval" in name:
                    return resolved_path, "casia-v4-interval"
                if "interval" in name:
                    return resolved_path, "casia-v3-interval"
                if "lamp" in name:
                    return resolved_path, "casia-v3-lamp"
                if "twins" in name:
                    return resolved_path, "casia-v3-twins"
                return resolved_path, "casia-v3-interval"
            if any(resolved_path.glob("*/*.jpg")):
                if "distance" in name:
                    return resolved_path, "casia-distance"
            if any(resolved_path.glob("*.bmp")):
                if "mmu2 iris database" in name:
                    return resolved_path, "mmu2"
            raise FileNotFoundError(
                f"Could not infer dataset format from '{resolved_path}'. "
                "Use --dataset explicitly."
            )

        return resolved_path, dataset_format

    if dataset_format in ("auto", "casia-v1") and CASIA_V1_PATH.exists():
        return CASIA_V1_PATH, "casia-v1"
    if dataset_format in ("auto", "casia-v3-interval") and CASIA_V3_INTERVAL_PATH.exists():
        return CASIA_V3_INTERVAL_PATH, "casia-v3-interval"
    if dataset_format in ("auto", "casia-v4-interval") and CASIA_V4_INTERVAL_PATH.exists():
        return CASIA_V4_INTERVAL_PATH, "casia-v4-interval"
    if dataset_format in ("auto", "casia-distance") and CASIA_DISTANCE_PATH.exists():
        return CASIA_DISTANCE_PATH, "casia-distance"
    if dataset_format in ("auto", "casia-1000") and CASIA_1000_PATH.exists():
        return CASIA_1000_PATH, "casia-1000"
    if dataset_format in ("auto", "casia-v3-lamp") and CASIA_V3_LAMP_PATH.exists():
        return CASIA_V3_LAMP_PATH, "casia-v3-lamp"
    if dataset_format in ("auto", "casia-v3-twins") and CASIA_V3_TWINS_PATH.exists():
        return CASIA_V3_TWINS_PATH, "casia-v3-twins"
    if dataset_format in ("auto", "iitd") and IITD_PATH.exists():
        return IITD_PATH, "iitd"
    if dataset_format in ("auto", "mmu") and MMU_PATH.exists():
        return MMU_PATH, "mmu"
    if dataset_format in ("auto", "mmu2") and MMU2_PATH.exists():
        return MMU2_PATH, "mmu2"

    raise FileNotFoundError(
        "Could not find a default dataset path. "
        "Pass --dataset-path explicitly or place the dataset in one of these locations:\n"
        f"{CASIA_V1_PATH}\n"
        f"{CASIA_V3_INTERVAL_PATH}\n"
        f"{CASIA_V4_INTERVAL_PATH}\n"
        f"{CASIA_DISTANCE_PATH}\n"
        f"{CASIA_1000_PATH}\n"
        f"{CASIA_V3_LAMP_PATH}\n"
        f"{CASIA_V3_TWINS_PATH}\n"
        f"{IITD_PATH}\n"
        f"{MMU_PATH}\n"
        f"{MMU2_PATH}"
    )


def load_dataset(dataset_path, dataset_format):
    if dataset_format == "casia-v1":
        return load_casia_v1(dataset_path)
    if dataset_format == "casia-v3-interval":
        return load_casia_v3_interval(dataset_path)
    if dataset_format == "casia-v4-interval":
        return load_casia_v4_interval(dataset_path)
    if dataset_format == "casia-distance":
        return load_casia_distance(dataset_path)
    if dataset_format == "casia-1000":
        return load_casia_1000(dataset_path)
    if dataset_format == "casia-v3-lamp":
        return load_casia_v3_lamp(dataset_path)
    if dataset_format == "casia-v3-twins":
        return load_casia_v3_twins(dataset_path)
    if dataset_format == "iitd":
        return load_iitd(dataset_path)
    if dataset_format == "mmu":
        return load_mmu(dataset_path)
    if dataset_format == "mmu2":
        return load_mmu2(dataset_path)
    raise ValueError(f"Unsupported dataset format: {dataset_format}")


def load_dataset_index(dataset_path, dataset_format):
    dataset_dir = Path(dataset_path).expanduser().resolve()
    if dataset_format == "casia-v1":
        image_paths = sorted(path for path in dataset_dir.glob("*/*/*.bmp"))
        return _build_index(
            dataset_dir,
            image_paths,
            lambda image_path: f"{image_path.parent.parent.name}_{image_path.parent.name}",
        )
    if dataset_format == "casia-v3-interval":
        image_paths = sorted(path for path in dataset_dir.glob("*/*/*.jpg") if path.name.lower() != "thumbs.db")
        return _build_index(
            dataset_dir,
            image_paths,
            lambda image_path: f"{image_path.parent.parent.name}_{image_path.parent.name}",
        )
    if dataset_format == "casia-v4-interval":
        image_paths = sorted(path for path in dataset_dir.glob("*/*/*.jpg") if path.name.lower() != "thumbs.db")
        return _build_index(
            dataset_dir,
            image_paths,
            lambda image_path: f"{image_path.parent.parent.name}_{image_path.parent.name}",
        )
    if dataset_format == "casia-distance":
        image_paths = sorted(path for path in dataset_dir.glob("*/*.jpg") if path.name.lower() != "thumbs.db")
        def label_builder(image_path):
            match = re.match(r"S4(?P<subject>\d+)D\d+$", image_path.stem, flags=re.IGNORECASE)
            if not match:
                raise ValueError(f"Unexpected CASIA-Iris-Distance filename format: {image_path.name}")
            return match.group("subject")
        return _build_index(dataset_dir, image_paths, label_builder)
    if dataset_format == "casia-1000":
        image_paths = sorted(path for path in dataset_dir.glob("*/*/*.jpg") if path.name.lower() != "thumbs.db")
        def label_builder(image_path):
            side = image_path.parent.name.upper()
            if side not in {"L", "R"}:
                raise ValueError(f"Unexpected CASIA-Iris-Thousand eye directory: {image_path}")
            return f"{image_path.parent.parent.name}_{side}"
        return _build_index(dataset_dir, image_paths, label_builder)
    if dataset_format == "casia-v3-lamp":
        image_paths = sorted(path for path in dataset_dir.glob("*/*/*.jpg") if path.name.lower() != "thumbs.db")
        return _build_index(
            dataset_dir,
            image_paths,
            lambda image_path: f"{image_path.parent.parent.name}_{image_path.parent.name}",
        )
    if dataset_format == "casia-v3-twins":
        image_paths = sorted(path for path in dataset_dir.glob("*/*/*.jpg") if path.name.lower() != "thumbs.db")
        return _build_index(
            dataset_dir,
            image_paths,
            lambda image_path: f"{image_path.parent.parent.name}_{image_path.parent.name}",
        )
    if dataset_format == "iitd":
        image_paths = sorted(
            path
            for path in dataset_dir.glob("[0-9][0-9][0-9]/*.bmp")
            if path.name.lower() not in {"thumbs.db", ".ds_store"}
        )
        def label_builder(image_path):
            match = re.match(r"\d+_([LR])$", image_path.stem, flags=re.IGNORECASE)
            if not match:
                raise ValueError(f"Unexpected IITD filename format: {image_path.name}")
            return f"{image_path.parent.name}_{match.group(1).upper()}"
        return _build_index(dataset_dir, image_paths, label_builder)
    if dataset_format == "mmu":
        image_paths = sorted(
            path for path in dataset_dir.glob("*/*/*.bmp") if path.name.lower() not in {"thumbs.db", ".ds_store"}
        )
        def label_builder(image_path):
            side = image_path.parent.name.lower()
            if side not in {"left", "right"}:
                raise ValueError(f"Unexpected MMU directory structure: {image_path}")
            side_code = "L" if side == "left" else "R"
            return f"{image_path.parent.parent.name}_{side_code}"
        return _build_index(dataset_dir, image_paths, label_builder)
    if dataset_format == "mmu2":
        image_paths = sorted(
            path for path in dataset_dir.glob("*.bmp") if path.name.lower() not in {"thumbs.db", ".ds_store"}
        )
        def label_builder(image_path):
            stem = image_path.stem
            if len(stem) < 6 or not stem.isdigit():
                raise ValueError(f"Unexpected MMU2 filename format: {image_path.name}")
            subject = stem[:-4]
            eye_code = stem[-4:-2]
            if eye_code not in {"01", "02"}:
                raise ValueError(f"Unexpected MMU2 eye code in filename: {image_path.name}")
            side_code = "L" if eye_code == "01" else "R"
            return f"{subject}_{side_code}"
        return _build_index(dataset_dir, image_paths, label_builder)
    raise ValueError(f"Unsupported dataset format: {dataset_format}")
