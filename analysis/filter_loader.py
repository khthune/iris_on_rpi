from __future__ import annotations

import importlib.util
from pathlib import Path
from types import ModuleType
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
ANALYSIS_ROOT = Path(__file__).resolve().parent
DEFAULT_FILTERS_PATH = PROJECT_ROOT / "filters.py"


def resolve_filters_path(path: str | Path | None) -> Path:
    if path is None:
        return DEFAULT_FILTERS_PATH

    candidate = Path(path).expanduser()
    if candidate.is_absolute():
        return candidate

    for base in (Path.cwd(), ANALYSIS_ROOT, PROJECT_ROOT):
        resolved = (base / candidate).resolve()
        if resolved.exists():
            return resolved
    return (Path.cwd() / candidate).resolve()


def load_module_from_path(path: Path) -> ModuleType:
    spec = importlib.util.spec_from_file_location(f"_iris_filters_{path.stem}", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load filters file: {path}")

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def load_filter_bank(path: str | Path | None = None) -> tuple[list[dict[str, Any]], str]:
    filters_path = resolve_filters_path(path)
    if not filters_path.exists():
        raise FileNotFoundError(f"Filters file does not exist: {filters_path}")

    module = load_module_from_path(filters_path)
    selected_filters = getattr(module, "filters", None)
    if not isinstance(selected_filters, list):
        raise ValueError(f"Filters file must define a list named 'filters': {filters_path}")

    return selected_filters, str(filters_path)
