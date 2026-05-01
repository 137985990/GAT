from __future__ import annotations

from pathlib import Path
from typing import Any


CODE_ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = CODE_ROOT.parent
ARTIFACTS_ROOT = PROJECT_ROOT / "artifacts"

_LEGACY_EXACT = {
    "Logs": ARTIFACTS_ROOT / "logs",
    "Checkpoints": ARTIFACTS_ROOT / "checkpoints",
    "runs": ARTIFACTS_ROOT / "runs",
    "Data": PROJECT_ROOT / "Data",
    "Data/cache": ARTIFACTS_ROOT / "cache" / "datasets",
}

_LEGACY_PREFIXES = (
    ("Logs/", ARTIFACTS_ROOT / "logs"),
    ("Checkpoints/", ARTIFACTS_ROOT / "checkpoints"),
    ("runs/", ARTIFACTS_ROOT / "runs"),
    ("Data/cache/", ARTIFACTS_ROOT / "cache" / "datasets"),
)

_CONFIG_PATH_KEYS = (
    "data_dir",
    "log_dir",
    "checkpoint_dir",
    "tensorboard_dir",
    "cache_dir",
    "resume_path",
)

_CONFIG_LIST_PATH_KEYS = (
    "data_files",
    "p_data_files",
    "pp_data_files",
)


def code_root() -> Path:
    return CODE_ROOT


def project_root() -> Path:
    return PROJECT_ROOT


def artifacts_root() -> Path:
    return ARTIFACTS_ROOT


def default_log_dir() -> Path:
    return ARTIFACTS_ROOT / "logs"


def default_checkpoint_dir() -> Path:
    return ARTIFACTS_ROOT / "checkpoints"


def default_tensorboard_dir() -> Path:
    return ARTIFACTS_ROOT / "runs"


def default_cache_dir() -> Path:
    return ARTIFACTS_ROOT / "cache" / "datasets"


def default_data_dir() -> Path:
    return PROJECT_ROOT / "Data"


def _normalize_rel(raw: Any) -> str:
    return str(raw).replace("\\", "/").strip()


def map_legacy_relative_path(raw: Any) -> Path | None:
    normalized = _normalize_rel(raw)
    if normalized in _LEGACY_EXACT:
        return _LEGACY_EXACT[normalized]
    for prefix, root in _LEGACY_PREFIXES:
        if normalized.startswith(prefix):
            suffix = normalized[len(prefix):]
            return (root / suffix) if suffix else root
    return None


def resolve_existing_path(path_like: Any, base_dir: Path | None = None) -> Path:
    candidate = Path(str(path_like))
    if candidate.is_absolute():
        return candidate.resolve()

    search_roots = []
    if base_dir is not None:
        search_roots.append(Path(base_dir))
    search_roots.extend([Path.cwd(), CODE_ROOT, PROJECT_ROOT])
    for root in search_roots:
        probe = (root / candidate).resolve()
        if probe.exists():
            return probe
    if base_dir is not None:
        return (Path(base_dir) / candidate).resolve()
    return (CODE_ROOT / candidate).resolve()


def resolve_project_path(path_like: Any, base_dir: Path | None = None) -> Path:
    candidate = Path(str(path_like))
    if candidate.is_absolute():
        return candidate.resolve()

    mapped = map_legacy_relative_path(path_like)
    if mapped is not None:
        return mapped.resolve()

    if base_dir is not None:
        return (Path(base_dir) / candidate).resolve()
    return (CODE_ROOT / candidate).resolve()


def normalize_config_paths(config: dict, config_path: str | Path | None = None) -> dict:
    normalized = dict(config or {})
    config_dir = resolve_existing_path(config_path).parent if config_path else CODE_ROOT

    for key in _CONFIG_PATH_KEYS:
        if key in normalized and normalized[key]:
            normalized[key] = str(resolve_project_path(normalized[key], base_dir=config_dir))

    for key in _CONFIG_LIST_PATH_KEYS:
        values = normalized.get(key)
        if isinstance(values, list):
            normalized[key] = [
                str(resolve_project_path(value, base_dir=config_dir)) for value in values
            ]

    return normalized
