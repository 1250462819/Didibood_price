"""Versioned models, and which version answers predictions.

Every training used to overwrite `{key}.cbm`, so a retrain that came out worse
destroyed the model it replaced. Now each training is kept as its own version
under `{key}/`, and `{key}/active.json` names the one the predictor uses.
Nothing becomes active by being trained; an admin chooses, and can go back.

Layout::

    MODEL_DIR/tehran__apartment__sale/
        20260919T101500Z.cbm
        20260919T101500Z.meta.json
        active.json            {"version": "20260919T101500Z", ...}

The legacy single-file models are imported as the first version on first use,
and left where they are as a backup.
"""
from __future__ import annotations

import json
import logging
import shutil
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from catboost import CatBoostRegressor

from config.settings import settings
from domain.model_key import ModelKey

logger = logging.getLogger(__name__)

_LOCK = threading.Lock()
_MODEL_CACHE: dict[tuple[str, str], tuple[CatBoostRegressor, dict[str, Any]]] = {}


def new_version_id(now: datetime | None = None) -> str:
    stamp = (now or datetime.now(timezone.utc)).astimezone(timezone.utc)
    return stamp.strftime("%Y%m%dT%H%M%SZ")


def key_dir(key: ModelKey) -> Path:
    return settings.MODEL_DIR / key.slug()


def version_paths(key: ModelKey, version: str) -> tuple[Path, Path]:
    base = key_dir(key) / version
    return base.with_suffix(".cbm"), base.with_suffix(".meta.json")


def _legacy_paths(key: ModelKey) -> tuple[Path, Path]:
    base = settings.MODEL_DIR / key.slug()
    return base.with_suffix(".cbm"), base.with_suffix(".meta.json")


def _import_legacy(key: ModelKey) -> None:
    """Bring a pre-registry model in as version one, active."""
    model_path, meta_path = _legacy_paths(key)
    directory = key_dir(key)
    if directory.is_dir() or not model_path.is_file() or not meta_path.is_file():
        return

    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    try:
        trained = datetime.fromisoformat(meta.get("trained_at", ""))
    except ValueError:
        trained = datetime.fromtimestamp(model_path.stat().st_mtime, tz=timezone.utc)
    version = new_version_id(trained)

    directory.mkdir(parents=True, exist_ok=True)
    target_model, target_meta = version_paths(key, version)
    shutil.copy2(model_path, target_model)
    meta.update(
        {
            "version": version,
            # Trained before the registry: all data at the time, random split.
            "window_days": None,
            "eval_method": meta.get("eval_method", "random"),
            "imported_from_legacy": True,
        }
    )
    target_meta.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    _write_active(key, version, reason="imported legacy model")
    logger.info("model registry: imported legacy %s as %s", key.slug(), version)


def ensure_registry(key: ModelKey) -> None:
    with _LOCK:
        _import_legacy(key)


def _write_active(key: ModelKey, version: str, *, reason: str) -> None:
    payload = {
        "version": version,
        "activated_at": datetime.now(timezone.utc).isoformat(),
        "reason": reason,
    }
    path = key_dir(key) / "active.json"
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    tmp.replace(path)


def active_version(key: ModelKey) -> str | None:
    ensure_registry(key)
    path = key_dir(key) / "active.json"
    if not path.is_file():
        return None
    return json.loads(path.read_text(encoding="utf-8")).get("version")


def active_record(key: ModelKey) -> dict[str, Any] | None:
    ensure_registry(key)
    path = key_dir(key) / "active.json"
    if not path.is_file():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def list_versions(key: ModelKey) -> list[dict[str, Any]]:
    """Every version's metadata, newest first."""
    ensure_registry(key)
    directory = key_dir(key)
    if not directory.is_dir():
        return []
    versions: list[dict[str, Any]] = []
    for meta_path in directory.glob("*.meta.json"):
        try:
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            logger.warning("model registry: unreadable meta %s", meta_path)
            continue
        meta.setdefault("version", meta_path.name.removesuffix(".meta.json"))
        versions.append(meta)
    versions.sort(key=lambda meta: meta["version"], reverse=True)
    return versions


def save_version(key: ModelKey, model: CatBoostRegressor, meta: dict[str, Any]) -> str:
    """Persist a freshly trained model as a new, *inactive* version."""
    ensure_registry(key)
    version = meta.get("version") or new_version_id()
    meta["version"] = version
    key_dir(key).mkdir(parents=True, exist_ok=True)
    model_path, meta_path = version_paths(key, version)
    model.save_model(str(model_path))
    meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    # The very first model of a key has nothing to compete with.
    if active_version(key) is None:
        _write_active(key, version, reason="first model for this key")
    return version


def activate(key: ModelKey, version: str) -> dict[str, Any]:
    model_path, meta_path = version_paths(key, version)
    if not model_path.is_file() or not meta_path.is_file():
        raise FileNotFoundError(f"No version {version} for {key.slug()}")
    with _LOCK:
        _write_active(key, version, reason="activated by admin")
    logger.info("model registry: %s now serves %s", key.slug(), version)
    return active_record(key) or {}


def load_version(key: ModelKey, version: str) -> tuple[CatBoostRegressor, dict[str, Any]]:
    cache_key = (key.slug(), version)
    cached = _MODEL_CACHE.get(cache_key)
    if cached is not None:
        return cached
    model_path, meta_path = version_paths(key, version)
    if not model_path.is_file() or not meta_path.is_file():
        raise FileNotFoundError(f"Model version {version} not found for {key.slug()}")
    model = CatBoostRegressor()
    model.load_model(str(model_path))
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    _MODEL_CACHE[cache_key] = (model, meta)
    return model, meta


def load_active(key: ModelKey) -> tuple[CatBoostRegressor, dict[str, Any]]:
    """The model predictions use. Falls back to the legacy file if the registry
    has nothing, so a service upgraded mid-flight keeps answering."""
    version = active_version(key)
    if version:
        return load_version(key, version)
    model_path, meta_path = _legacy_paths(key)
    if not model_path.is_file() or not meta_path.is_file():
        raise FileNotFoundError(f"Model not found for {key.slug()}")
    model = CatBoostRegressor()
    model.load_model(str(model_path))
    return model, json.loads(meta_path.read_text(encoding="utf-8"))
