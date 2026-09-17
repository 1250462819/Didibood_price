"""The listing frame every neighbourhood answer is computed from.

Same source and same parsing as the trained price model: Divar crawl rows turned
into model features by `row_from_divar_join`, then cleaned by the training
cleaners. The page and the model therefore never disagree about what a
neighbourhood costs.

Rebuilding that frame means parsing tens of thousands of attribute blobs, which
is far too slow for a request and too heavy for the production box's memory. So
it is built once into a parquet file and read back from there; a stale file is
still served while a background thread refreshes it, and only a *missing* file
makes a request wait.
"""
from __future__ import annotations

import logging
import threading
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pandas as pd

from config.settings import settings
from data.clean import prepare_training_frame_for_key
from data.extract import extract_analytics_dataframe
from data.neighbourhood_persian import (
    city_persian_name,
    normalize_persian_neighbourhood,
)
from domain.features import target_column_for
from domain.model_key import ModelKey

logger = logging.getLogger(__name__)

#: Columns the analytics endpoints read. Everything else is dropped before the
#: parquet is written — the server has little RAM, and an unused column is paid
#: for on every load.
KEEP_COLUMNS = (
    "neighbourhood",
    "area",
    "rooms",
    "year_built",
    "building_age",
    "floor_number",
    "total_floors",
    "has_parking",
    "has_elevator",
    "has_storage",
    "has_balcony",
    "number_of_bathrooms",
    "location_lat",
    "location_long",
    "post_token",
    "first_seen_at",
    "last_seen_at",
    "price_total_toman",
    "deposit_toman",
    "monthly_rent_toman",
)

_BUILD_LOCK = threading.Lock()
_REFRESHING: set[str] = set()
_MEMO: dict[str, tuple[float, pd.DataFrame]] = {}


@dataclass(frozen=True)
class AnalyticsFrame:
    """A cleaned listing frame plus the provenance the UI shows under the numbers."""

    key: ModelKey
    frame: pd.DataFrame
    target_column: str
    built_at: datetime
    stale: bool

    @property
    def empty(self) -> bool:
        return self.frame.empty


def analytics_cache_path(key: ModelKey) -> Path:
    return settings.DATA_CACHE_DIR / f"analytics__{key.slug()}.parquet"


def build_analytics_frame(key: ModelKey) -> pd.DataFrame:
    """Extract, parse and clean — the expensive path, run off the request thread."""
    raw = extract_analytics_dataframe(key)
    if raw.empty:
        return raw

    cleaned = prepare_training_frame_for_key(raw, key)
    if cleaned.empty:
        return cleaned

    target = target_column_for(key)
    columns = [c for c in (*KEEP_COLUMNS, target) if c in cleaned.columns]
    out = cleaned[columns].copy()

    # Listings with no usable neighbourhood cannot be placed on the map or in the
    # ranking, and "unknown" is what the feature builder writes for them.
    out["neighbourhood"] = out["neighbourhood"].astype(str).str.strip()
    out = out[(out["neighbourhood"] != "") & (out["neighbourhood"].str.lower() != "unknown")]
    # A listing whose district never resolved falls back to the city's own name.
    # That is not a neighbourhood: on the map it has no polygon, and in the
    # ranking it would sit at the top as a bucket holding half the city.
    city_name = city_persian_name(key.city_slug)
    if city_name:
        normalised = out["neighbourhood"].map(normalize_persian_neighbourhood)
        out = out[normalised != normalize_persian_neighbourhood(city_name)]

    for column in out.columns:
        if column in ("neighbourhood", "post_token", "first_seen_at", "last_seen_at"):
            continue
        out[column] = pd.to_numeric(out[column], errors="coerce").astype("float32")

    return out.reset_index(drop=True)


def refresh_analytics_frame(key: ModelKey) -> Path:
    """Rebuild the parquet for one model key. Used by the CLI and the refresher."""
    frame = build_analytics_frame(key)
    path = analytics_cache_path(key)
    path.parent.mkdir(parents=True, exist_ok=True)
    # Write beside the target and move into place: a reader must never see a
    # half-written file, and the refresh runs while requests are being served.
    tmp = path.with_suffix(".parquet.tmp")
    frame.to_parquet(tmp, index=False)
    tmp.replace(path)
    logger.info("analytics frame refreshed key=%s rows=%s", key.slug(), len(frame))
    return path


def _refresh_in_background(key: ModelKey) -> None:
    slug = key.slug()
    with _BUILD_LOCK:
        if slug in _REFRESHING:
            return
        _REFRESHING.add(slug)

    def _run() -> None:
        try:
            refresh_analytics_frame(key)
        except Exception:
            logger.exception("analytics refresh failed key=%s", slug)
        finally:
            with _BUILD_LOCK:
                _REFRESHING.discard(slug)

    threading.Thread(target=_run, name=f"analytics-refresh-{slug}", daemon=True).start()


def _read_cached(path: Path) -> pd.DataFrame:
    """Read the parquet, memoised on its mtime so repeat requests cost nothing."""
    stamp = path.stat().st_mtime
    cached = _MEMO.get(str(path))
    if cached and cached[0] == stamp:
        return cached[1]
    frame = pd.read_parquet(path)
    _MEMO[str(path)] = (stamp, frame)
    return frame


def load_analytics_frame(key: ModelKey) -> AnalyticsFrame:
    """The frame for `key`, rebuilding only when nothing has ever been built."""
    path = analytics_cache_path(key)
    target = target_column_for(key)

    if not path.is_file():
        with _BUILD_LOCK:
            pass
        refresh_analytics_frame(key)

    frame = _read_cached(path)
    built_at = datetime.fromtimestamp(path.stat().st_mtime, tz=timezone.utc)
    age = datetime.now(tz=timezone.utc) - built_at
    stale = age > timedelta(hours=settings.NEIGHBOURHOOD_CACHE_HOURS)
    if stale:
        _refresh_in_background(key)

    return AnalyticsFrame(
        key=key,
        frame=frame,
        target_column=target,
        built_at=built_at,
        stale=stale,
    )
