"""Extract training rows from crawl DB."""
from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd
import psycopg

from config.settings import settings
from domain.features import row_from_divar_join
from domain.model_key import ModelKey, category_slug_for

EXTRACT_SQL = """
SELECT
    plp.post_token,
    plp.category_slug,
    plp.listing_type,
    plp.property_type,
    plp.city_slug,
    plp.city_name,
    plp.location,
    plp.geo_lat,
    plp.geo_lon,
    plp.price_total,
    plp.price_per_unit,
    pdp.district,
    pdp.attributes
FROM divar_plp plp
INNER JOIN divar_pdp pdp ON pdp.post_token = plp.post_token
WHERE plp.pdp_fetched = TRUE
  AND plp.city_slug = %(city_slug)s
  AND plp.category_slug = %(category_slug)s
"""

# Market history is a different question from comparables: it asks when a listing
# entered the market (first_seen_at), over a window as long as the caller asked
# for. Reusing the 30-day comparables window and bucketing by last_seen_at
# measured crawler activity, not the market, and could never span more than two
# calendar months.
EXTRACT_MARKET_HISTORY_SQL = """
SELECT
    plp.post_token,
    plp.category_slug,
    plp.listing_type,
    plp.property_type,
    plp.city_slug,
    plp.city_name,
    plp.location,
    plp.geo_lat,
    plp.geo_lon,
    plp.price_total,
    plp.price_per_unit,
    pdp.district,
    pdp.attributes,
    plp.first_seen_at
FROM divar_plp plp
INNER JOIN divar_pdp pdp ON pdp.post_token = plp.post_token
WHERE plp.pdp_fetched = TRUE
  AND plp.city_slug = %(city_slug)s
  AND plp.category_slug = %(category_slug)s
  AND plp.first_seen_at >= %(since)s
"""

EXTRACT_COMPARABLES_SQL = """
SELECT
    plp.post_token,
    plp.category_slug,
    plp.listing_type,
    plp.property_type,
    plp.city_slug,
    plp.city_name,
    plp.location,
    plp.geo_lat,
    plp.geo_lon,
    plp.price_total,
    plp.price_per_unit,
    pdp.district,
    pdp.attributes,
    plp.last_seen_at
FROM divar_plp plp
INNER JOIN divar_pdp pdp ON pdp.post_token = plp.post_token
WHERE plp.pdp_fetched = TRUE
  AND plp.city_slug = %(city_slug)s
  AND plp.category_slug = %(category_slug)s
  AND plp.last_seen_at >= (NOW() AT TIME ZONE 'Asia/Tehran') - (%(recent_days)s || ' days')::interval
"""


def _rows_to_dataframe(rows: list[dict[str, Any]]) -> pd.DataFrame:
    parsed: list[dict[str, Any]] = []
    for row in rows:
        record = dict(row)
        if isinstance(record.get("attributes"), str):
            record["attributes"] = json.loads(record["attributes"])
        built = row_from_divar_join(record)
        if built:
            for stamp in ("last_seen_at", "first_seen_at"):
                if record.get(stamp) is not None:
                    built[stamp] = record[stamp]
            parsed.append(built)
    return pd.DataFrame(parsed)


def extract_dataframe(key: ModelKey) -> pd.DataFrame:
    category_slug = category_slug_for(key)
    with psycopg.connect(settings.DATABASE_URL) as conn:
        with conn.cursor(row_factory=psycopg.rows.dict_row) as cur:
            cur.execute(
                EXTRACT_SQL,
                {"city_slug": key.city_slug, "category_slug": category_slug},
            )
            rows = cur.fetchall()

    return _rows_to_dataframe(rows)


def extract_comparables_dataframe(
    key: ModelKey,
    *,
    recent_days: int | None = None,
) -> pd.DataFrame:
    """Live crawl rows for comparables — only listings seen recently on Divar PLP."""
    category_slug = category_slug_for(key)
    window_days = recent_days if recent_days is not None else settings.COMPARABLE_RECENT_DAYS
    window_days = max(1, int(window_days))
    with psycopg.connect(settings.DATABASE_URL) as conn:
        with conn.cursor(row_factory=psycopg.rows.dict_row) as cur:
            cur.execute(
                EXTRACT_COMPARABLES_SQL,
                {
                    "city_slug": key.city_slug,
                    "category_slug": category_slug,
                    "recent_days": window_days,
                },
            )
            rows = cur.fetchall()

    return _rows_to_dataframe(rows)


def extract_market_history_dataframe(
    key: ModelKey,
    *,
    since: datetime,
) -> pd.DataFrame:
    """Rows that entered the market on/after `since`, for the monthly trend."""
    category_slug = category_slug_for(key)
    with psycopg.connect(settings.DATABASE_URL) as conn:
        with conn.cursor(row_factory=psycopg.rows.dict_row) as cur:
            cur.execute(
                EXTRACT_MARKET_HISTORY_SQL,
                {
                    "city_slug": key.city_slug,
                    "category_slug": category_slug,
                    "since": since,
                },
            )
            rows = cur.fetchall()

    return _rows_to_dataframe(rows)


def cache_path(key: ModelKey) -> Path:
    return settings.DATA_CACHE_DIR / f"{key.slug()}.parquet"


def load_or_extract(key: ModelKey, *, refresh: bool = False) -> pd.DataFrame:
    path = cache_path(key)
    if path.is_file() and not refresh:
        return pd.read_parquet(path)
    df = extract_dataframe(key)
    if not df.empty:
        df.to_parquet(path, index=False)
    return df
