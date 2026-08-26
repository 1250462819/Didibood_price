"""Persian neighbourhood labels → training district slugs (from crawl DB)."""
from __future__ import annotations

import logging
import re
import unicodedata
from functools import lru_cache

import psycopg

from config.settings import settings

logger = logging.getLogger(__name__)

_PERSIAN_CHAR_RE = re.compile(r"[\u0600-\u06FF]")

_LABEL_SQL = """
SELECT
    label,
    district,
    COUNT(*) AS cnt
FROM (
    SELECT
        TRIM(plp.location) AS label,
        TRIM(pdp.district) AS district
    FROM divar_plp plp
    INNER JOIN divar_pdp pdp ON pdp.post_token = plp.post_token
    WHERE plp.city_slug = %(city_slug)s
      AND plp.pdp_fetched = TRUE
      AND plp.category_slug LIKE '%%-sell'
      AND plp.location IS NOT NULL
      AND pdp.district IS NOT NULL
      AND TRIM(plp.location) <> ''
      AND TRIM(pdp.district) <> ''
    UNION ALL
    SELECT
        TRIM(pdp.location) AS label,
        TRIM(pdp.district) AS district
    FROM divar_plp plp
    INNER JOIN divar_pdp pdp ON pdp.post_token = plp.post_token
    WHERE plp.city_slug = %(city_slug)s
      AND plp.pdp_fetched = TRUE
      AND plp.category_slug LIKE '%%-sell'
      AND pdp.location IS NOT NULL
      AND pdp.district IS NOT NULL
      AND TRIM(pdp.location) <> ''
      AND TRIM(pdp.district) <> ''
) AS rows
GROUP BY label, district
"""

_LOOKUP_SQL = """
SELECT
    TRIM(pdp.district) AS district,
    COUNT(*) AS cnt
FROM divar_plp plp
INNER JOIN divar_pdp pdp ON pdp.post_token = plp.post_token
WHERE plp.city_slug = %(city_slug)s
  AND plp.pdp_fetched = TRUE
  AND plp.category_slug LIKE '%%-sell'
  AND pdp.district IS NOT NULL
  AND TRIM(pdp.district) <> ''
  AND (
    plp.location ILIKE %(pattern)s
    OR pdp.location ILIKE %(pattern)s
  )
GROUP BY TRIM(pdp.district)
ORDER BY cnt DESC
LIMIT 1
"""


def normalize_persian_label(value: str) -> str:
    text = unicodedata.normalize("NFKC", value.strip())
    text = text.replace("\u200c", " ").replace("ي", "ی").replace("ك", "ک")
    text = re.sub(r"\s+", " ", text)
    return text.strip()


@lru_cache(maxsize=16)
def label_to_district_map(city_slug: str) -> dict[str, str]:
    """Best-effort Persian label → district slug map for one city."""
    slug = city_slug.strip().lower()
    if not slug:
        return {}
    try:
        with psycopg.connect(settings.DATABASE_URL) as conn:
            with conn.cursor() as cur:
                cur.execute(_LABEL_SQL, {"city_slug": slug})
                rows = cur.fetchall()
    except Exception:
        logger.warning("Failed to load neighbourhood label map for %s", slug, exc_info=True)
        return {}

    best: dict[str, tuple[str, int]] = {}
    for label, district, count in rows:
        if not label or not district:
            continue
        key = normalize_persian_label(str(label))
        district_slug = str(district).strip().lower()
        prev = best.get(key)
        if prev is None or int(count) > prev[1]:
            best[key] = (district_slug, int(count))

    return {label: district for label, (district, _) in best.items()}


def _lookup_district_ilike(city_slug: str, label: str) -> str | None:
    pattern = f"%{normalize_persian_label(label)}%"
    try:
        with psycopg.connect(settings.DATABASE_URL) as conn:
            with conn.cursor() as cur:
                cur.execute(
                    _LOOKUP_SQL,
                    {"city_slug": city_slug.strip().lower(), "pattern": pattern},
                )
                row = cur.fetchone()
    except Exception:
        logger.warning(
            "Failed ILIKE neighbourhood lookup for %s / %s",
            city_slug,
            label,
            exc_info=True,
        )
        return None
    if not row or not row[0]:
        return None
    return str(row[0]).strip().lower()


def resolve_label_to_district(city_slug: str, label: str | None) -> str | None:
    if not label or not label.strip():
        return None
    city = city_slug.strip().lower()
    normalized = normalize_persian_label(label)
    if not normalized:
        return None

    mapping = label_to_district_map(city)
    if mapping:
        if normalized in mapping:
            return mapping[normalized]
        lower = normalized.lower()
        for key, district in mapping.items():
            if key.lower() == lower:
                return district
        contains = [
            district
            for key, district in mapping.items()
            if normalized in key or key in normalized
        ]
        unique = set(contains)
        if len(unique) == 1:
            return next(iter(unique))

    if _PERSIAN_CHAR_RE.search(normalized):
        return _lookup_district_ilike(city, normalized)

    return None
