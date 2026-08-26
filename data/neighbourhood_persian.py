"""Persian neighbourhood labels for training and inference."""
from __future__ import annotations

import json
import logging
import re
import unicodedata
from functools import lru_cache
from urllib.parse import quote
from urllib.request import urlopen

import psycopg

from config.settings import settings
from domain.neighbourhood import is_persian_neighbourhood, is_slug_like
from domain.model_key import SUPPORTED_CITIES

logger = logging.getLogger(__name__)

_PERSIAN_CHAR_RE = re.compile(r"[\u0600-\u06FF]")
_BANNED_LABEL_PARTS = (
    "آژانس",
    "املاک",
    "دقیق",
    "لحظاتی",
    "ساعاتی",
    "پیش در",
    "پیشِ",
    "http",
    "www",
)
_CITY_FA_NAME = {
    "tehran": "تهران",
    "mashhad": "مشهد",
    "isfahan": "اصفهان",
}
_MAP_API_BASE = "https://api.didibood.ir/api/v1/map/locations/neighborhoods"

_DISTRICT_TO_PERSIAN_SQL = """
SELECT
    district,
    label,
    COUNT(*) AS cnt
FROM (
    SELECT
        LOWER(TRIM(pdp.district)) AS district,
        TRIM(plp.location) AS label
    FROM divar_plp plp
    INNER JOIN divar_pdp pdp ON pdp.post_token = plp.post_token
    WHERE plp.city_slug = %(city_slug)s
      AND plp.pdp_fetched = TRUE
      AND plp.category_slug LIKE '%%-sell'
      AND pdp.district IS NOT NULL
      AND plp.location IS NOT NULL
      AND TRIM(pdp.district) <> ''
      AND TRIM(plp.location) <> ''
    UNION ALL
    SELECT
        LOWER(TRIM(pdp.district)) AS district,
        TRIM(pdp.location) AS label
    FROM divar_plp plp
    INNER JOIN divar_pdp pdp ON pdp.post_token = plp.post_token
    WHERE plp.city_slug = %(city_slug)s
      AND plp.pdp_fetched = TRUE
      AND plp.category_slug LIKE '%%-sell'
      AND pdp.district IS NOT NULL
      AND pdp.location IS NOT NULL
      AND TRIM(pdp.district) <> ''
      AND TRIM(pdp.location) <> ''
) AS rows
GROUP BY district, label
"""


def normalize_persian_neighbourhood(value: str | None) -> str:
    if not value:
        return ""
    text = unicodedata.normalize("NFKC", str(value).strip())
    text = text.replace("\u200c", " ").replace("ي", "ی").replace("ك", "ک")
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def clean_location_label(value: str | None) -> str:
    text = normalize_persian_neighbourhood(value)
    if not text:
        return ""
    for sep in ("،", ","):
        if sep in text:
            text = text.split(sep, 1)[0].strip()
    return normalize_persian_neighbourhood(text)


def _is_neighbourhood_label(label: str) -> bool:
    cleaned = clean_location_label(label)
    if not cleaned or not _PERSIAN_CHAR_RE.search(cleaned):
        return False
    if len(cleaned) < 2 or len(cleaned) > 32:
        return False
    lowered = cleaned.lower()
    if any(part in lowered for part in _BANNED_LABEL_PARTS):
        return False
    if cleaned.startswith("در "):
        return False
    return True


def extract_neighbourhood_label(value: str | None) -> str | None:
    """Pull a neighbourhood title out of noisy Divar location strings."""
    cleaned = clean_location_label(value)
    if not cleaned:
        return None

    match = re.search(r"(?:^|\s)در\s+(.+)$", cleaned)
    if match:
        candidate = clean_location_label(match.group(1))
        if _is_neighbourhood_label(candidate):
            return candidate

    if _is_neighbourhood_label(cleaned):
        return cleaned
    return None


@lru_cache(maxsize=16)
def map_catalog_titles(city_slug: str) -> frozenset[str]:
    """Canonical Persian neighbourhood titles from Didibood Map (Divar catalog)."""
    city = city_slug.strip().lower()
    city_fa = _CITY_FA_NAME.get(city)
    if not city_fa or city not in SUPPORTED_CITIES:
        return frozenset()

    try:
        url = f"{_MAP_API_BASE}?city={quote(city_fa)}"
        with urlopen(url, timeout=30) as response:
            payload = json.load(response)
    except Exception:
        logger.warning("Failed to fetch Map neighbourhood catalog for %s", city, exc_info=True)
        return frozenset()

    titles: set[str] = set()
    for item in payload.get("neighbourhoods") or []:
        title = normalize_persian_neighbourhood(str(item.get("title") or ""))
        if title:
            titles.add(title)
    return frozenset(titles)


def _pick_district_title(
    candidates: dict[str, int],
    *,
    catalog: frozenset[str],
) -> str | None:
    if not candidates:
        return None

    ranked = sorted(
        candidates.items(),
        key=lambda item: (
            item[0] in catalog,
            item[1],
            -len(item[0]),
        ),
        reverse=True,
    )
    return ranked[0][0]


@lru_cache(maxsize=16)
def district_to_persian_map(city_slug: str) -> dict[str, str]:
    """Divar district slug → canonical Persian title."""
    city = city_slug.strip().lower()
    if not city:
        return {}

    catalog = map_catalog_titles(city)
    try:
        with psycopg.connect(settings.DATABASE_URL) as conn:
            with conn.cursor() as cur:
                cur.execute(_DISTRICT_TO_PERSIAN_SQL, {"city_slug": city})
                rows = cur.fetchall()
    except Exception:
        logger.warning("Failed to load district→Persian map for %s", city, exc_info=True)
        return {}

    by_district: dict[str, dict[str, int]] = {}
    for district, label, count in rows:
        if not district or not label:
            continue
        extracted = extract_neighbourhood_label(str(label))
        if not extracted:
            continue
        district_key = str(district).strip().lower()
        title_counts = by_district.setdefault(district_key, {})
        title_counts[extracted] = title_counts.get(extracted, 0) + int(count)

    result: dict[str, str] = {}
    for district, title_counts in by_district.items():
        picked = _pick_district_title(title_counts, catalog=catalog)
        if picked:
            result[district] = picked
    return result


def district_slugs_for_neighbourhood(city_slug: str, neighbourhood: str) -> list[str]:
    """Resolve a Persian neighbourhood title (or slug) to Divar district slug(s)."""
    raw = (neighbourhood or "").strip()
    if not raw:
        return []

    city = city_slug.strip().lower()
    target = normalize_persian_neighbourhood(raw)
    mapping = district_to_persian_map(city)

    slugs: list[str] = []
    for slug, title in mapping.items():
        if normalize_persian_neighbourhood(title) == target:
            slugs.append(slug)

    if slugs:
        return sorted(set(slugs))

    if is_slug_like(raw):
        return [raw.lower()]

    return []


def resolve_training_neighbourhood(
    *,
    city_slug: str,
    district: str | None,
    location: str | None,
) -> str:
    """Map crawl district/location to the Persian title used by the pricing model."""
    city = (city_slug or "").strip().lower()
    district_raw = (district or "").strip()
    location_raw = (location or "").strip()
    catalog = map_catalog_titles(city)

    if district_raw:
        if is_persian_neighbourhood(district_raw):
            cleaned = clean_location_label(district_raw)
            if cleaned in catalog or _is_neighbourhood_label(cleaned):
                return cleaned
        mapped = district_to_persian_map(city).get(district_raw.lower())
        if mapped:
            return mapped

    extracted = extract_neighbourhood_label(location_raw)
    if extracted and (extracted in catalog or _is_neighbourhood_label(extracted)):
        return extracted

    return "unknown"


def resolve_request_neighbourhood(
    raw: str | None,
    known_titles: set[str],
) -> str | None:
    """Normalize user/API neighbourhood input to a known Persian training title."""
    if raw is None:
        return None
    candidate = clean_location_label(raw)
    if not candidate:
        return None
    if not known_titles:
        return candidate

    if candidate in known_titles:
        return candidate

    normalized = normalize_persian_neighbourhood(candidate)
    for title in known_titles:
        if normalize_persian_neighbourhood(title) == normalized:
            return title

    return candidate
