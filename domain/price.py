"""Price domain helpers."""
from __future__ import annotations

from decimal import Decimal


def normalize_city(city: str) -> str:
    return city.strip()


def safe_per_sqm(total_irr: Decimal | None, area_sqm: float) -> Decimal | None:
    if total_irr is None or area_sqm <= 0:
        return None
    return (total_irr / Decimal(str(area_sqm))).quantize(Decimal("1"))
