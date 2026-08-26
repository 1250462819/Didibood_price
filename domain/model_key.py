"""Model identity: one CatBoost model per city + property_type + purpose."""
from __future__ import annotations

from dataclasses import dataclass

SUPPORTED_CITIES = ("tehran", "mashhad", "isfahan")
SUPPORTED_PURPOSES = ("sale", "rent")
# v1: apartment only; extend later (house_villa, land, …)
SUPPORTED_PROPERTY_TYPES = ("apartment",)

CATEGORY_TO_PROPERTY_TYPE = {
    "apartment-sell": "apartment",
    "apartment-rent": "apartment",
    "house-villa-sell": "house_villa",
    "plot-old": "land",
    "office-sell": "office",
    "shop-sell": "store",
}


@dataclass(frozen=True)
class ModelKey:
    city_slug: str
    property_type: str
    purpose: str = "sale"

    def slug(self) -> str:
        return f"{self.city_slug}__{self.property_type}__{self.purpose}"

    @classmethod
    def from_slug(cls, raw: str) -> ModelKey:
        parts = raw.split("__")
        if len(parts) != 3:
            raise ValueError(f"Invalid model slug: {raw}")
        return cls(city_slug=parts[0], property_type=parts[1], purpose=parts[2])


def category_slug_for(key: ModelKey) -> str:
    if key.property_type == "apartment" and key.purpose == "sale":
        return "apartment-sell"
    if key.property_type == "apartment" and key.purpose == "rent":
        return "apartment-rent"
    raise ValueError(f"No Divar category mapping for {key}")
