"""Feature columns aligned with Didibood property_details."""
from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

from config.settings import settings
from data.neighbourhood_persian import resolve_training_neighbourhood
from domain.divar_attrs import (
    attr_bool,
    attr_int,
    flat_attrs,
    parse_bathrooms,
    parse_floor,
    property_type_from_category,
)
from domain.model_key import ModelKey
from domain.target import (
    RENT_TARGET_COLUMN,
    SALE_TARGET_COLUMN,
    equivalent_deposit_toman,
    normalize_divar_rent_prices,
    price_per_sqm_toman,
)

# CatBoost categorical + numeric feature names (Didibood-first)
CATEGORICAL_FEATURES = ("neighbourhood",)
NUMERIC_FEATURES = (
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
)
FEATURE_COLUMNS = CATEGORICAL_FEATURES + NUMERIC_FEATURES
TARGET_COLUMN = SALE_TARGET_COLUMN


def target_column_for(key: ModelKey | str) -> str:
    purpose = key.purpose if isinstance(key, ModelKey) else str(key)
    if purpose == "rent":
        return RENT_TARGET_COLUMN
    return SALE_TARGET_COLUMN


@dataclass
class PricingFeatures:
    city_slug: str
    property_type: str
    purpose: str
    neighbourhood: str | None
    area: float | None
    rooms: int | None
    year_built: int | None
    building_age: int | None
    floor_number: int | None
    total_floors: int | None
    has_parking: int | None
    has_elevator: int | None
    has_storage: int | None
    has_balcony: int | None
    number_of_bathrooms: int | None
    location_lat: float | None
    location_long: float | None

    def to_model_row(self) -> dict[str, Any]:
        row = asdict(self)
        for key in FEATURE_COLUMNS:
            row.setdefault(key, None)
        row["neighbourhood"] = (self.neighbourhood or "unknown").strip() or "unknown"
        return {k: row[k] for k in FEATURE_COLUMNS}

    @classmethod
    def from_request(cls, payload: dict[str, Any]) -> PricingFeatures:
        year = payload.get("year_built")
        building_age = payload.get("building_age")
        if building_age is None and year is not None:
            building_age = 1404 - int(year) if 1300 <= int(year) <= 1500 else None

        def as_bool_int(value: Any) -> int | None:
            if value is None:
                return None
            return 1 if bool(value) else 0

        return cls(
            city_slug=str(payload.get("city_slug") or payload.get("city") or "").lower(),
            property_type=str(payload.get("property_type") or "apartment"),
            purpose=str(payload.get("purpose") or "sale"),
            neighbourhood=payload.get("neighbourhood") or payload.get("district"),
            area=_to_float(payload.get("area")),
            rooms=_to_int(payload.get("rooms")),
            year_built=_to_int(year),
            building_age=_to_int(building_age),
            floor_number=_to_int(payload.get("floor_number")),
            total_floors=_to_int(payload.get("total_floors")),
            has_parking=as_bool_int(payload.get("has_parking", payload.get("parking"))),
            has_elevator=as_bool_int(payload.get("has_elevator", payload.get("elevator"))),
            has_storage=as_bool_int(payload.get("has_storage", payload.get("warehouse"))),
            has_balcony=as_bool_int(payload.get("has_balcony")),
            number_of_bathrooms=_to_int(payload.get("number_of_bathrooms")),
            location_lat=_to_float(payload.get("location_lat")),
            location_long=_to_float(payload.get("location_long")),
        )


def row_from_divar_join(record: dict[str, Any], *, jalali_year: int = 1404) -> dict[str, Any] | None:
    """Build training row from SQL join dict."""
    category = record.get("category_slug") or ""
    if category.endswith("-rent"):
        return _row_from_divar_rent(record, jalali_year=jalali_year)
    if category.endswith("-sell"):
        return _row_from_divar_sale(record, jalali_year=jalali_year)
    return None


def _row_from_divar_sale(record: dict[str, Any], *, jalali_year: int) -> dict[str, Any] | None:
    if record.get("listing_type") not in (None, "BUY"):
        return None

    category = record.get("category_slug") or ""
    attrs = flat_attrs(record.get("attributes"))
    area = attr_int(attrs, "متراژ")
    if area is None or area <= 0:
        return None

    y, y_source = price_per_sqm_toman(
        price_total=_to_int(record.get("price_total")),
        price_per_unit=_to_int(record.get("price_per_unit")),
        area_sqm=float(area),
    )
    if y is None or y <= 0:
        return None

    return _finalize_row(
        record,
        category=category,
        purpose="sale",
        area=float(area),
        attrs=attrs,
        jalali_year=jalali_year,
        target_column=SALE_TARGET_COLUMN,
        target_value=y,
        y_source=y_source,
        extra={
            "price_total_toman": _to_int(record.get("price_total")),
        },
    )


def _row_from_divar_rent(record: dict[str, Any], *, jalali_year: int) -> dict[str, Any] | None:
    if record.get("listing_type") not in (None, "RENT"):
        return None

    category = record.get("category_slug") or ""
    if not category.endswith("-rent"):
        return None

    attrs = flat_attrs(record.get("attributes"))
    area = attr_int(attrs, "متراژ")
    if area is None or area <= 0:
        return None

    deposit = _to_int(record.get("price_total")) or 0
    monthly = _to_int(record.get("price_per_unit")) or 0
    deposit, monthly, remap_note = normalize_divar_rent_prices(
        deposit,
        monthly,
        mismapped_deposit_max_toman=settings.RENT_MISMAPPED_DEPOSIT_MAX_TOMAN,
    )
    y, y_source = equivalent_deposit_toman(
        deposit,
        monthly,
        ratio_deposit_toman=settings.RENT_RATIO_DEPOSIT_TOMAN,
        ratio_monthly_toman=settings.RENT_RATIO_MONTHLY_TOMAN,
    )
    if y is None or y <= 0:
        return None
    if remap_note:
        y_source = remap_note

    return _finalize_row(
        record,
        category=category,
        purpose="rent",
        area=float(area),
        attrs=attrs,
        jalali_year=jalali_year,
        target_column=RENT_TARGET_COLUMN,
        target_value=y,
        y_source=y_source,
        extra={
            "deposit_toman": deposit,
            "monthly_rent_toman": monthly,
        },
    )


def _finalize_row(
    record: dict[str, Any],
    *,
    category: str,
    purpose: str,
    area: float,
    attrs: dict[str, str],
    jalali_year: int,
    target_column: str,
    target_value: float,
    y_source: str | None,
    extra: dict[str, Any],
) -> dict[str, Any]:
    year_built = attr_int(attrs, "ساخت")
    floor_number, total_floors = parse_floor(attrs)

    features = PricingFeatures(
        city_slug=str(record.get("city_slug") or ""),
        property_type=property_type_from_category(category),
        purpose=purpose,
        neighbourhood=resolve_training_neighbourhood(
            city_slug=str(record.get("city_slug") or ""),
            district=record.get("district"),
            location=record.get("location"),
        ),
        area=area,
        rooms=attr_int(attrs, "اتاق"),
        year_built=year_built,
        building_age=(jalali_year - year_built) if year_built and year_built >= 1300 else None,
        floor_number=floor_number,
        total_floors=total_floors,
        has_parking=_bool_to_int(attr_bool(attrs, "پارکینگ")),
        has_elevator=_bool_to_int(attr_bool(attrs, "آسانسور")),
        has_storage=_bool_to_int(attr_bool(attrs, "انباری")),
        has_balcony=_bool_to_int(attr_bool(attrs, "بالکن")),
        number_of_bathrooms=parse_bathrooms(attrs),
        location_lat=_to_float(record.get("geo_lat")),
        location_long=_to_float(record.get("geo_lon")),
    )

    out = features.to_model_row()
    out[target_column] = target_value
    out["y_source"] = y_source
    out["post_token"] = record.get("post_token")
    out.update(extra)
    return out


def _bool_to_int(value: bool | None) -> int | None:
    if value is None:
        return None
    return 1 if value else 0


def _to_int(value: Any) -> int | None:
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _to_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None
