"""Target variables — sale (price/sqm) and rent (equivalent deposit)."""
from __future__ import annotations

from typing import Any

SALE_TARGET_COLUMN = "price_per_sqm_toman"
RENT_TARGET_COLUMN = "equivalent_deposit_toman"


def price_per_sqm_toman(
    *,
    price_total: int | None,
    price_per_unit: int | None,
    area_sqm: float | None,
) -> tuple[float | None, str | None]:
    if price_per_unit is not None and price_per_unit > 0:
        return float(price_per_unit), "price_per_unit"
    if (
        price_total is not None
        and price_total > 0
        and area_sqm is not None
        and area_sqm > 0
    ):
        return price_total / area_sqm, "computed"
    return None, None


def equivalent_deposit_toman(
    deposit_toman: int | None,
    monthly_rent_toman: int | None,
    *,
    ratio_deposit_toman: float,
    ratio_monthly_toman: float,
) -> tuple[float | None, str | None]:
    """Didibood-aligned: deposit + monthly × (ratio_deposit / ratio_monthly)."""
    deposit = deposit_toman or 0
    monthly = monthly_rent_toman or 0
    if deposit <= 0 and monthly <= 0:
        return None, None
    if ratio_deposit_toman <= 0 or ratio_monthly_toman <= 0:
        return None, None

    equivalent = deposit + monthly * (ratio_deposit_toman / ratio_monthly_toman)
    if equivalent <= 0:
        return None, None

    if deposit > 0 and monthly > 0:
        source = "deposit_and_monthly"
    elif deposit > 0:
        source = "deposit_only"
    else:
        source = "monthly_only"
    return float(equivalent), source


def normalize_divar_rent_prices(
    deposit_toman: int | None,
    monthly_rent_toman: int | None,
    *,
    mismapped_deposit_max_toman: float,
) -> tuple[int, int, str | None]:
    """Fix listings where monthly rent was stored in price_total (deposit field)."""
    deposit = deposit_toman or 0
    monthly = monthly_rent_toman or 0
    if (
        deposit > 0
        and monthly == 0
        and deposit <= mismapped_deposit_max_toman
    ):
        return 0, deposit, "remapped_deposit_as_monthly"
    return deposit, monthly, None


def rent_conversion_multiplier(
    *,
    ratio_deposit_toman: float,
    ratio_monthly_toman: float,
) -> float:
    return ratio_deposit_toman / ratio_monthly_toman


def split_rent_on_conversion_line(
    equivalent_deposit: float,
    *,
    ratio_deposit_toman: float,
    ratio_monthly_toman: float,
    preferred_deposit_toman: int | None = None,
    preferred_monthly_rent_toman: int | None = None,
) -> tuple[int, int]:
    """Map equivalent deposit to (deposit, monthly) on the conversion line."""
    multiplier = rent_conversion_multiplier(
        ratio_deposit_toman=ratio_deposit_toman,
        ratio_monthly_toman=ratio_monthly_toman,
    )
    if preferred_deposit_toman is not None:
        deposit = max(0, preferred_deposit_toman)
        monthly = max(0.0, (equivalent_deposit - deposit) / multiplier)
        return int(round(deposit)), int(round(monthly))
    if preferred_monthly_rent_toman is not None:
        monthly = max(0, preferred_monthly_rent_toman)
        deposit = max(0.0, equivalent_deposit - monthly * multiplier)
        return int(round(deposit)), int(round(monthly))
    return int(round(equivalent_deposit)), 0


def total_price_toman(price_per_sqm: float, area_sqm: float) -> int:
    return int(round(price_per_sqm * area_sqm))


def prediction_bounds_sale(
    predicted_pps: float,
    area_sqm: float,
    *,
    mape: float | None,
) -> dict[str, Any]:
    """±MAPE band for sale (per sqm + derived total)."""
    margin = float(mape) if mape and mape > 0 else 0.0
    low_pps = max(0.0, predicted_pps * (1 - margin))
    high_pps = predicted_pps * (1 + margin)
    low_pps_int = int(round(low_pps))
    high_pps_int = int(round(high_pps))
    out: dict[str, Any] = {
        "low_price_per_sqm_toman": low_pps_int,
        "high_price_per_sqm_toman": high_pps_int,
        "margin_method": "test_mape",
        "margin_value": margin,
    }
    if area_sqm > 0:
        out["low_total_price_toman"] = total_price_toman(low_pps_int, area_sqm)
        out["high_total_price_toman"] = total_price_toman(high_pps_int, area_sqm)
    else:
        out["low_total_price_toman"] = None
        out["high_total_price_toman"] = None
    return out


def prediction_bounds_rent(
    predicted_equivalent: float,
    *,
    mape: float | None,
) -> dict[str, Any]:
    """±MAPE band for rent equivalent deposit (total)."""
    margin = float(mape) if mape and mape > 0 else 0.0
    low = max(0.0, predicted_equivalent * (1 - margin))
    high = predicted_equivalent * (1 + margin)
    return {
        "low_equivalent_deposit_toman": int(round(low)),
        "high_equivalent_deposit_toman": int(round(high)),
        "margin_method": "test_mape",
        "margin_value": margin,
    }


# Backward-compatible alias
prediction_bounds = prediction_bounds_sale
