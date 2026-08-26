"""Parse Divar PDP attributes → Didibood-aligned fields."""
from __future__ import annotations

import re
from typing import Any

from domain.model_key import CATEGORY_TO_PROPERTY_TYPE


def normalize_digits(text: str) -> str:
    trans = str.maketrans("۰۱۲۳۴۵۶۷۸۹", "0123456789")
    return text.translate(trans)


def flat_attrs(attributes: list[dict[str, Any]] | None) -> dict[str, str]:
    out: dict[str, str] = {}
    for item in attributes or []:
        key = str(item.get("key") or "").strip()
        value = item.get("value")
        if key and value is not None:
            out[key] = str(value).strip()
    return out


def attr_int(attrs: dict[str, str], key: str) -> int | None:
    text = attrs.get(key)
    if not text:
        return None
    digits = re.sub(r"[^\d]", "", normalize_digits(text))
    return int(digits) if digits else None


def attr_bool(attrs: dict[str, str], key: str) -> bool | None:
    for k, text in attrs.items():
        if k == key or k.startswith(f"{key} "):
            norm = normalize_digits(text.strip())
            if norm in {"ندارد", "0"} or "ندارد" in norm:
                return False
            if norm in {"دارد", "1"} or "دارد" in norm:
                return True
            count = attr_int({k: text}, k)
            if count is not None:
                return count > 0
    return None


def parse_floor(attrs: dict[str, str]) -> tuple[int | None, int | None]:
    text = attrs.get("طبقه")
    if not text:
        return None, None
    norm = normalize_digits(text.strip())
    if "همکف" in norm:
        return 0, None
    if "زیر" in norm and "همک" in norm:
        return -1, None
    match = re.search(r"(\d+)(?:\s*از\s*(\d+))?", norm)
    if not match:
        return None, None
    floor = int(match.group(1))
    total = int(match.group(2)) if match.group(2) else None
    return floor, total


def parse_bathrooms(attrs: dict[str, str]) -> int | None:
    total = 0
    found = False
    for key in ("سرویس", "سرویس ایرانی", "سرویس فرنگی", "تعداد سرویس"):
        text = attrs.get(key)
        if not text:
            continue
        found = True
        norm = normalize_digits(text.strip())
        if norm in {"ندارد", "0"}:
            continue
        if norm == "دارد":
            total += 1
            continue
        count = attr_int({key: text}, key)
        if count:
            total += count
    if not found:
        return None
    return total if total > 0 else None


def property_type_from_category(category_slug: str) -> str:
    return CATEGORY_TO_PROPERTY_TYPE.get(category_slug, "other")
