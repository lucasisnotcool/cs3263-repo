from __future__ import annotations

import re
from typing import Any, Mapping

from .listing_kind import LISTING_KIND_DEVICE, infer_listing_kind_from_row, normalize_listing_kind


_NON_WORD_RE = re.compile(r"[^a-z0-9]+")

_ACCESSORY_CATEGORY_LEAVES = {
    "accessories",
    "bags cases sleeves",
    "batteries",
    "battery chargers",
    "cables",
    "cables interconnects",
    "camera photo accessories",
    "cases",
    "chargers",
    "charging stations",
    "cleaning repair",
    "cleaning kits",
    "docks",
    "earbud headphones accessories",
    "earpads",
    "headphone cases",
    "lens accessories",
    "mounts",
    "screen protectors",
    "stands",
    "straps",
    "tripods monopods",
    "viewfinders",
}

_ACCESSORY_TITLE_PHRASES = {
    "adapter",
    "battery charger",
    "bumper",
    "cable",
    "carrying case",
    "case cover",
    "case for",
    "charger",
    "cleaning kit",
    "cover for",
    "dock",
    "ear pads",
    "ear tips",
    "earbuds tips",
    "earphone tips",
    "hdmi cable",
    "keychain",
    "lanyard",
    "lens cap",
    "mount",
    "protector",
    "replacement",
    "screen protector",
    "sleeve",
    "stand",
    "strap",
    "tripod",
    "usb cable",
    "viewfinder",
}

_DEVICE_TITLE_ALLOW_PHRASES = {
    "airpod",
    "airpods",
    "camera",
    "camcorder",
    "desktop",
    "dslr camera",
    "earbud",
    "earbuds",
    "earphone",
    "earphones",
    "headphone",
    "headphones",
    "headset",
    "ipad",
    "iphone",
    "laptop",
    "macbook",
    "microphone",
    "monitor",
    "notebook",
    "phone",
    "router",
    "smartphone",
    "smartwatch",
    "speaker",
    "tablet",
    "television",
    "tv",
    "watch",
    "webcam",
}


def is_actual_electronics_device(row: Mapping[str, Any]) -> bool:
    """Return True for likely primary electronics, excluding obvious accessories."""

    listing_kind = normalize_listing_kind(row.get("listing_kind"))
    if listing_kind != LISTING_KIND_DEVICE:
        inferred_kind = normalize_listing_kind(infer_listing_kind_from_row(row))
        if inferred_kind != LISTING_KIND_DEVICE:
            return False

    title = _normalize_text(row.get("title"))
    categories = [_normalize_text(value) for value in _to_text_list(row.get("categories"))]
    leaf_category = categories[-1] if categories else ""

    has_device_title = _contains_any(title, _DEVICE_TITLE_ALLOW_PHRASES)
    if leaf_category in _ACCESSORY_CATEGORY_LEAVES and not has_device_title:
        return False

    if _contains_any(title, _ACCESSORY_TITLE_PHRASES) and not _is_device_bundle_title(title):
        return False

    if "accessories" in leaf_category and not has_device_title:
        return False

    return True


def _is_device_bundle_title(title: str) -> bool:
    if not _contains_any(title, _DEVICE_TITLE_ALLOW_PHRASES):
        return False
    return any(
        phrase in title
        for phrase in (
            "with charging case",
            "with case",
            "with charger",
            "with adapter",
        )
    )


def _contains_any(text: str, phrases: set[str]) -> bool:
    return any(_normalize_text(phrase) in text for phrase in phrases)


def _normalize_text(value: Any) -> str:
    normalized = _NON_WORD_RE.sub(" ", str(value or "").lower()).strip()
    return re.sub(r"\s+", " ", normalized)


def _to_text_list(value: Any) -> list[str]:
    if not isinstance(value, list):
        return []
    return [str(item or "") for item in value if str(item or "").strip()]
