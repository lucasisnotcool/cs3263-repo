from __future__ import annotations

import re
from typing import Any, Iterable, Mapping


LISTING_KIND_DEVICE = "device"
LISTING_KIND_CASE = "case"
LISTING_KIND_CABLE = "cable"
LISTING_KIND_CHARGER = "charger"
LISTING_KIND_TIPS = "tips"
LISTING_KIND_CLEANING = "cleaning"
LISTING_KIND_ACCESSORY = "accessory"
LISTING_KIND_OTHER = "other"

LISTING_KIND_VALUES = (
    LISTING_KIND_DEVICE,
    LISTING_KIND_CASE,
    LISTING_KIND_CABLE,
    LISTING_KIND_CHARGER,
    LISTING_KIND_TIPS,
    LISTING_KIND_CLEANING,
    LISTING_KIND_ACCESSORY,
    LISTING_KIND_OTHER,
)

CASE_TERMS = {
    "bumper",
    "case",
    "cover",
    "guard",
    "pouch",
    "protector",
    "shell",
    "skin",
    "sleeve",
}

CABLE_TERMS = {
    "3.5mm",
    "aux",
    "cable",
    "cord",
    "displayport",
    "ethernet",
    "hdmi",
    "lightning",
    "thunderbolt",
    "usb",
    "usb c",
    "usb-c",
}

CHARGER_TERMS = {
    "adapter",
    "charger",
    "charging station",
    "charging stand",
    "dock",
    "dock station",
    "magsafe charger",
    "power adapter",
    "wireless charger",
}

TIPS_TERMS = {
    "earbud tip",
    "earbud tips",
    "ear cushion",
    "ear cushions",
    "ear hook",
    "ear hooks",
    "ear pad",
    "ear pads",
    "ear tip",
    "ear tips",
    "earbud cover",
    "eartip",
    "eartips",
    "earpad",
    "earpads",
    "earplug",
    "earplugs",
}

CLEANING_TERMS = {
    "brush",
    "cleaner",
    "cleaning kit",
    "cleaning pen",
    "dust remover",
}

ACCESSORY_TERMS = {
    "accessories",
    "accessory",
    "anti lost",
    "anti-lost",
    "bundle",
    "chain",
    "clip",
    "decal",
    "decoration",
    "decorations",
    "film",
    "holder",
    "hook",
    "kit",
    "keychain",
    "lanyard",
    "mount",
    "replacement",
    "stand",
    "sticker",
    "strap",
    "tool",
}

DEVICE_TERMS = {
    "airpod",
    "airpods",
    "camera",
    "camcorder",
    "console",
    "dslr",
    "earbud",
    "earbuds",
    "earphone",
    "earphones",
    "gaming headset",
    "headphone",
    "headphones",
    "headset",
    "iphone",
    "ipad",
    "laptop",
    "macbook",
    "microphone",
    "monitor",
    "notebook",
    "phone",
    "pixel",
    "router",
    "smartphone",
    "smartwatch",
    "speaker",
    "tablet",
    "watch",
    "webcam",
}

CATEGORY_HINTS = {
    LISTING_KIND_CASE: {
        "cases",
        "headphone cases",
    },
    LISTING_KIND_CABLE: {
        "cables",
        "cables interconnects",
        "lightning cables",
    },
    LISTING_KIND_CHARGER: {
        "chargers",
        "charging stations",
        "docks",
    },
    LISTING_KIND_TIPS: {
        "earpads",
    },
    LISTING_KIND_DEVICE: {
        "camera photo",
        "cell phones",
        "earbud headphones",
        "headphones earbuds",
        "home audio theater",
        "portable audio accessories",
    },
}

NON_WORD_RE = re.compile(r"[^a-z0-9]+")


def normalize_listing_kind(value: Any) -> str:
    normalized = _normalize_text(value)
    return normalized if normalized in LISTING_KIND_VALUES else LISTING_KIND_OTHER


def infer_listing_kind_from_parts(
    *,
    title: str = "",
    main_category: str = "",
    categories: Iterable[str] | None = None,
    features: Iterable[str] | None = None,
    description: Iterable[str] | None = None,
    details_text: str = "",
) -> str:
    normalized_title = _normalize_text(title)
    category_text = _normalize_text(" ".join(str(item or "") for item in (categories or [])))
    combined_text = _normalize_text(
        " ".join(
            part
            for part in (
                title,
                main_category,
                " ".join(str(item or "") for item in (categories or [])),
                " ".join(str(item or "") for item in (features or [])),
                " ".join(str(item or "") for item in (description or [])),
                details_text,
            )
            if part
        )
    )
    if (
        " for " not in f" {normalized_title} "
        and _text_matches_kind(normalized_title, listing_kind=LISTING_KIND_DEVICE)
        and not any(
        _text_matches_kind(normalized_title, listing_kind=listing_kind)
        for listing_kind in (
            LISTING_KIND_CLEANING,
            LISTING_KIND_TIPS,
            LISTING_KIND_CABLE,
            LISTING_KIND_CHARGER,
            LISTING_KIND_CASE,
            LISTING_KIND_ACCESSORY,
        )
        )
    ):
        return LISTING_KIND_DEVICE

    for listing_kind in (
        LISTING_KIND_CLEANING,
        LISTING_KIND_TIPS,
        LISTING_KIND_CABLE,
        LISTING_KIND_CHARGER,
        LISTING_KIND_CASE,
        LISTING_KIND_ACCESSORY,
    ):
        if _category_matches_kind(category_text, listing_kind) or _text_matches_kind(
            combined_text,
            listing_kind=listing_kind,
        ):
            return listing_kind

    if _category_matches_kind(category_text, LISTING_KIND_DEVICE) or _text_matches_kind(
        combined_text,
        listing_kind=LISTING_KIND_DEVICE,
    ):
        return LISTING_KIND_DEVICE
    return LISTING_KIND_OTHER


def infer_listing_kind_from_row(row: Mapping[str, Any]) -> str:
    return infer_listing_kind_from_parts(
        title=str(row.get("title") or ""),
        main_category=str(row.get("main_category") or ""),
        categories=_to_text_list(row.get("categories")),
        features=_to_text_list(row.get("features")),
        description=_to_text_list(row.get("description")),
        details_text=str(
            row.get("details_text")
            or row.get("details")
            or row.get("product_document")
            or ""
        ),
    )


def _category_matches_kind(category_text: str, listing_kind: str) -> bool:
    return any(hint in category_text for hint in CATEGORY_HINTS.get(listing_kind, set()))


def _text_matches_kind(text: str, *, listing_kind: str) -> bool:
    if not text:
        return False
    if listing_kind == LISTING_KIND_CLEANING:
        return _contains_any_phrase(text, CLEANING_TERMS)
    if listing_kind == LISTING_KIND_TIPS:
        return _contains_any_phrase(text, TIPS_TERMS)
    if listing_kind == LISTING_KIND_CABLE:
        return _contains_any_phrase(text, CABLE_TERMS)
    if listing_kind == LISTING_KIND_CHARGER:
        if "charging case" in text and not _contains_any_phrase(text, CHARGER_TERMS - {"charger"}):
            return False
        return _contains_any_phrase(text, CHARGER_TERMS)
    if listing_kind == LISTING_KIND_CASE:
        return _contains_any_phrase(text, CASE_TERMS) or "charging case" in text
    if listing_kind == LISTING_KIND_ACCESSORY:
        return _contains_any_phrase(text, ACCESSORY_TERMS)
    if listing_kind == LISTING_KIND_DEVICE:
        return _contains_any_phrase(text, DEVICE_TERMS)
    return False


def _contains_any_phrase(text: str, phrases: Iterable[str]) -> bool:
    return any(_normalize_text(phrase) in text for phrase in phrases if phrase)


def _normalize_text(value: Any) -> str:
    normalized = NON_WORD_RE.sub(" ", str(value or "").lower()).strip()
    return re.sub(r"\s+", " ", normalized)


def _to_text_list(value: Any) -> list[str]:
    if not isinstance(value, list):
        return []
    return [str(item or "") for item in value if str(item or "").strip()]
