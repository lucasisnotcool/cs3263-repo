from __future__ import annotations

import re
from typing import Any

import pandas as pd


_WHITESPACE_RE = re.compile(r"\s+")


def _coerce_float(value: Any, default: float = 0.0) -> float:
    try:
        if value is None or pd.isna(value) or value == "":
            return float(default)
        return float(value)
    except (TypeError, ValueError):
        return float(default)


def _coerce_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if value is None or pd.isna(value):
        return False
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"1", "true", "yes", "y"}:
            return True
        if normalized in {"0", "false", "no", "n", ""}:
            return False
    return bool(value)


class HelpfulnessPreprocessor:
    """Normalizes review fields and derives inference-safe metadata features."""

    FEATURE_COLUMNS = (
        "combined_text",
        "rating",
        "verified_purchase",
        "review_len_words",
        "title_len_chars",
        "text_len_chars",
    )

    def clean_text(self, text: Any) -> str:
        text = "" if text is None or pd.isna(text) else str(text)
        return _WHITESPACE_RE.sub(" ", text).strip()

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        frame = df.copy()

        title = frame["title"] if "title" in frame.columns else pd.Series("", index=frame.index)
        text = frame["text"] if "text" in frame.columns else pd.Series("", index=frame.index)
        rating = (
            frame["rating"] if "rating" in frame.columns else pd.Series(0.0, index=frame.index)
        )
        verified_purchase = (
            frame["verified_purchase"]
            if "verified_purchase" in frame.columns
            else pd.Series(False, index=frame.index)
        )

        title = title.map(self.clean_text)
        text = text.map(self.clean_text)

        transformed = pd.DataFrame(index=frame.index)
        transformed["title"] = title
        transformed["text"] = text
        transformed["combined_text"] = (title + " " + text).str.strip()
        transformed["rating"] = rating.map(_coerce_float).astype(float)
        transformed["verified_purchase"] = verified_purchase.map(_coerce_bool).astype(bool)
        transformed["review_len_words"] = text.map(lambda value: len(value.split())).astype(float)
        transformed["title_len_chars"] = title.str.len().astype(float)
        transformed["text_len_chars"] = text.str.len().astype(float)

        if "label" in frame.columns:
            transformed["label"] = frame["label"].astype(int)

        return transformed
