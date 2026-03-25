from __future__ import annotations

import re
from typing import Iterable

import pandas as pd


class SentimentPreprocessor:
    """Lowercases text and normalizes repeated whitespace."""

    def clean_text(self, text: str) -> str:
        text = str(text).strip().lower()
        return re.sub(r"\s+", " ", text)

    def transform_texts(self, texts: Iterable[str]) -> list[str]:
        return [self.clean_text(text) for text in texts]

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        df["text"] = df["text"].fillna("").map(self.clean_text)
        return df
