from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix, hstack
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import MaxAbsScaler


@dataclass(frozen=True)
class HelpfulnessFeatureConfig:
    max_features: int = 50000
    min_df: int = 5
    max_df: float = 0.95
    ngram_range: tuple[int, int] = (1, 2)


class HelpfulnessFeatureBuilder:
    NUMERIC_FEATURE_NAMES = (
        "rating",
        "verified_purchase",
        "review_len_words",
        "title_len_chars",
        "text_len_chars",
    )

    def __init__(self, config: HelpfulnessFeatureConfig):
        self.config = config
        self.vectorizer = TfidfVectorizer(
            max_features=config.max_features,
            min_df=config.min_df,
            max_df=config.max_df,
            ngram_range=config.ngram_range,
            sublinear_tf=True,
        )
        self.numeric_scaler = MaxAbsScaler()

    def fit_transform(self, df: pd.DataFrame) -> csr_matrix:
        text_matrix = self.vectorizer.fit_transform(self._get_texts(df))
        numeric_matrix = self.numeric_scaler.fit_transform(self._build_numeric_matrix(df))
        return hstack([text_matrix, numeric_matrix], format="csr")

    def transform(self, df: pd.DataFrame) -> csr_matrix:
        text_matrix = self.vectorizer.transform(self._get_texts(df))
        numeric_matrix = self.numeric_scaler.transform(self._build_numeric_matrix(df))
        return hstack([text_matrix, numeric_matrix], format="csr")

    def _get_texts(self, df: pd.DataFrame) -> list[str]:
        if "combined_text" not in df.columns:
            raise ValueError("Expected preprocessed DataFrame with a 'combined_text' column.")
        return df["combined_text"].fillna("").astype(str).tolist()

    def _build_numeric_matrix(self, df: pd.DataFrame) -> csr_matrix:
        missing_columns = [
            column for column in self.NUMERIC_FEATURE_NAMES if column not in df.columns
        ]
        if missing_columns:
            raise ValueError(
                "Expected preprocessed DataFrame with numeric feature columns: "
                + ", ".join(missing_columns)
            )

        numeric_df = df.loc[:, list(self.NUMERIC_FEATURE_NAMES)].copy()
        numeric_df["verified_purchase"] = numeric_df["verified_purchase"].astype(float)
        values = numeric_df.astype(float).to_numpy(copy=False)
        values = np.nan_to_num(values, nan=0.0, posinf=0.0, neginf=0.0)
        return csr_matrix(values)
