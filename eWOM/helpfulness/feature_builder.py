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
    use_rating: bool = False
    use_verified_purchase: bool = False
    use_text_length_features: bool = True


class HelpfulnessFeatureBuilder:
    EXTERNAL_METADATA_FEATURE_NAMES = (
        "rating",
        "verified_purchase",
    )
    TEXT_LENGTH_FEATURE_NAMES = (
        "review_len_words",
        "title_len_chars",
        "text_len_chars",
    )
    NUMERIC_FEATURE_NAMES = (
        *EXTERNAL_METADATA_FEATURE_NAMES,
        *TEXT_LENGTH_FEATURE_NAMES,
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

    @property
    def active_numeric_feature_names(self) -> tuple[str, ...]:
        names: list[str] = []
        # Older pickled feature builders do not have these flags. Treat missing
        # attributes as the legacy metadata-inclusive configuration.
        if getattr(self.config, "use_rating", True):
            names.append("rating")
        if getattr(self.config, "use_verified_purchase", True):
            names.append("verified_purchase")
        if getattr(self.config, "use_text_length_features", True):
            names.extend(self.TEXT_LENGTH_FEATURE_NAMES)
        return tuple(names)

    def fit_transform(self, df: pd.DataFrame) -> csr_matrix:
        text_matrix = self.vectorizer.fit_transform(self._get_texts(df))
        if not self.active_numeric_feature_names:
            return text_matrix
        numeric_matrix = self.numeric_scaler.fit_transform(self._build_numeric_matrix(df))
        return hstack([text_matrix, numeric_matrix], format="csr")

    def transform(self, df: pd.DataFrame) -> csr_matrix:
        text_matrix = self.vectorizer.transform(self._get_texts(df))
        if not self.active_numeric_feature_names:
            return text_matrix
        numeric_matrix = self.numeric_scaler.transform(self._build_numeric_matrix(df))
        return hstack([text_matrix, numeric_matrix], format="csr")

    def _get_texts(self, df: pd.DataFrame) -> list[str]:
        if "combined_text" not in df.columns:
            raise ValueError("Expected preprocessed DataFrame with a 'combined_text' column.")
        return df["combined_text"].fillna("").astype(str).tolist()

    def _build_numeric_matrix(self, df: pd.DataFrame) -> csr_matrix:
        missing_columns = [
            column for column in self.active_numeric_feature_names if column not in df.columns
        ]
        if missing_columns:
            raise ValueError(
                "Expected preprocessed DataFrame with numeric feature columns: "
                + ", ".join(missing_columns)
            )

        numeric_df = df.loc[:, list(self.active_numeric_feature_names)].copy()
        if "verified_purchase" in numeric_df.columns:
            numeric_df["verified_purchase"] = numeric_df["verified_purchase"].astype(float)
        values = numeric_df.astype(float).to_numpy(copy=False)
        values = np.nan_to_num(values, nan=0.0, posinf=0.0, neginf=0.0)
        return csr_matrix(values)
