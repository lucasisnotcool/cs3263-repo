from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

from scipy.sparse import csr_matrix
from sklearn.feature_extraction.text import TfidfVectorizer


@dataclass
class SentimentFeatureConfig:
    max_features: int = 50000
    min_df: int = 5
    max_df: float = 0.95
    ngram_range: tuple[int, int] = (1, 2)


class SentimentFeatureBuilder:
    def __init__(self, config: SentimentFeatureConfig):
        self.config = config
        self.vectorizer = TfidfVectorizer(
            max_features=config.max_features,
            min_df=config.min_df,
            max_df=config.max_df,
            ngram_range=config.ngram_range,
            sublinear_tf=True,
        )

    def fit_transform(self, texts: Iterable[str]) -> csr_matrix:
        return self.vectorizer.fit_transform(texts)

    def transform(self, texts: Iterable[str]) -> csr_matrix:
        return self.vectorizer.transform(texts)
