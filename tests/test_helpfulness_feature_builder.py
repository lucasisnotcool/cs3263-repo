from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

from eWOM.helpfulness.feature_builder import (
    HelpfulnessFeatureBuilder,
    HelpfulnessFeatureConfig,
)


class HelpfulnessFeatureBuilderTest(unittest.TestCase):
    def test_text_derived_lengths_only_ignores_external_metadata(self) -> None:
        frame = pd.DataFrame(
            [
                {
                    "combined_text": "battery lasts all day",
                    "review_len_words": 4.0,
                    "title_len_chars": 7.0,
                    "text_len_chars": 21.0,
                },
                {
                    "combined_text": "battery lasts all day",
                    "review_len_words": 4.0,
                    "title_len_chars": 7.0,
                    "text_len_chars": 21.0,
                },
            ]
        )
        builder = HelpfulnessFeatureBuilder(
            HelpfulnessFeatureConfig(
                max_features=100,
                min_df=1,
                max_df=1.0,
                ngram_range=(1, 1),
                use_rating=False,
                use_verified_purchase=False,
                use_text_length_features=True,
            )
        )

        matrix = builder.fit_transform(frame).toarray()

        self.assertEqual(
            builder.active_numeric_feature_names,
            ("review_len_words", "title_len_chars", "text_len_chars"),
        )
        self.assertEqual(matrix.shape[1], len(builder.vectorizer.vocabulary_) + 3)
        self.assertTrue(np.allclose(matrix[0], matrix[1]))

    def test_default_numeric_feature_selection_preserves_legacy_metadata(self) -> None:
        builder = HelpfulnessFeatureBuilder(
            HelpfulnessFeatureConfig(
                max_features=100,
                min_df=1,
                max_df=1.0,
                ngram_range=(1, 1),
            )
        )

        self.assertEqual(
            builder.active_numeric_feature_names,
            builder.NUMERIC_FEATURE_NAMES,
        )


if __name__ == "__main__":
    unittest.main()
