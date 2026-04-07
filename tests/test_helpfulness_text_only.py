from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import pytest

from eWOM.helpfulness.dataset_loader import PreparedHelpfulnessSplitLoader
from eWOM.helpfulness.feature_builder import (
    HelpfulnessFeatureBuilder,
    HelpfulnessFeatureConfig,
)
from eWOM.helpfulness.pipeline import run_pipeline
from eWOM.helpfulness.predictor import HelpfulnessPredictor
from eWOM.helpfulness.preprocess import HelpfulnessPreprocessor
from eWOM.helpfulness.trainer import (
    build_helpfulness_sample_weights,
    build_vote_stage_labels,
)


class DummyVoteStageModel:
    classes_ = np.array([0, 1, 2], dtype=int)

    def predict_proba(self, x):
        if x.shape[0] != 2:
            raise AssertionError("This dummy model expects exactly two rows.")
        return np.asarray(
            [
                [0.10, 0.20, 0.70],
                [0.50, 0.40, 0.10],
            ],
            dtype=float,
        )


def test_prepared_split_loader_keeps_helpful_votes(tmp_path: Path) -> None:
    split_path = tmp_path / "split.jsonl"
    split_path.write_text(
        json.dumps(
            {
                "title": "Great",
                "text": "Detailed review text",
                "rating": 5,
                "verified_purchase": True,
                "helpful_votes": 7,
                "label": 1,
            }
        )
        + "\n",
        encoding="utf-8",
    )

    loaded = PreparedHelpfulnessSplitLoader(split_path).load()

    assert list(loaded["helpful_votes"]) == [7]


def test_text_only_feature_builder_ignores_title_and_metadata() -> None:
    frame = pd.DataFrame(
        [
            {
                "title": "Short title",
                "text": "Battery life is excellent",
                "rating": 1.0,
                "verified_purchase": False,
                "helpful_votes": 0,
                "label": 0,
            },
            {
                "title": "Completely different title",
                "text": "Battery life is excellent",
                "rating": 5.0,
                "verified_purchase": True,
                "helpful_votes": 10,
                "label": 1,
            },
        ]
    )
    transformed = HelpfulnessPreprocessor().transform(frame)
    builder = HelpfulnessFeatureBuilder(
        HelpfulnessFeatureConfig(
            max_features=100,
            min_df=1,
            max_df=1.0,
            ngram_range=(1, 1),
            text_only=True,
            use_char_ngrams=True,
            char_max_features=50,
            char_max_df=1.0,
        )
    )

    matrix = builder.fit_transform(transformed).toarray()

    assert builder.active_numeric_feature_names == ()
    assert builder.char_vectorizer is not None
    assert matrix.shape[1] == (
        len(builder.vectorizer.vocabulary_) + len(builder.char_vectorizer.vocabulary_)
    )
    assert np.allclose(matrix[0], matrix[1])


def test_vote_stage_labels_respect_positive_threshold() -> None:
    labels = build_vote_stage_labels([0, 1, 2, 3, 8], positive_threshold=3)
    assert labels.tolist() == [0, 1, 1, 2, 2]


def test_vote_stage_sample_weights_use_helpful_vote_magnitude() -> None:
    weights = build_helpfulness_sample_weights(
        labels=[0, 1, 2, 2],
        helpful_votes=[0, 2, 3, 20],
        positive_threshold=3,
        vote_weight_cap=20,
        target_strategy="vote_stages",
    )

    assert weights[0] == 1.0
    assert 0.75 < weights[1] < 1.25
    assert weights[2] > 1.0
    assert weights[3] > weights[2]


def test_predictor_supports_vote_stage_bundle(tmp_path: Path) -> None:
    feature_builder = HelpfulnessFeatureBuilder(
        HelpfulnessFeatureConfig(
            max_features=50,
            min_df=1,
            max_df=1.0,
            ngram_range=(1, 1),
            text_only=True,
            use_char_ngrams=False,
        )
    )
    frame = HelpfulnessPreprocessor().transform(
        pd.DataFrame(
            [
                {"title": "", "text": "first review", "label": 0},
                {"title": "", "text": "second review", "label": 1},
            ]
        )
    )
    feature_builder.fit_transform(frame)

    model_path = tmp_path / "vote_stage_model.joblib"
    feature_builder_path = tmp_path / "vote_stage_feature_builder.joblib"
    joblib.dump(
        {
            "model": DummyVoteStageModel(),
            "model_name": "dummy_vote_stage_model",
            "threshold": 0.6,
            "target_strategy": "vote_stages",
            "weak_signal_weight": 0.5,
        },
        model_path,
    )
    joblib.dump(feature_builder, feature_builder_path)

    predictor = HelpfulnessPredictor(
        model_path=str(model_path),
        feature_builder_path=str(feature_builder_path),
    )
    predictions = predictor.predict_many(
        titles=["", ""],
        texts=["first review", "second review"],
    )

    assert predictions[0]["usefulness_probability"] == pytest.approx(0.8)
    assert predictions[0]["is_useful"] is True
    assert predictions[1]["usefulness_probability"] == pytest.approx(0.3)
    assert predictions[1]["is_useful"] is False


def test_pipeline_writes_vote_stage_metadata(tmp_path: Path) -> None:
    train_path = tmp_path / "train.jsonl"
    val_path = tmp_path / "val.jsonl"
    test_path = tmp_path / "test.jsonl"
    model_output = tmp_path / "helpfulness_model"

    train_rows = [
        {
            "title": "Generic",
            "text": "good",
            "rating": 5,
            "verified_purchase": True,
            "helpful_votes": 0,
            "label": 0,
        },
        {
            "title": "Weak signal",
            "text": "battery decent for travel",
            "rating": 4,
            "verified_purchase": True,
            "helpful_votes": 1,
            "label": 0,
        },
        {
            "title": "Detailed",
            "text": "The battery lasted for nine hours and charging took ninety minutes",
            "rating": 5,
            "verified_purchase": True,
            "helpful_votes": 8,
            "label": 1,
        },
        {
            "title": "Generic",
            "text": "ok product",
            "rating": 3,
            "verified_purchase": False,
            "helpful_votes": 2,
            "label": 0,
        },
        {
            "title": "Detailed",
            "text": "Screen quality is sharp but the speakers distort above eighty percent volume",
            "rating": 4,
            "verified_purchase": True,
            "helpful_votes": 6,
            "label": 1,
        },
        {
            "title": "Detailed",
            "text": "Setup took ten minutes and the manual clearly explained the wifi pairing steps",
            "rating": 4,
            "verified_purchase": False,
            "helpful_votes": 10,
            "label": 1,
        },
    ]
    val_rows = [
        {
            "title": "Generic",
            "text": "nice",
            "rating": 4,
            "verified_purchase": False,
            "helpful_votes": 0,
            "label": 0,
        },
        {
            "title": "Weak signal",
            "text": "screen looks fine",
            "rating": 4,
            "verified_purchase": True,
            "helpful_votes": 1,
            "label": 0,
        },
        {
            "title": "Detailed",
            "text": "Camera quality is good in daylight but struggles in dim rooms",
            "rating": 4,
            "verified_purchase": True,
            "helpful_votes": 4,
            "label": 1,
        },
        {
            "title": "Detailed",
            "text": "The charger became hot after thirty minutes and customer support replaced it quickly",
            "rating": 3,
            "verified_purchase": True,
            "helpful_votes": 5,
            "label": 1,
        },
    ]
    test_rows = [
        {
            "title": "Generic",
            "text": "love it",
            "rating": 5,
            "verified_purchase": True,
            "helpful_votes": 0,
            "label": 0,
        },
        {
            "title": "Weak signal",
            "text": "works well so far",
            "rating": 4,
            "verified_purchase": True,
            "helpful_votes": 2,
            "label": 0,
        },
        {
            "title": "Detailed",
            "text": "Keyboard travel feels precise and the backlight remains even across all keys",
            "rating": 5,
            "verified_purchase": True,
            "helpful_votes": 7,
            "label": 1,
        },
        {
            "title": "Detailed",
            "text": "Packaging was secure but the included cable is too short for a desktop setup",
            "rating": 4,
            "verified_purchase": True,
            "helpful_votes": 4,
            "label": 1,
        },
    ]

    for path, rows in [
        (train_path, train_rows),
        (val_path, val_rows),
        (test_path, test_rows),
    ]:
        with path.open("w", encoding="utf-8") as handle:
            for row in rows:
                handle.write(json.dumps(row) + "\n")

    result = run_pipeline(
        train_path=train_path,
        val_path=val_path,
        test_path=test_path,
        model_output=model_output,
        max_features=100,
        char_max_features=40,
        min_df=1,
        max_df=1.0,
        char_max_df=1.0,
        ngram_max=1,
        candidate_model_names=["sgd_logistic"],
        positive_threshold=3,
        vote_weight_cap=12,
        use_vote_weighting=True,
        target_strategy="vote_stages",
        selection_metric="f1_positive",
        weak_signal_weights=[0.0, 0.5, 1.0],
        sgd_epochs=2,
    )

    metadata_path = model_output.with_name(f"{model_output.name}_metadata.json")
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))

    assert result["config"]["text_only"] is True
    assert result["config"]["use_vote_weighting"] is True
    assert result["config"]["target_strategy"] == "vote_stages"
    assert result["config"]["selection_metric"] == "f1_positive"
    assert result["config"]["use_char_ngrams"] is True
    assert result["training_weight_summary"]["enabled"] is True
    assert result["training_weight_summary"]["target_strategy"] == "vote_stages"
    assert metadata["text_source"] == "text"
    assert metadata["numeric_feature_names"] == []
    assert metadata["feature_config"]["text_only"] is True
    assert metadata["feature_config"]["use_char_ngrams"] is True
    assert "weak_signal_weight" in metadata
