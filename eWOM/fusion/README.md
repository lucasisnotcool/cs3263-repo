# eWOM Fusion

This module fuses the outputs of the helpfulness, sentiment, and deception
pipelines into a single eWOM score.

It uses late fusion rather than feature-level fusion:

- helpfulness decides whether a review should be trusted as influential
- sentiment provides the direction and strength of the message
- deception discounts reviews that look fake or manipulative

This design fits the current repo because the two branches are already trained
independently and already expose prediction probabilities.

It now supports two levels of scoring:

- review-level fusion for a single review
- product-level aggregation for a whole set of review texts

## Inputs

The fusion layer consumes:

- `usefulness_probability` from `eWOM.helpfulness.predictor.HelpfulnessPredictor`
- `positive_probability` and `negative_probability` from
  `eWOM.sentiment_analysis.predictor.SentimentPredictor`
- `deception_probability` from `eWOM.deception.predictor.DeceptionPredictor`

## Scoring Rule

The current scorer is implemented in `scorer.py` and uses a two-stage gate:

```text
helpfulness_gate = sigmoid(sharpness * (p_helpful - center))
deception_weight = 1 - p_deception
informative_gate = helpfulness_gate * deception_weight
signed_score = informative_gate * (p_positive - p_negative)
magnitude_score = informative_gate * abs(p_positive - p_negative)
```

Default configuration:

- `center = 0.5`
- `sharpness = 8.0`

Interpretation:

- if helpfulness is low, the final score is suppressed even when sentiment is strong
- if deception is high, the final score is further discounted
- if helpfulness is high and deception is low, the final score follows the sentiment branch
- strongly negative but useful reviews remain strongly negative
- uncertain sentiment stays near neutral

## Output Fields

`EWOMFusionScorer.score(...)` returns:

- `usefulness_probability`: helpfulness probability after clamping
- `helpfulness_gate`: soft gate value in `[0, 1]`
- `deception_probability`: deception probability when available, else `None`
- `deception_weight`: authenticity multiplier in `[0, 1]`
- `informative_gate`: `helpfulness_gate * deception_weight`
- `positive_probability`: normalized positive sentiment probability
- `negative_probability`: normalized negative sentiment probability
- `sentiment_polarity`: `positive_probability - negative_probability`
- `sentiment_strength`: `abs(sentiment_polarity)`
- `signed_ewom_score`: final signed score in `[-1, 1]`
- `magnitude_ewom_score`: final magnitude-only score in `[0, 1]`
- `ewom_score_0_to_100`: signed score scaled to `[0, 100]`, where `50` is neutral
- `ewom_magnitude_0_to_100`: magnitude-only score scaled to `[0, 100]`

Recommended usage:

- use `ewom_score_0_to_100` when you want one number that keeps polarity
- use `ewom_magnitude_0_to_100` when you want to rank impact regardless of polarity

## End-to-End Usage

`predictor.py` wraps both existing predictors and applies the scorer:

```python
from eWOM.fusion import EWOMFusionPredictor

predictor = EWOMFusionPredictor(
    helpfulness_model_path="models/helpfulness/amazon_helpfulness_streaming_all.joblib",
    helpfulness_feature_builder_path="models/helpfulness/amazon_helpfulness_streaming_all_feature_builder.joblib",
    sentiment_model_path="models/sentiment/amazon_polarity_baseline.joblib",
    sentiment_feature_builder_path="models/sentiment/amazon_polarity_baseline_feature_builder.joblib",
)

result = predictor.predict_one(
    title="Useful and reliable",
    text="Battery life is excellent and setup was easy.",
    rating=5.0,
    verified_purchase=True,
)

print(result["fusion"]["ewom_score_0_to_100"])
print(result["fusion"]["ewom_magnitude_0_to_100"])
```

Returned structure:

```python
{
    "helpfulness": {...},
    "sentiment": {...},
    "deception": {...},
    "fusion": {...},
}
```

## Product-Level Aggregation

When you have many reviews for the same product, use the review-set path instead
of averaging raw sentiment alone.

The aggregation stage does two things:

- it weights each review by its review-level `informative_gate`
- it applies a second `review_set_gate` based on review volume

Implemented rule:

```text
informative_review_weight = sum(informative_gate_i)
weighted_sentiment_polarity =
    sum(informative_gate_i * sentiment_polarity_i) / informative_review_weight
review_set_gate = 1 - exp(-review_count / review_set_gate_scale)
final_signed_score = review_set_gate * weighted_sentiment_polarity
```

This is a better fit for whole-product evaluation because:

- unhelpful reviews contribute less
- a larger review set gets a stronger final confidence gate
- the final score becomes more confident when multiple informative reviews agree

Example:

```python
from eWOM.api import score_review_set

result = score_review_set(
    [
        "Battery life is excellent and setup was easy.",
        "Sound quality is good but the case scratches easily.",
        "Very disappointed. The left earbud stopped working after a week.",
    ]
)

print(result["aggregate"]["final_ewom_score_0_to_100"])
```

Returned structure:

```python
{
    "review_count": 3,
    "reviews": [
        {
            "text": "...",
            "helpfulness": {...},
            "sentiment": {...},
            "deception": {...},
            "fusion": {...},
        }
    ],
    "aggregate": {
        "review_count": 3,
        "review_set_gate": ...,
        "final_ewom_score_0_to_100": ...,
    },
}
```

## Mock Demo

The demo runner can now read seller feedback arrays from
`mock/mock_ewom.json`:

```bash
python -m eWOM.run_fusion_demo --mock-case-id listing_feedback_negative_complaints
```

This loads the selected case's `seller_feedback_texts` array and runs the
product-level aggregator.

## Tuning Guidance

The current scorer is an explainable baseline. For better performance, tune it on
validation data:

- lower `center` if the gate is too strict
- raise `center` if too many weak reviews pass through
- raise `sharpness` for a more binary gate
- lower `sharpness` for a smoother gate

If you later collect a true downstream eWOM target, you can replace the fixed
formula with a second-stage model trained on:

- `usefulness_probability`
- `positive_probability`
- `negative_probability`
- `rating`
- `verified_purchase`
- review length features

Until then, the current soft gate is a solid baseline because it is simple,
stable, and easy to explain in a report or demo.
