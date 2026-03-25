# eWOM Fusion

This module fuses the outputs of the existing helpfulness and sentiment
pipelines into a single eWOM score.

It uses late fusion rather than feature-level fusion:

- helpfulness decides whether a review should be trusted as influential
- sentiment provides the direction and strength of the message

This design fits the current repo because the two branches are already trained
independently and already expose prediction probabilities.

## Inputs

The fusion layer consumes:

- `usefulness_probability` from `eWOM.helpfulness.predictor.HelpfulnessPredictor`
- `positive_probability` and `negative_probability` from
  `eWOM.sentiment_analysis.predictor.SentimentPredictor`

## Scoring Rule

The current scorer is implemented in `scorer.py` and uses a soft gate:

```text
gate = sigmoid(sharpness * (p_helpful - center))
signed_score = gate * (p_positive - p_negative)
magnitude_score = gate * abs(p_positive - p_negative)
```

Default configuration:

- `center = 0.5`
- `sharpness = 8.0`

Interpretation:

- if helpfulness is low, the final score is suppressed even when sentiment is strong
- if helpfulness is high, the final score follows the sentiment branch
- strongly negative but useful reviews remain strongly negative
- uncertain sentiment stays near neutral

## Output Fields

`EWOMFusionScorer.score(...)` returns:

- `usefulness_probability`: helpfulness probability after clamping
- `helpfulness_gate`: soft gate value in `[0, 1]`
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
    "fusion": {...},
}
```

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
