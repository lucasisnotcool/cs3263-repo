# Fusion Heuristic Report

## Goal

The fusion layer combines three model outputs into one explainable eWOM score:

- helpfulness estimates whether the review should influence the final score
- sentiment estimates the positive or negative direction
- deception discounts reviews that look fake or manipulative

The heuristic is intentionally simple because there is no downstream supervised
target for training a second-stage fusion model yet.

## Review-Level Rule

For one review, the score is:

```text
helpfulness_gate = sigmoid(sharpness * (p_helpful - center))
deception_weight = 1 - p_deception
informative_gate = helpfulness_gate * deception_weight
sentiment_polarity = p_positive - p_negative
signed_ewom_score = informative_gate * sentiment_polarity
ewom_score_0_to_100 = 50 * (signed_ewom_score + 1)
```

Current defaults:

```text
center = 0.5
sharpness = 8.0
```

This keeps strong positive and negative sentiment, but only when the review is
likely helpful and authentic. Low-helpfulness or high-deception reviews move
toward the neutral score of `50`.

## Review-Set Rule

For many reviews, the model first computes an informative-weighted sentiment
average:

```text
informative_review_weight = sum(informative_gate_i)
weighted_sentiment_polarity =
    sum(informative_gate_i * sentiment_polarity_i) / informative_review_weight
```

The final confidence gate uses effective informative support:

```text
review_set_gate = 1 - exp(-informative_review_weight / review_set_gate_scale)
final_signed_ewom_score = review_set_gate * weighted_sentiment_polarity
final_ewom_score_0_to_100 = 50 * (final_signed_ewom_score + 1)
```

Current default:

```text
review_set_gate_scale = 3.0
```

This is better than using raw review count because many weak, unhelpful, or
deceptive reviews should not create high confidence. A review set becomes more
confident only when it contains enough informative reviews.

## Interpretation

- `50` means neutral or insufficient reliable evidence.
- Above `50` means useful authentic evidence is more positive than negative.
- Below `50` means useful authentic evidence is more negative than positive.
- `final_ewom_magnitude_0_to_100` measures strength without caring whether the
  signal is positive or negative.

## Tuning

Tune the heuristic on validation examples before replacing it with a learned
fusion model:

- lower `helpfulness_gate_center` if helpful reviews are being suppressed too much
- raise `helpfulness_gate_center` if weak reviews pass through too often
- lower `helpfulness_gate_sharpness` for smoother weighting
- raise `helpfulness_gate_sharpness` for a more binary helpful/unhelpful gate
- lower `review_set_gate_scale` if fewer informative reviews should create
  confidence
- raise `review_set_gate_scale` if the aggregate score needs more evidence
