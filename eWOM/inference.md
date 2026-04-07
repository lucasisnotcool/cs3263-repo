# Setup

```bash
cd /Users/fuijingmin/Project/cs3263-repo
source venv/bin/activate
```

---

# Helpfulness Model

Current direct helpfulness checkpoint:

- `models/helpfulness/amazon_helpfulness.joblib`
- `models/helpfulness/amazon_helpfulness_feature_builder.joblib`

The helpfulness model expects `title` and `text`. It does not need `rating` or `verified_purchase`; the only non-TF-IDF features are derived from the supplied text:

- `review_len_words`
- `title_len_chars`
- `text_len_chars`

Run direct helpfulness inference:

```bash
python -m eWOM.helpfulness.run_helpfulness_inference \
  --title "Useful and balanced review" \
  --text "Detailed pros and cons, setup notes, and long-term battery observations."
```

Use a different helpfulness checkpoint:

```bash
python -m eWOM.helpfulness.run_helpfulness_inference \
  --model-prefix models/helpfulness/amazon_helpfulness_logistic_regression \
  --title "Useful and balanced review" \
  --text "Detailed pros and cons, setup notes, and long-term battery observations."
```

When the training command uses multiple candidates, the unsuffixed prefix is the selected model and the suffixed prefixes are the individual candidate models.

---

# Full eWOM Fusion

`run_fusion_demo` uses `EWOMModelPaths.defaults()` from `eWOM/api.py` unless model paths are explicitly overridden. The current default artifacts are:

- `models/helpfulness/amazon_helpfulness_logistic_regression.joblib`
- `models/helpfulness/amazon_helpfulness_logistic_regression_feature_builder.joblib`
- `models/sentiment/amazon_polarity_logistic_regression.joblib`
- `models/sentiment/amazon_polarity_logistic_regression_feature_builder.joblib`

These are the LR candidate artifacts. The unsuffixed artifacts, such as `models/helpfulness/amazon_helpfulness.joblib`, are still written by training as the selected-model prefix, but fusion defaults are pinned to LR to avoid ambiguity when multiple candidate files exist.

Run the full helpfulness + sentiment + deception + fusion flow for one review:

```bash
python -m eWOM.run_fusion_demo \
  --title "Useful and balanced review" \
  --text "Detailed pros and cons, setup notes, and long-term battery observations."
```

Run the full flow for multiple reviews and return one final aggregate score:

```bash
python -m eWOM.run_fusion_demo \
  --mock-json mock/mock_ewom.json \
  --mock-case-id listing_feedback_mixed_half_good_half_bad
```

This loads the selected case's `seller_feedback_texts` array, scores each review, and returns:

- `reviews`: per-review helpfulness, sentiment, deception, and fusion outputs
- `aggregate`: one final review-set score, including `final_ewom_score_0_to_100`

Other built-in mock case IDs include:

- `listing_feedback_positive_repeat_phrases`
- `product_page_feedback_positive_short_form`
- `listing_feedback_negative_complaints`
- `listing_feedback_mixed_half_good_half_bad`

Use the full eWOM flow from Python for one review:

```python
from eWOM import score_review

result = score_review(
    {
        "title": "Useful and balanced review",
        "text": "Detailed pros and cons, setup notes, and long-term battery observations.",
    }
)

print(result["helpfulness"])
print(result["sentiment"])
print(result["deception"])
print(result["fusion"])
```

Use the full eWOM flow from Python for multiple reviews:

```python
from eWOM import score_review_set

result = score_review_set(
    [
        "Great communication and the item arrived quickly.",
        "Product matched the description perfectly.",
        "Shipping was delayed and updates were unclear.",
        "Seller ignored my refund request.",
    ]
)

print(result["review_count"])
print(result["aggregate"]["final_ewom_score_0_to_100"])
print(result["aggregate"]["final_ewom_magnitude_0_to_100"])
```

For review-set scoring, `score_review_set` currently accepts review text strings. It does not take per-review `title`, `rating`, or `verified_purchase`; the current default helpfulness model only needs the text-derived fields.

Override model paths from Python:

```python
from eWOM import EWOMModelPaths, score_review_set

result = score_review_set(
    [
        "Detailed buying guide with clear pros and cons.",
        "Very slow shipping and poor communication.",
    ],
    model_paths=EWOMModelPaths(
        helpfulness_model_path="models/helpfulness/amazon_helpfulness_logistic_regression.joblib",
        helpfulness_feature_builder_path="models/helpfulness/amazon_helpfulness_logistic_regression_feature_builder.joblib",
        sentiment_model_path="models/sentiment/amazon_polarity_logistic_regression.joblib",
        sentiment_feature_builder_path="models/sentiment/amazon_polarity_logistic_regression_feature_builder.joblib",
    ),
)
```

---

# Sentiment Model

Current direct sentiment checkpoint:

- `models/sentiment/amazon_polarity.joblib`
- `models/sentiment/amazon_polarity_feature_builder.joblib`

Run direct sentiment inference:

```bash
python -m eWOM.sentiment_analysis.run_sentiment_inference \
  --text "Battery life is excellent and setup was easy."
```

Use a different sentiment checkpoint:

```bash
python -m eWOM.sentiment_analysis.run_sentiment_inference \
  --model-prefix models/sentiment/amazon_polarity_logistic_regression \
  --text "Battery life is excellent and setup was easy."
```

When the benchmark command uses multiple candidates, the unsuffixed prefix is the selected model and the suffixed prefixes are the individual candidate models.

---

# FAQ

## Do I need `rating` or `verified_purchase`?

No for the current default helpfulness model. Those fields are still accepted by the API and CLI for compatibility with older metadata-inclusive checkpoints, but the current default checkpoint ignores them.

## Which model does fusion use by default?

Fusion uses `EWOMModelPaths.defaults()` from `eWOM/api.py`. The current default model paths are:

- `models/helpfulness/amazon_helpfulness_logistic_regression.joblib`
- `models/helpfulness/amazon_helpfulness_logistic_regression_feature_builder.joblib`
- `models/sentiment/amazon_polarity_logistic_regression.joblib`
- `models/sentiment/amazon_polarity_logistic_regression_feature_builder.joblib`

## How do I retrain?

Use `eWOM/training.md`.
