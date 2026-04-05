# eWOM Agent

This package scores a single review with four linked outputs:

- `helpfulness`: whether the review is likely to be useful
- `sentiment`: whether the review is positive or negative
- `deception`: whether the review looks deceptive, using `experiment_trust_fake_reviews`
- `fusion`: a late-fusion eWOM score that gates sentiment by helpfulness and authenticity

The current public entrypoint is `score_review` in `eWOM/api.py`, which loads the trained helpfulness and sentiment artifacts, calls the fake-review trust deploy pipeline for deception, validates the request payload, and returns a structured JSON-compatible response.

## What Is In Scope

The current agent is a Python package, not an installed console app or web service.

- Package API: `eWOM/api.py`
- Helpfulness model: `eWOM/helpfulness`
- Sentiment model: `eWOM/sentiment_analysis`
- Deception model adapter: `eWOM/deception`
- Fusion layer: `eWOM/fusion`

The default runtime artifact paths are:

- `models/helpfulness/amazon_helpfulness_electronics_tfidf_lr.joblib`
- `models/helpfulness/amazon_helpfulness_electronics_tfidf_lr_feature_builder.joblib`
- `models/sentiment/amazon_polarity_baseline.joblib`
- `models/sentiment/amazon_polarity_baseline_feature_builder.joblib`

## Architecture

### 1. Helpfulness

The helpfulness branch uses TF-IDF text features plus metadata features such as:

- `rating`
- `verified_purchase`
- `review_len_words`
- `title_len_chars`
- `text_len_chars`

The current helpfulness training pipeline compares three classifier candidates:

- `logistic_regression`
- `multinomial_nb`
- `complement_nb`

Model selection is done on the validation split with `macro_f1` as the primary metric, then `average_precision`, `roc_auc`, and `balanced_accuracy` as tiebreakers. The checked-in helpfulness metadata currently shows:

- selected model: `logistic_regression`
- selected threshold: `0.7515016646130513`

References:

- `eWOM/helpfulness/trainer.py`
- `models/helpfulness/amazon_helpfulness_electronics_tfidf_lr_metadata.json`

### 2. Sentiment

The default sentiment runtime model is a logistic-regression classifier over TF-IDF text features. The predictor returns:

- `negative_probability`
- `positive_probability`
- `predicted_label`
- `predicted_label_text`

References:

- `eWOM/sentiment_analysis/predictor.py`
- `models/sentiment/amazon_polarity_baseline.joblib`

### 3. Deception

The deception branch is a thin adapter over the deploy-ready runtime in
`experiment_trust_fake_reviews`.

For each review it sends the review text into that pipeline and returns:

- `deception_probability`
- `authenticity_probability`
- `trust_probability`
- `graph_uncertainty_entropy`
- `overall_confidence`
- `status` and structured `error` details when the trust pipeline is unavailable

This branch does not retrain anything. It just uses:

- `experiment_trust_fake_reviews.run_deployment_pipeline`
- `experiment_trust_fake_reviews.TrustFakeReviewsDeployPipeline`

### 4. Fusion

The fusion layer applies a soft helpfulness gate and then downweights the review
again when the deception branch says the review is likely fake:

```text
helpfulness_gate = sigmoid(sharpness * (p_helpful - center))
deception_weight = 1 - p_deception
informative_gate = helpfulness_gate * deception_weight
signed_score = informative_gate * (p_positive - p_negative)
magnitude_score = informative_gate * abs(p_positive - p_negative)
```

Default configuration:

- `helpfulness_gate_center = 0.5`
- `helpfulness_gate_sharpness = 8.0`

This means strong sentiment is intentionally suppressed when the review looks
unhelpful or deceptive. If the deception stream is unavailable, fusion falls
back to full authenticity weight.

Reference:

- `eWOM/fusion/scorer.py`

## Request And Response Shape

Minimal request:

```json
{
  "text": "Battery life is excellent and setup was easy."
}
```

Optional request fields:

- `title`
- `rating`
- `verified_purchase`

Response shape:

```json
{
  "helpfulness": {
    "usefulness_probability": 0.1408,
    "is_useful": false
  },
  "sentiment": {
    "negative_probability": 0.0049,
    "positive_probability": 0.9951,
    "predicted_label": 1,
    "predicted_label_text": "positive"
  },
  "deception": {
    "status": "ok",
    "source": "experiment_trust_fake_reviews",
    "deception_probability": 0.25,
    "authenticity_probability": 0.75,
    "trust_probability": 0.75,
    "is_deceptive": false,
    "graph_uncertainty_entropy": 0.4,
    "overall_confidence": 0.8,
    "error": null
  },
  "fusion": {
    "usefulness_probability": 0.1408,
    "helpfulness_gate": 0.0535,
    "deception_probability": 0.25,
    "deception_weight": 0.75,
    "informative_gate": 0.0401,
    "positive_probability": 0.9951,
    "negative_probability": 0.0049,
    "sentiment_polarity": 0.9901,
    "sentiment_strength": 0.9901,
    "signed_ewom_score": 0.0397,
    "magnitude_ewom_score": 0.0397,
    "ewom_score_0_to_100": 51.98,
    "ewom_magnitude_0_to_100": 3.97
  }
}
```

The example above was produced by the current default artifacts using:

```bash
python -m eWOM.run_fusion_demo \
  --text "Battery life is excellent and setup was easy." \
  --rating 5 \
  --verified-purchase
```

## Python Usage

```python
from eWOM import score_review

result = score_review(
    {
        "title": "Useful and reliable",
        "text": "Battery life is excellent and setup was easy.",
        "rating": 5.0,
        "verified_purchase": True,
    }
)

print(result["fusion"]["ewom_score_0_to_100"])
```

If you need non-default artifacts:

```python
from eWOM import EWOMModelPaths, score_review

result = score_review(
    {"text": "Detailed buying guide with clear pros and cons."},
    model_paths=EWOMModelPaths(
        helpfulness_model_path="models/helpfulness/amazon_helpfulness_electronics_tfidf_lr.joblib",
        helpfulness_feature_builder_path="models/helpfulness/amazon_helpfulness_electronics_tfidf_lr_feature_builder.joblib",
        sentiment_model_path="models/sentiment/amazon_polarity_baseline.joblib",
        sentiment_feature_builder_path="models/sentiment/amazon_polarity_baseline_feature_builder.joblib",
    ),
)
```

## CLI Available

There are currently no installed `console_scripts` entrypoints. The available CLIs are Python module or script entrypoints.

### 1. Fusion demo CLI

Command:

```bash
python -m eWOM.run_fusion_demo --help
```

Available flags:

- `--title`
- `--text` required
- `--rating`
- `--verified-purchase`
- `--helpfulness-model-path`
- `--helpfulness-feature-builder-path`
- `--sentiment-model-path`
- `--sentiment-feature-builder-path`

Purpose:

- runs one review through the full helpfulness + sentiment + deception + fusion stack
- prints a JSON result to stdout

Implementation:

- `eWOM/run_fusion_demo.py`

### 2. Helpfulness dataset split builder

Command:

```bash
python -m eWOM.helpfulness.train_test_splitter --help
```

Notes:

- this CLI creates `train.jsonl`, `val.jsonl`, `test.jsonl`, and `split_summary.json`
- the split is label-stratified and randomized
- the main knobs are `--review-path`, `--output-dir`, `--val-size`, and `--test-size`

Implementation:

- `eWOM/helpfulness/train_test_splitter.py`

### 3. Helpfulness training pipeline

Command:

```bash
python -m eWOM.helpfulness.pipeline --help
```

Notes:

- this trains and evaluates the helpfulness branch from explicit `train/val/test` files
- it writes model artifacts, metadata, and summary JSON under `models/helpfulness`
- the main knobs are `--train-path`, `--val-path`, `--test-path`, `--model-output`, and `--candidate-models`
- `--reuse-existing-artifacts` skips fitting and returns the stored summary when the checkpoint files already exist

Implementation:

- `eWOM/helpfulness/pipeline.py`

### 4. Helpfulness direct inference CLI

Command:

```bash
python -m eWOM.helpfulness.run_helpfulness_inference --help
```

Notes:

- this loads an existing helpfulness checkpoint and predicts one review directly
- the main knobs are `--text`, `--title`, `--rating`, `--verified-purchase`, and `--model-prefix`

Implementation:

- `eWOM/helpfulness/run_helpfulness_inference.py`

### 5. Sentiment training pipeline

Command:

```bash
python eWOM/sentiment_analysis/run_sentiment_pipeline.py
```

Notes:

- this trains and evaluates the default sentiment baseline
- it does not expose CLI flags yet
- on this repository state it loads the local Arrow dataset under `data/raw/amazon_polarity`

Implementation:

- `eWOM/sentiment_analysis/run_sentiment_pipeline.py`

### 6. Sentiment benchmark CLI

Command:

```bash
python -m eWOM.sentiment_analysis.run_sentiment_benchmark --help
```

Notes:

- this benchmarks TF-IDF sentiment classifiers on a train/val split carved from the Amazon Polarity train split
- it supports `logistic_regression`, `multinomial_nb`, and `complement_nb`
- the main knobs are `--data-dir`, `--model-output`, `--val-ratio`, and `--candidate-models`
- when multiple candidates are provided, it saves the validation-selected model artifact and writes the full comparison into the summary JSON
- `--reuse-existing-artifacts` skips fitting and returns the stored summary when the checkpoint files already exist

Implementation:

- `eWOM/sentiment_analysis/run_sentiment_benchmark.py`

### 7. Sentiment direct inference CLI

Command:

```bash
python -m eWOM.sentiment_analysis.run_sentiment_inference --help
```

Notes:

- this loads an existing sentiment checkpoint and predicts one review directly
- the main knobs are `--text` and `--model-prefix`

Implementation:

- `eWOM/sentiment_analysis/run_sentiment_inference.py`

### 8. Related normalization CLI

Command:

```bash
python scripts/run_normalization.py --help
```

Available flags:

- `--url`
- `--raw`
- `--auth-raw`

Purpose:

- normalizes eBay product URLs into candidate records
- can also print raw eBay API responses for inspection

Environment variables:

- `EBAY_CLIENT_ID`
- `EBAY_CLIENT_SECRET`
- `EBAY_ENVIRONMENT`
- optional `EBAY_OAUTH_SCOPES`

Implementation:

- `scripts/run_normalization.py`

## Data And Artifacts

Prepared helpfulness data currently lives at:

- `data/helpfulness/train.jsonl`
- `data/helpfulness/val.jsonl`
- `data/helpfulness/test.jsonl`

Raw source datasets currently referenced by the codebase:

- `data/raw/amazon-reviews-2023/Electronics.jsonl`
- `data/raw/amazon_polarity`

Current tracked summaries:

- `models/helpfulness/amazon_helpfulness_electronics_tfidf_lr_summary.json`
- `models/sentiment/amazon_polarity_full_benchmark_summary.json`

## Setup

Use the existing virtual environment or install the dependencies from `requirements.txt`.

Example:

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

## Current Limitations

- The package is documented as an "agent", but it is currently a Python inference/training package rather than a deployed service or tool-calling agent.
- The runtime defaults in `eWOM/api.py` still point to the baseline sentiment checkpoint, not the newer benchmark-selected sentiment checkpoint.
- Old checked-in summary files may still refer to `dev` in metadata produced before the explicit `train/val/test` CLI refactor.
- The final eWOM score is an explainable heuristic fusion layer, not a second-stage model trained on downstream business outcomes.
