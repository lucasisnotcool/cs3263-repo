# Setup

```bash
cd /Users/fuijingmin/Project/cs3263-repo
source venv/bin/activate
pip install -r requirements.txt
```

Inference commands are in `eWOM/inference.md`.

---

# Helpfulness

## 1. Split Dataset

Balancing is enabled by default. `--balanced-total-rows 8000000` requests a fixed 4,000,000 helpful / 4,000,000 not-helpful split. Omit it to use the largest balanced subset available from the scanned rows.

```bash
python -m eWOM.helpfulness.train_test_splitter \
  --review-path data/raw/amazon-reviews-2023/Electronics.jsonl \
  --output-dir data/helpfulness \
  --max-rows 20000000 \
  --val-size 0.1 \
  --test-size 0.1 \
  --positive-threshold 1 \
  --balanced-total-rows 8000000 \
  --no-drop-middle \
  --min-review-words 0 \
  --overwrite-output
```

## 2. Train All Helpfulness Candidates

```bash
python -m eWOM.helpfulness.pipeline \
  --train-path data/helpfulness/train.jsonl \
  --val-path data/helpfulness/val.jsonl \
  --test-path data/helpfulness/test.jsonl \
  --model-output models/helpfulness/amazon_helpfulness \
  --candidate-models logistic_regression multinomial_nb complement_nb \
  --max-features 50000 \
  --min-df 5 \
  --max-df 0.95 \
  --ngram-max 2 \
  --random-state 42 \
  --log-level INFO
```

By default, the terminal prints a compact table with candidate metrics and selected model metrics. The full JSON summary is still written to `models/helpfulness/amazon_helpfulness_summary.json`. Use `--output-format json` only if you need JSON printed to stdout. Use `--log-level WARNING` if you want less progress logging and mostly the final table.

The default helpfulness feature set is TF-IDF text plus derived `review_len_words`, `title_len_chars`, and `text_len_chars`. It excludes `rating` and `verified_purchase`; use `--no-text-derived-lengths-only` only for legacy metadata-inclusive experiments.

The training command relabels prepared rows with `helpful_vote`/`helpful_votes >= 1` when vote counts are present. It trains Logistic Regression, Multinomial Naive Bayes, and Complement Naive Bayes; selects the best model on validation `macro_f1`; saves the selected model at the main prefix; and saves every candidate at a suffixed prefix.

Main selected checkpoint:

- `models/helpfulness/amazon_helpfulness.joblib`
- `models/helpfulness/amazon_helpfulness_feature_builder.joblib`

Per-candidate checkpoints:

- `models/helpfulness/amazon_helpfulness_logistic_regression.joblib`
- `models/helpfulness/amazon_helpfulness_logistic_regression_feature_builder.joblib`
- `models/helpfulness/amazon_helpfulness_multinomial_nb.joblib`
- `models/helpfulness/amazon_helpfulness_multinomial_nb_feature_builder.joblib`
- `models/helpfulness/amazon_helpfulness_complement_nb.joblib`
- `models/helpfulness/amazon_helpfulness_complement_nb_feature_builder.joblib`

Metadata and summary:

- `models/helpfulness/amazon_helpfulness_metadata.json`
- `models/helpfulness/amazon_helpfulness_summary.json`

## Helpfulness FAQ

### Can I Train Only One Candidate Model?

Yes. Keep the same command and change `--candidate-models` to one model name:

```bash
python -m eWOM.helpfulness.pipeline \
  --train-path data/helpfulness/train.jsonl \
  --val-path data/helpfulness/val.jsonl \
  --test-path data/helpfulness/test.jsonl \
  --model-output models/helpfulness/amazon_helpfulness_single_lr \
  --candidate-models logistic_regression \
  --max-features 50000 \
  --min-df 5 \
  --max-df 0.95 \
  --ngram-max 2 \
  --random-state 42 \
  --log-level INFO
```

Allowed candidate names are `logistic_regression`, `multinomial_nb`, and `complement_nb`.

### Can I Continue Training?

The current helpfulness pipeline does not incrementally continue training from an old checkpoint. It either retrains from the split files or reuses an existing checkpoint. To skip retraining when artifacts already exist, run:

```bash
python -m eWOM.helpfulness.pipeline \
  --model-output models/helpfulness/amazon_helpfulness \
  --reuse-existing-artifacts
```

To retrain, run the main training command again with the same `--model-output` to replace that checkpoint, or use a new `--model-output` prefix to keep both runs.

### Can I Use Metadata Features Again?

Yes, but only for legacy experiments where inference will also provide `rating` and `verified_purchase`. Add:

```bash
--no-text-derived-lengths-only
```

### Can I Create An Unbalanced Split?

Yes. Balanced splitting is the default; add `--no-balance-labels` to keep the original class distribution:

```bash
python -m eWOM.helpfulness.train_test_splitter \
  --review-path data/raw/amazon-reviews-2023/Electronics.jsonl \
  --output-dir data/helpfulness_unbalanced \
  --max-rows 20000000 \
  --positive-threshold 1 \
  --no-balance-labels \
  --overwrite-output
```

---

# Sentiment Analysis

## Train All Sentiment Candidates

```bash
python -m eWOM.sentiment_analysis.run_sentiment_benchmark \
  --data-dir data/raw/amazon_polarity \
  --model-output models/sentiment/amazon_polarity \
  --val-ratio 0.1 \
  --candidate-models logistic_regression multinomial_nb complement_nb \
  --max-features 50000 \
  --min-df 5 \
  --max-df 0.95 \
  --ngram-max 2 \
  --random-state 42 \
  --log-level INFO
```

By default, the terminal prints a compact table with candidate metrics and selected model metrics. The full JSON summary is still written to `models/sentiment/amazon_polarity_summary.json`. Use `--output-format json` only if you need JSON printed to stdout. Use `--log-level WARNING` if you want less progress logging and mostly the final table.

This command trains Logistic Regression, Multinomial Naive Bayes, and Complement Naive Bayes on the Amazon Polarity `train` split; compares them on the internal validation split; saves the selected model at the main prefix; saves every candidate at a suffixed prefix; and evaluates the selected model on the official `test` split.

Main selected checkpoint:

- `models/sentiment/amazon_polarity.joblib`
- `models/sentiment/amazon_polarity_feature_builder.joblib`

Per-candidate checkpoints:

- `models/sentiment/amazon_polarity_logistic_regression.joblib`
- `models/sentiment/amazon_polarity_logistic_regression_feature_builder.joblib`
- `models/sentiment/amazon_polarity_multinomial_nb.joblib`
- `models/sentiment/amazon_polarity_multinomial_nb_feature_builder.joblib`
- `models/sentiment/amazon_polarity_complement_nb.joblib`
- `models/sentiment/amazon_polarity_complement_nb_feature_builder.joblib`

Metadata and summary:

- `models/sentiment/amazon_polarity_metadata.json`
- `models/sentiment/amazon_polarity_summary.json`

## Sentiment FAQ

### Can I Train Only One Candidate Model?

Yes. Keep the same command and change `--candidate-models` to one model name:

```bash
python -m eWOM.sentiment_analysis.run_sentiment_benchmark \
  --data-dir data/raw/amazon_polarity \
  --model-output models/sentiment/amazon_polarity_single_lr \
  --val-ratio 0.1 \
  --candidate-models logistic_regression \
  --max-features 50000 \
  --min-df 5 \
  --max-df 0.95 \
  --ngram-max 2 \
  --random-state 42 \
  --log-level INFO
```

Allowed candidate names are `logistic_regression`, `multinomial_nb`, and `complement_nb`.

### Can I Continue Training?

The current sentiment benchmark pipeline does not incrementally continue training from an old checkpoint. It either retrains from the dataset or reuses an existing checkpoint. To skip retraining when artifacts already exist, run:

```bash
python -m eWOM.sentiment_analysis.run_sentiment_benchmark \
  --model-output models/sentiment/amazon_polarity \
  --reuse-existing-artifacts
```

To retrain, run the main training command again with the same `--model-output` to replace that checkpoint, or use a new `--model-output` prefix to keep both runs.
