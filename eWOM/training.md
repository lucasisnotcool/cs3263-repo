# Setup

```bash
cd /Users/fuijingmin/Project/cs3263-repo
source venv/bin/activate
pip install -r requirements.txt
```

---

# Helpfulness

1. Split dataset
```bash
python -m eWOM.helpfulness.train_test_splitter \
  --review-path data/raw/amazon-reviews-2023/Electronics.jsonl \
  --output-dir data/helpfulness \
  --max-rows 10000000 \
  --val-size 0.1 \
  --test-size 0.1 \
  --positive-threshold 1 \
  --no-drop-middle \
  --min-review-words 0 \
  --overwrite-output
```

For a roughly equal positive/negative helpfulness split, enable majority-label undersampling:

```bash
python -m eWOM.helpfulness.train_test_splitter \
  --review-path data/raw/amazon-reviews-2023/Electronics.jsonl \
  --output-dir data/helpfulness_balanced_8m \
  --max-rows 20000000 \
  --val-size 0.1 \
  --test-size 0.1 \
  --positive-threshold 1 \
  --balance-labels \
  --balanced-total-rows 8000000 \
  --no-drop-middle \
  --min-review-words 0 \
  --overwrite-output
```

`--balanced-total-rows 8000000` means 4,000,000 helpful and 4,000,000 not-helpful rows across train, val, and test. If the command reports that it found too few helpful rows, increase `--max-rows` or omit `--balanced-total-rows` to use the largest possible balanced subset from the scanned rows.

2. Train Logistic Regression and save a reusable checkpoint:

```bash
python -m eWOM.helpfulness.pipeline \
  --train-path data/helpfulness/train.jsonl \
  --val-path data/helpfulness/val.jsonl \
  --test-path data/helpfulness/test.jsonl \
  --model-output models/helpfulness/amazon_helpfulness_lr \
  --candidate-models logistic_regression \
  --max-features 50000 \
  --min-df 5 \
  --max-df 0.95 \
  --ngram-max 2 \
  --random-state 42 \
  --log-level INFO
```

The training pipeline also relabels prepared rows with `helpful_vote`/`helpful_votes >= 1` when vote counts are present, so older split files built with a higher threshold do not have to be regenerated before retraining.

Use `--text-derived-lengths-only` when inference inputs will not include external metadata. This keeps TF-IDF features and derived `review_len_words`, `title_len_chars`, and `text_len_chars`, but excludes `rating` and `verified_purchase`.

Checkpoint files written by that command:

- `models/helpfulness/amazon_helpfulness_lr.joblib`
- `models/helpfulness/amazon_helpfulness_lr_feature_builder.joblib`
- `models/helpfulness/amazon_helpfulness_lr_metadata.json`
- `models/helpfulness/amazon_helpfulness_lr_summary.json`

Reuse the saved checkpoint without retraining:

```bash
python -m eWOM.helpfulness.pipeline \
  --model-output models/helpfulness/amazon_helpfulness_lr \
  --reuse-existing-artifacts
```

Run direct helpfulness inference from the saved checkpoint:

```bash
python -m eWOM.helpfulness.run_helpfulness_inference \
  --model-prefix models/helpfulness/amazon_helpfulness_lr \
  --title "Useful and balanced review" \
  --text "Detailed pros and cons, setup notes, and long-term battery observations." \
  --rating 4 \
  --verified-purchase
```

3. Train Multinomial Naive Bayes:
```bash
python -m eWOM.helpfulness.pipeline \
  --train-path data/helpfulness/train.jsonl \
  --val-path data/helpfulness/val.jsonl \
  --test-path data/helpfulness/test.jsonl \
  --model-output models/helpfulness/amazon_helpfulness_multinomial_nb \
  --candidate-models multinomial_nb \
  --max-features 50000 \
  --min-df 5 \
  --max-df 0.95 \
  --ngram-max 2 \
  --random-state 42 \
  --log-level INFO
```

Reuse the saved checkpoint without retraining:

```bash
python -m eWOM.helpfulness.pipeline \
  --model-output models/helpfulness/amazon_helpfulness_multinomial_nb \
  --reuse-existing-artifacts
```

Run direct helpfulness inference:

```bash
python -m eWOM.helpfulness.run_helpfulness_inference \
  --model-prefix models/helpfulness/amazon_helpfulness_multinomial_nb \
  --text "Step-by-step usage notes with clear pros and cons."
```

4. Train Complement Naive Bayes:
```bash
python -m eWOM.helpfulness.pipeline \
  --train-path data/helpfulness/train.jsonl \
  --val-path data/helpfulness/val.jsonl \
  --test-path data/helpfulness/test.jsonl \
  --model-output models/helpfulness/amazon_helpfulness_complement_nb \
  --candidate-models complement_nb \
  --max-features 50000 \
  --min-df 5 \
  --max-df 0.95 \
  --ngram-max 2 \
  --random-state 42 \
  --log-level INFO
```

Train all
```bash
python -m eWOM.helpfulness.pipeline \
  --train-path data/helpfulness/train.jsonl \
  --val-path data/helpfulness/val.jsonl \
  --test-path data/helpfulness/test.jsonl \
  --model-output models/helpfulness/amazon_helpfulness_benchmark \
  --candidate-models logistic_regression multinomial_nb complement_nb \
  --max-features 50000 \
  --min-df 5 \
  --max-df 0.95 \
  --ngram-max 2 \
  --random-state 42 \
  --log-level INFO
```

Reuse the saved checkpoint without retraining:

```bash
python -m eWOM.helpfulness.pipeline \
  --model-output models/helpfulness/amazon_helpfulness_complement_nb \
  --reuse-existing-artifacts
```

Run direct helpfulness inference:

```bash
python -m eWOM.helpfulness.run_helpfulness_inference \
  --model-prefix models/helpfulness/amazon_helpfulness_complement_nb \
  --text "Honest comparison with concrete buying advice."
```

5. Run the helpfulness benchmark across all three:
```bash
python -m eWOM.helpfulness.pipeline \
  --train-path data/helpfulness/train.jsonl \
  --val-path data/helpfulness/val.jsonl \
  --test-path data/helpfulness/test.jsonl \
  --model-output models/helpfulness/amazon_helpfulness_benchmark \
  --candidate-models logistic_regression multinomial_nb complement_nb \
  --max-features 50000 \
  --min-df 5 \
  --max-df 0.95 \
  --ngram-max 2 \
  --random-state 42 \
  --log-level INFO
```

This benchmark command trains the listed candidates on the `train` split, compares them on the `val` split, saves the selected checkpoint under `models/helpfulness/amazon_helpfulness_benchmark*`, and evaluates the selected checkpoint on `test`.

Reuse the selected benchmark checkpoint without retraining:

```bash
python -m eWOM.helpfulness.pipeline \
  --model-output models/helpfulness/amazon_helpfulness_benchmark \
  --reuse-existing-artifacts
```

Run direct helpfulness inference from the selected benchmark checkpoint:

```bash
python -m eWOM.helpfulness.run_helpfulness_inference \
  --model-prefix models/helpfulness/amazon_helpfulness_benchmark \
  --text "Detailed buying guide with concrete strengths, weaknesses, and durability notes."
```

---

# Sentiment Analysis

1. Train sentiment Logistic Regression and save a reusable checkpoint:
```bash
python -m eWOM.sentiment_analysis.run_sentiment_benchmark \
  --data-dir data/raw/amazon_polarity \
  --model-output models/sentiment/amazon_polarity_lr \
  --val-ratio 0.1 \
  --candidate-models logistic_regression \
  --max-features 50000 \
  --min-df 5 \
  --max-df 0.95 \
  --ngram-max 2 \
  --random-state 42 \
  --log-level INFO
```

Checkpoint files written by that command:

- `models/sentiment/amazon_polarity_lr.joblib`
- `models/sentiment/amazon_polarity_lr_feature_builder.joblib`
- `models/sentiment/amazon_polarity_lr_metadata.json`
- `models/sentiment/amazon_polarity_lr_summary.json`

Reuse the saved checkpoint without retraining:

```bash
python -m eWOM.sentiment_analysis.run_sentiment_benchmark \
  --model-output models/sentiment/amazon_polarity_lr \
  --reuse-existing-artifacts
```

Run direct sentiment inference from the saved checkpoint:

```bash
python -m eWOM.sentiment_analysis.run_sentiment_inference \
  --model-prefix models/sentiment/amazon_polarity_lr \
  --text "Battery life is excellent and setup was easy."
```

2. Train sentiment Multinomial Naive Bayes:
```bash
python -m eWOM.sentiment_analysis.run_sentiment_benchmark \
  --data-dir data/raw/amazon_polarity \
  --model-output models/sentiment/amazon_polarity_multinomial_nb \
  --val-ratio 0.1 \
  --candidate-models multinomial_nb \
  --max-features 50000 \
  --min-df 5 \
  --max-df 0.95 \
  --ngram-max 2 \
  --random-state 42 \
  --log-level INFO
```

Reuse the saved checkpoint without retraining:

```bash
python -m eWOM.sentiment_analysis.run_sentiment_benchmark \
  --model-output models/sentiment/amazon_polarity_multinomial_nb \
  --reuse-existing-artifacts
```

Run direct sentiment inference:

```bash
python -m eWOM.sentiment_analysis.run_sentiment_inference \
  --model-prefix models/sentiment/amazon_polarity_multinomial_nb \
  --text "This was disappointing and stopped working after two days."
```

3. Train sentiment Complement Naive Bayes:
```bash
python -m eWOM.sentiment_analysis.run_sentiment_benchmark \
  --data-dir data/raw/amazon_polarity \
  --model-output models/sentiment/amazon_polarity_complement_nb \
  --val-ratio 0.1 \
  --candidate-models complement_nb \
  --max-features 50000 \
  --min-df 5 \
  --max-df 0.95 \
  --ngram-max 2 \
  --random-state 42 \
  --log-level INFO
```

Reuse the saved checkpoint without retraining:

```bash
python -m eWOM.sentiment_analysis.run_sentiment_benchmark \
  --model-output models/sentiment/amazon_polarity_complement_nb \
  --reuse-existing-artifacts
```

Run direct sentiment inference:

```bash
python -m eWOM.sentiment_analysis.run_sentiment_inference \
  --model-prefix models/sentiment/amazon_polarity_complement_nb \
  --text "The quality is solid and I would buy it again."
```

4. Run the sentiment benchmark across all three:
```bash
python -m eWOM.sentiment_analysis.run_sentiment_benchmark \
  --data-dir data/raw/amazon_polarity \
  --model-output models/sentiment/amazon_polarity_full_benchmark \
  --val-ratio 0.1 \
  --candidate-models logistic_regression multinomial_nb complement_nb \
  --max-features 50000 \
  --min-df 5 \
  --max-df 0.95 \
  --ngram-max 2 \
  --random-state 42 \
  --log-level INFO
```

This benchmark command trains the listed candidates on the Amazon Polarity `train` split, compares them on the internal `val` split, saves the selected checkpoint under `models/sentiment/amazon_polarity_full_benchmark*`, and evaluates the selected checkpoint on the official `test` split.

Reuse the selected benchmark checkpoint without retraining:

```bash
python -m eWOM.sentiment_analysis.run_sentiment_benchmark \
  --model-output models/sentiment/amazon_polarity_full_benchmark \
  --reuse-existing-artifacts
```

Run direct sentiment inference from the selected benchmark checkpoint:

```bash
python -m eWOM.sentiment_analysis.run_sentiment_inference \
  --model-prefix models/sentiment/amazon_polarity_full_benchmark \
  --text "Battery life is excellent and setup was easy."
```

---

# No-Retrain Workflow

Train once to create the checkpoint. After that, do one of these:

- Reuse the saved artifacts with `--reuse-existing-artifacts` when you want the stored summary and metadata without fitting again.
- Use `python -m eWOM.helpfulness.run_helpfulness_inference ...` for helpfulness predictions.
- Use `python -m eWOM.sentiment_analysis.run_sentiment_inference ...` for sentiment predictions.
- Use `python -m eWOM.run_fusion_demo ...` with explicit model paths if you want full eWOM scoring from the saved helpfulness and sentiment checkpoints.
