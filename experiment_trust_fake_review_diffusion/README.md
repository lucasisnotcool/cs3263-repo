# Fake Review Diffusion Experiment

This is a standalone experiment folder for diffusion-based fake-vs-real review modeling.

It mirrors the experiment style of `experiment_trust_fake_reviews`:

- notebook entrypoint
- Python pipeline module
- artifacts output directory
- run metadata (`run_info.json`) and phase metrics CSV

## Files

- [`experiment_diffusion_review.ipynb`](/Users/lohzh/Desktop/cs3263-repo/experiment_trust_fake_review_diffusion/experiment_diffusion_review.ipynb): notebook entrypoint
- [`diffusion_review_pipeline.py`](/Users/lohzh/Desktop/cs3263-repo/experiment_trust_fake_review_diffusion/diffusion_review_pipeline.py): importable module + CLI
- [`__init__.py`](/Users/lohzh/Desktop/cs3263-repo/experiment_trust_fake_review_diffusion/__init__.py): package exports
- [`artifacts/diffusion_fake_review/`](/Users/lohzh/Desktop/cs3263-repo/experiment_trust_fake_review_diffusion/artifacts/diffusion_fake_review): generated outputs

## Dataset and Labels

- Dataset: `data/raw/fake-reviews-dataset/fake reviews dataset.csv`
- Label mapping:
  - `OR` -> `1` (real/original review)
  - `CG` -> `0` (fake/computer-generated review)

## Config Expectations (mirrors trust-fake-reviews notebook defaults)

- `PHASE_A_TARGET_ROWS = 240` (balanced sample)
- `TEST_SIZE = 0.25` (stratified split)
- `RANDOM_STATE = 42`

## Run from CLI

```bash
python -m experiment_trust_fake_review_diffusion.diffusion_review_pipeline \
  --phase-a-target-rows 240 \
  --test-size 0.25 \
  --random-state 42
```

Optional scoring after training:

```bash
python -m experiment_trust_fake_review_diffusion.diffusion_review_pipeline \
  --score-text "Excellent quality and exactly as described." \
  --score-text "Buy now!!! limited stock!!!"
```

## Artifacts Produced

- `phase_a_metrics.csv`
- `phase_a_test_predictions.csv`
- `diffusion_model_bundle.joblib`
- `run_info.json`

## Notebook Long-Run Cell

The notebook includes an optional **\"Optional Long Run (~10 minutes)\"** cell that increases:

- `phase_a_target_rows` (default `12000`)
- diffusion training footprint
- feature/model capacity

It saves to a separate artifact folder:

- `experiment_trust_fake_review_diffusion/artifacts/diffusion_fake_review_long_run`

You can override defaults before executing that cell with:

```python
RUN_LONG = {
    "phase_a_target_rows": 12000,  # set 0 to use full dataset
    "diffusion_steps": 80,
    "latent_dim": 256,
    "max_features": 50000,
    "denoiser_samples_per_row": 12,
    "classifier_samples_per_row": 6,
    "inference_samples": 64,
}
```
