# Fake Reviews Trust Deploy Pipeline

This folder now includes a deploy-ready runtime for the fake-review-calibrated LLM-BN pipeline from [`experiment_llm_trust_graph.ipynb`](/Users/lohzh/Desktop/cs3263-repo/experiment_trust_fake_reviews/experiment_llm_trust_graph.ipynb).

The deploy entrypoint is [`deploy_pipeline.py`](/Users/lohzh/Desktop/cs3263-repo/experiment_trust_fake_reviews/deploy_pipeline.py). It:

- accepts one or many products
- labels them with Ollama using the same trust schema as the notebook
- applies the saved Bayesian graph and saved logistic baseline
- returns one standard JSON schema
- does not retrain anything at runtime
- validates Ollama and model availability before scoring

## Dev Shortcut

Minimal case:

```python
from experiment_trust_fake_reviews import run_deployment_pipeline

result = run_deployment_pipeline(
    [
        {
            "product_id": "demo-1",
            "title": "USB-C Hub",
            "bullet_points": "4K HDMI; USB 3.0; PD passthrough",
            "description": "Six-port aluminium hub.",
        }
    ]
)

row = result["results"][0]
key_score = row["scores"]["trust_risk_index_graph"]
print(key_score)
```

Use this first:

- `trust_risk_index_graph`: main deploy risk score. Higher means more trust risk.

Other scores:

- `phase_b_truth_likelihood_graph`: BN-estimated probability the product content looks trustworthy.
- `trust_risk_index_logistic`: alternate risk score from the saved logistic baseline.
- `phase_b_truth_likelihood_logistic`: alternate trust probability from the saved logistic baseline.
- `graph_uncertainty_entropy`: BN uncertainty. Higher means the graph is less sure.

## Files

- [`deploy_pipeline.py`](/Users/lohzh/Desktop/cs3263-repo/experiment_trust_fake_reviews/deploy_pipeline.py): importable module and CLI entrypoint
- [`__init__.py`](/Users/lohzh/Desktop/cs3263-repo/experiment_trust_fake_reviews/__init__.py): package exports
- [`artifacts/llm_trust_graph/graph_model.json`](/Users/lohzh/Desktop/cs3263-repo/experiment_trust_fake_reviews/artifacts/llm_trust_graph/graph_model.json): saved BN graph
- [`artifacts/llm_trust_graph/logistic_model.json`](/Users/lohzh/Desktop/cs3263-repo/experiment_trust_fake_reviews/artifacts/llm_trust_graph/logistic_model.json): saved logistic baseline coefficients
- [`artifacts/llm_trust_graph/deploy_labels.jsonl`](/Users/lohzh/Desktop/cs3263-repo/experiment_trust_fake_reviews/artifacts/llm_trust_graph/deploy_labels.jsonl): runtime cache for successful labels, created on first run

## Runtime Requirements

- Python environment with packages from [`requirements.txt`](/Users/lohzh/Desktop/cs3263-repo/requirements.txt)
- Ollama installed and on `PATH`
- Local Ollama service reachable
- Model `llama3.1:8b` pulled locally, unless you override `--model`

Recommended checks:

```bash
ollama serve
ollama pull llama3.1:8b
python -m experiment_trust_fake_reviews.deploy_pipeline --check-env
```

If the environment is wrong, the script returns structured error details under `environment.errors` and `environment.suggested_commands`.

## Python Usage

```python
from experiment_trust_fake_reviews import run_deployment_pipeline

result = run_deployment_pipeline(
    [
        {
            "product_id": "sku-001",
            "title": "Portable Espresso Maker",
            "bullet_points": "Manual pressure pump; Travel size; Reusable filter",
            "description": "Compact espresso maker for camping and travel.",
        },
        {
            "product_id": "sku-002",
            "text": "Title: Miracle detox patch Bullet Points: Lose 10kg overnight Description: Limited stock, act now.",
        },
    ]
)

for row in result["results"]:
    print(row["record_id"], row["status"], row["scores"])
```

If you want explicit control:

```python
from experiment_trust_fake_reviews import DeployConfig, TrustFakeReviewsDeployPipeline

pipeline = TrustFakeReviewsDeployPipeline(
    DeployConfig(
        ollama_model="llama3.1:8b",
    )
)

env = pipeline.validate_environment()
if not env["ok"]:
    raise RuntimeError(env)

result = pipeline.run(
    {
        "products": [
            {
                "product_id": "sku-003",
                "title": "USB-C Hub",
                "bullet_points": "4K HDMI; USB 3.0; PD passthrough",
                "description": "Aluminium hub with six ports.",
            }
        ]
    }
)
```

## CLI Usage

Input file example:

```json
[
  {
    "product_id": "sku-001",
    "title": "Portable Espresso Maker",
    "bullet_points": "Manual pressure pump; Travel size; Reusable filter",
    "description": "Compact espresso maker for camping and travel."
  },
  {
    "product_id": "sku-002",
    "title": "Miracle Detox Patch",
    "bullet_points": "Lose weight fast; Limited stock",
    "description": "Results guaranteed overnight."
  }
]
```

Run:

```bash
python -m experiment_trust_fake_reviews.deploy_pipeline \
  --input /path/to/products.json \
  --output /path/to/results.json
```

Or inline:

```bash
python -m experiment_trust_fake_reviews.deploy_pipeline \
  --product-json '{"product_id":"sku-001","title":"USB-C Hub","bullet_points":"4K HDMI","description":"Six-port hub"}'
```

Or via stdin:

```bash
cat /path/to/products.json | python -m experiment_trust_fake_reviews.deploy_pipeline --stdin
```

## Accepted Input Shape

Each product can provide either:

- `text`

or:

- `title`
- `bullet_points`
- `description`

Optional fields:

- `product_id`
- `product_type_id`
- `record_id`

The loader also accepts notebook-style uppercase keys:

- `PRODUCT_ID`
- `PRODUCT_TYPE_ID`
- `TITLE`
- `BULLET_POINTS`
- `DESCRIPTION`

If `record_id` is omitted, the script uses `product_id` when present, otherwise it creates a stable hashed ID from the normalized content.

## Output Schema

Top level:

```json
{
  "schema_version": "trust_fake_reviews_deploy/v1",
  "generated_at_utc": "...",
  "pipeline": {
    "mode": "deploy",
    "module": "experiment_trust_fake_reviews.deploy_pipeline",
    "ollama_model": "llama3.1:8b",
    "artifacts_dir": "...",
    "cache_path": "...",
    "artifacts": {
      "graph_model": "...",
      "logistic_model": "..."
    }
  },
  "environment": {
    "ok": true,
    "errors": [],
    "warnings": [],
    "suggested_commands": []
  },
  "results": [...],
  "summary": {...}
}
```

Per product result:

```json
{
  "record_id": "sku-001",
  "status": "ok",
  "input": {
    "product_id": "sku-001",
    "product_type_id": null,
    "title": "Portable Espresso Maker",
    "bullet_points": "Manual pressure pump; Travel size; Reusable filter",
    "description": "Compact espresso maker for camping and travel.",
    "text": "Title: Portable Espresso Maker\nBullet Points: ...\nDescription: ..."
  },
  "labels": {
    "claim_trust_score": 0.0,
    "signal_trust_score": 0.0,
    "heuristic_pressure_score": 0.0,
    "competence_score": 0.0,
    "benevolence_score": 0.0,
    "integrity_score": 0.0,
    "predictability_score": 0.0,
    "claim_trust_bucket": "low|medium|high",
    "signal_trust_bucket": "low|medium|high",
    "heuristic_pressure_bucket": "low|medium|high",
    "competence_bucket": "low|medium|high",
    "benevolence_bucket": "low|medium|high",
    "integrity_bucket": "low|medium|high",
    "predictability_bucket": "low|medium|high",
    "rationale_claim": "...",
    "rationale_signal": "...",
    "rationale_pressure": "...",
    "overall_confidence": 0.0
  },
  "scores": {
    "phase_b_truth_likelihood_graph": 0.0,
    "phase_b_truth_likelihood_logistic": 0.0,
    "trust_risk_index_graph": 0.0,
    "trust_risk_index_logistic": 0.0,
    "graph_uncertainty_entropy": 0.0
  },
  "error": null
}
```

Error rows keep the same outer structure but set `labels` and `scores` to `null` and populate `error`.

## Notes

- Runtime scoring uses the saved graph and logistic artifacts only. No phase-A retraining happens during deploy runs.
- Successful label responses are cached to reduce repeat Ollama calls.
- The notebook’s BN and logistic outputs are both preserved in deploy mode so downstream modules can choose either score.
