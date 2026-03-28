# Value Agent

Deterministic two-listing comparison agent for value-based decision support.

## Design Rules

- Compare exactly two listings: `listing_a` vs `listing_b`.
- Consume only listing/value fields from URL extraction (no Trust/eWOM dependency).
- Use deterministic scoring for reproducibility and auditability.
- If evidence is weak, return `insufficient_evidence` instead of guessing.

## Scoring Rubric (v1 defaults)

- `cost` weight: `0.60` (lower total price is better)
- `spec` weight: `0.25` (coverage + pairwise consistency/value)
- `service` weight: `0.15` (delivery, returns, warranty)

Final scores:

- `value_score_a = 0.60*cost_a + 0.25*spec_a + 0.15*service_a`
- `value_score_b = 0.60*cost_b + 0.25*spec_b + 0.15*service_b`

Verdict:

- `insufficient_evidence` if `confidence < 0.45`
- `tie` if `|value_score_a - value_score_b| <= 3.0`
- else `better_A` or `better_B`

## Confidence Policy

Confidence starts at `1.0` and is reduced by:

- missing fields
- estimated total price
- low listing comparability (title/spec mismatch)
- currency mismatch
- low required-spec coverage

This enforces responsible behavior for ambiguous/incomplete data.

## Input Contract

Required top-level keys:

- `listing_a`
- `listing_b`

Optional top-level key:

- `required_spec_keys`

Important listing fields:

- `url` (required)
- `platform`, `title`, `currency`
- `base_price`, `shipping_fee`, `platform_discount`, `seller_discount`, `voucher_discount`, `total_price`
- `delivery_days`, `warranty_months`, `return_window_days`
- `specs` (object)

## Output Contract

- `value_score_a`
- `value_score_b`
- `confidence`
- `verdict`: `better_A | better_B | tie | insufficient_evidence`
- `reasons` (human-readable justifications)
- `evidence` (structured trace for Decision Agent)
- `missing_fields`

## CLI Commands

Run with built-in mock data:

```bash
python -m value.cli --mock
```

Pretty JSON output:

```bash
python -m value.cli --mock --pretty
```

Run with your own JSON payload:

```bash
python -m value.cli --input ./sample_payload.json --pretty
```
