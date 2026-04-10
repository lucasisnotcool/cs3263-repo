from __future__ import annotations

import argparse
from collections import defaultdict
from itertools import product
import json
import logging
import math
import sys
from pathlib import Path
from typing import Any, Mapping

# Allow `python value/train_bayesian_value_model.py` to resolve package imports.
if __package__ in {None, ""}:
    sys.path.append(str(Path(__file__).resolve().parents[1]))

from value.bayes import DiscreteBayesNode, DiscreteBayesianNetwork
from value.bayesian_value import (
    BayesianValueInput,
    build_value_evidence,
    default_bayesian_value_network,
)


LOGGER = logging.getLogger(__name__)
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_TRAIN_PATH = (
    PROJECT_ROOT
    / "data"
    / "value"
    / "bayesian_training"
    / "amazon_electronics_bayesian_train.jsonl"
)
DEFAULT_OUTPUT_PATH = PROJECT_ROOT / "value" / "artifacts" / "amazon_bayesian_value_electronics.json"
ROOT_NODE_NAMES = (
    "TrustSignal",
    "ReviewPolarity",
    "ReviewStrength",
    "RatingSignal",
    "ReviewVolume",
    "VerifiedSignal",
    "RelativePriceBucket",
    "WarrantySignal",
    "ReturnSignal",
)
PSEUDO_OBSERVED_NODE_NAMES = (
    "Trustworthiness",
    "ReviewEvidence",
    "ServiceSupport",
    "ProductQuality",
)
TARGET_NODE_NAME = "GoodValueForMoney"


def configure_logging(level: str = "INFO") -> None:
    resolved_level = getattr(logging, str(level).upper(), logging.INFO)
    logging.basicConfig(
        level=resolved_level,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )


def train_bayesian_value_network(
    *,
    dataset_path: str | Path,
    output_path: str | Path,
    smoothing: float = 1.0,
    max_rows: int | None = None,
) -> dict[str, Any]:
    if smoothing < 0.0:
        raise ValueError("smoothing must be non-negative.")

    default_network = default_bayesian_value_network()
    assignment_cache: dict[tuple[tuple[str, str], ...], dict[str, str]] = {}
    counts: dict[str, dict[tuple[str, ...], dict[str, float]]] = {
        node_name: defaultdict(lambda: defaultdict(float)) for node_name in default_network.order
    }
    summary: dict[str, Any] = {
        "dataset_path": str(Path(dataset_path).expanduser().resolve()),
        "output_path": str(Path(output_path).expanduser().resolve()),
        "smoothing": float(smoothing),
        "max_rows": max_rows,
        "rows_seen": 0,
        "rows_used": 0,
        "rows_skipped": {},
        "target_counts": {"yes": 0, "no": 0},
    }

    resolved_dataset_path = Path(dataset_path).expanduser().resolve()
    with resolved_dataset_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            if max_rows is not None and summary["rows_seen"] >= max_rows:
                break
            row = json.loads(line)
            summary["rows_seen"] += 1
            label = _resolve_label(row)
            if label is None:
                _increment(summary["rows_skipped"], "missing_or_invalid_label")
                continue

            evidence = _resolve_evidence(row)
            if "RelativePriceBucket" not in evidence:
                _increment(summary["rows_skipped"], "missing_relative_price_bucket")
                continue

            assignments = _cached_complete_assignments(
                evidence=evidence,
                default_network=default_network,
                assignment_cache=assignment_cache,
            )
            if not _has_parent_assignments(default_network.nodes[TARGET_NODE_NAME], assignments):
                _increment(summary["rows_skipped"], "missing_target_parent_assignment")
                continue
            assignments[TARGET_NODE_NAME] = label
            _collect_counts(
                assignments=assignments,
                counts=counts,
                default_network=default_network,
            )
            summary["rows_used"] += 1
            summary["target_counts"][label] += 1

    trained_network = _build_smoothed_network(
        default_network=default_network,
        counts=counts,
        smoothing=smoothing,
    )
    summary["learned_cpt_rows"] = {
        node_name: int(sum(1 for row_counts in node_counts.values() if sum(row_counts.values()) > 0.0))
        for node_name, node_counts in counts.items()
    }
    summary["cached_evidence_patterns"] = len(assignment_cache)
    payload = network_to_payload(
        trained_network,
        metadata={
            "source": "value.train_bayesian_value_model",
            "training_summary": summary,
        },
    )
    resolved_output_path = Path(output_path).expanduser().resolve()
    resolved_output_path.parent.mkdir(parents=True, exist_ok=True)
    with resolved_output_path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)
        handle.write("\n")

    return summary


def _collect_counts(
    *,
    assignments: Mapping[str, str],
    counts: dict[str, dict[tuple[str, ...], dict[str, float]]],
    default_network: DiscreteBayesianNetwork,
) -> None:
    for root_name in ROOT_NODE_NAMES:
        if root_name in assignments:
            counts[root_name][()][assignments[root_name]] += 1.0

    for node_name in PSEUDO_OBSERVED_NODE_NAMES:
        node = default_network.nodes[node_name]
        if _has_parent_assignments(node, assignments):
            parent_key = tuple(assignments[parent] for parent in node.parents)
            counts[node_name][parent_key][assignments[node_name]] += 1.0

    target_node = default_network.nodes[TARGET_NODE_NAME]
    target_parent_key = tuple(assignments[parent] for parent in target_node.parents)
    counts[TARGET_NODE_NAME][target_parent_key][assignments[TARGET_NODE_NAME]] += 1.0


def load_bayesian_value_network(path: str | Path) -> DiscreteBayesianNetwork:
    resolved_path = Path(path).expanduser().resolve()
    with resolved_path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    nodes = []
    for node_payload in payload["nodes"]:
        nodes.append(
            DiscreteBayesNode(
                name=str(node_payload["name"]),
                states=tuple(str(state) for state in node_payload["states"]),
                parents=tuple(str(parent) for parent in node_payload["parents"]),
                cpt={
                    tuple(str(part) for part in row_payload["parents"]): {
                        str(state): float(probability)
                        for state, probability in row_payload["probabilities"].items()
                    }
                    for row_payload in node_payload["cpt"]
                },
            )
        )
    return DiscreteBayesianNetwork(nodes)


def network_to_payload(
    network: DiscreteBayesianNetwork,
    *,
    metadata: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    return {
        "format": "value.discrete_bayesian_network.v1",
        "metadata": dict(metadata or {}),
        "nodes": [
            {
                "name": node.name,
                "states": list(node.states),
                "parents": list(node.parents),
                "cpt": [
                    {
                        "parents": list(parent_key),
                        "probabilities": {
                            state: float(distribution[state]) for state in node.states
                        },
                    }
                    for parent_key, distribution in _sorted_cpt_rows(node)
                ],
            }
            for node in (network.nodes[node_name] for node_name in network.order)
        ],
    }


def _resolve_label(row: Mapping[str, Any]) -> str | None:
    label = str(row.get("good_value_label") or row.get("label") or "").strip().lower()
    if label in {"yes", "worth_buying", "good_value", "1", "true"}:
        return "yes"
    if label in {"no", "not_worth_buying", "not_good_value", "0", "false"}:
        return "no"
    return None


def _resolve_evidence(row: Mapping[str, Any]) -> dict[str, str]:
    evidence = row.get("bayesian_evidence")
    if isinstance(evidence, Mapping):
        return {str(key): str(value) for key, value in evidence.items() if value is not None}

    bayesian_input = BayesianValueInput.from_mapping(row)
    resolved_evidence, _ = build_value_evidence(bayesian_input)
    return resolved_evidence


def _build_complete_assignments(
    *,
    evidence: Mapping[str, str],
    default_network: DiscreteBayesianNetwork,
) -> dict[str, str]:
    assignments = {
        node_name: state
        for node_name, state in evidence.items()
        if node_name in default_network.nodes
    }
    for node_name in PSEUDO_OBSERVED_NODE_NAMES:
        posterior = default_network.posterior(node_name, evidence)
        assignments[node_name] = max(posterior, key=posterior.get)
    return assignments


def _cached_complete_assignments(
    *,
    evidence: Mapping[str, str],
    default_network: DiscreteBayesianNetwork,
    assignment_cache: dict[tuple[tuple[str, str], ...], dict[str, str]],
) -> dict[str, str]:
    cache_key = tuple(sorted((str(key), str(value)) for key, value in evidence.items()))
    cached = assignment_cache.get(cache_key)
    if cached is None:
        cached = _build_complete_assignments(
            evidence=evidence,
            default_network=default_network,
        )
        assignment_cache[cache_key] = cached
    return dict(cached)


def _build_smoothed_network(
    *,
    default_network: DiscreteBayesianNetwork,
    counts: Mapping[str, Mapping[tuple[str, ...], Mapping[str, float]]],
    smoothing: float,
) -> DiscreteBayesianNetwork:
    trained_nodes: list[DiscreteBayesNode] = []
    for node_name in default_network.order:
        node = default_network.nodes[node_name]
        cpt: dict[tuple[str, ...], dict[str, float]] = {}
        for parent_key in _all_parent_keys(node, default_network):
            cpt[parent_key] = _smooth_distribution(
                states=node.states,
                counts=counts.get(node_name, {}).get(parent_key, {}),
                prior=node.cpt[parent_key],
                smoothing=smoothing,
            )
        trained_nodes.append(
            DiscreteBayesNode(
                name=node.name,
                states=node.states,
                parents=node.parents,
                cpt=cpt,
            )
        )
    return DiscreteBayesianNetwork(trained_nodes)


def _smooth_distribution(
    *,
    states: tuple[str, ...],
    counts: Mapping[str, float],
    prior: Mapping[str, float],
    smoothing: float,
) -> dict[str, float]:
    observed_total = sum(float(counts.get(state, 0.0)) for state in states)
    if observed_total <= 0.0 and smoothing <= 0.0:
        return {state: float(prior[state]) for state in states}
    total = observed_total + smoothing
    if total <= 0.0:
        uniform = 1.0 / len(states)
        return {state: uniform for state in states}
    distribution = {
        state: (float(counts.get(state, 0.0)) + (smoothing * float(prior[state]))) / total
        for state in states
    }
    return _normalize(distribution)


def _all_parent_keys(
    node: DiscreteBayesNode,
    network: DiscreteBayesianNetwork,
) -> list[tuple[str, ...]]:
    if not node.parents:
        return [()]
    parent_states = [network.nodes[parent].states for parent in node.parents]
    return [tuple(values) for values in product(*parent_states)]


def _sorted_cpt_rows(
    node: DiscreteBayesNode,
) -> list[tuple[tuple[str, ...], Mapping[str, float]]]:
    return sorted(node.cpt.items(), key=lambda item: item[0])


def _has_parent_assignments(
    node: DiscreteBayesNode,
    assignments: Mapping[str, str],
) -> bool:
    return all(parent in assignments for parent in node.parents)


def _normalize(distribution: Mapping[str, float]) -> dict[str, float]:
    total = float(sum(distribution.values()))
    if total <= 0.0 or not math.isfinite(total):
        uniform = 1.0 / len(distribution)
        return {state: uniform for state in distribution}
    return {state: float(value) / total for state, value in distribution.items()}


def _increment(counts: dict[str, int], key: str) -> None:
    counts[key] = int(counts.get(key, 0)) + 1


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Train a discrete Bayesian value network from labeled Amazon Electronics "
            "good-value rows."
        )
    )
    parser.add_argument("--dataset-path", type=Path, default=DEFAULT_TRAIN_PATH)
    parser.add_argument("--output-path", type=Path, default=DEFAULT_OUTPUT_PATH)
    parser.add_argument("--smoothing", type=float, default=1.0)
    parser.add_argument("--max-rows", type=int, default=None)
    parser.add_argument("--log-level", type=str, default="INFO")
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    configure_logging(args.log_level)
    summary = train_bayesian_value_network(
        dataset_path=args.dataset_path,
        output_path=args.output_path,
        smoothing=args.smoothing,
        max_rows=args.max_rows,
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
