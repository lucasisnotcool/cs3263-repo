from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping


@dataclass(frozen=True)
class DiscreteBayesNode:
    name: str
    states: tuple[str, ...]
    parents: tuple[str, ...]
    cpt: Mapping[tuple[str, ...], Mapping[str, float]]


class DiscreteBayesianNetwork:
    def __init__(self, nodes: list[DiscreteBayesNode]):
        if not nodes:
            raise ValueError("nodes must not be empty")

        self._nodes_in_order = nodes
        self.nodes = {node.name: node for node in nodes}
        if len(self.nodes) != len(nodes):
            raise ValueError("node names must be unique")

        self.order = tuple(node.name for node in nodes)
        self._validate()

    def posterior(self, query_node: str, evidence: Mapping[str, str]) -> dict[str, float]:
        node = self._require_node(query_node)
        normalized_evidence = dict(evidence)

        invalid_keys = sorted(set(normalized_evidence) - set(self.nodes))
        if invalid_keys:
            raise ValueError(f"unknown evidence keys: {invalid_keys}")
        for key, value in normalized_evidence.items():
            allowed_states = self.nodes[key].states
            if value not in allowed_states:
                raise ValueError(
                    f"invalid state {value!r} for node {key!r}; expected one of {allowed_states}"
                )

        unnormalized: dict[str, float] = {}
        for state in node.states:
            scoped_evidence = dict(normalized_evidence)
            scoped_evidence[query_node] = state
            unnormalized[state] = self._enumerate_all(self.order, scoped_evidence)

        total = sum(unnormalized.values())
        if total <= 0.0:
            uniform_probability = 1.0 / len(node.states)
            return {state: uniform_probability for state in node.states}

        return {state: value / total for state, value in unnormalized.items()}

    def _enumerate_all(
        self,
        remaining_nodes: tuple[str, ...],
        evidence: dict[str, str],
    ) -> float:
        if not remaining_nodes:
            return 1.0

        node_name = remaining_nodes[0]
        rest = remaining_nodes[1:]
        node = self.nodes[node_name]

        if node_name in evidence:
            probability = self._conditional_probability(node_name, evidence[node_name], evidence)
            return probability * self._enumerate_all(rest, evidence)

        total = 0.0
        for state in node.states:
            scoped_evidence = dict(evidence)
            scoped_evidence[node_name] = state
            probability = self._conditional_probability(node_name, state, scoped_evidence)
            total += probability * self._enumerate_all(rest, scoped_evidence)
        return total

    def _conditional_probability(
        self,
        node_name: str,
        state: str,
        assignment: Mapping[str, str],
    ) -> float:
        node = self.nodes[node_name]
        parent_key = tuple(assignment[parent] for parent in node.parents)
        try:
            distribution = node.cpt[parent_key]
        except KeyError as exc:
            raise ValueError(
                f"missing CPT row for node {node_name!r} with parents {parent_key}"
            ) from exc

        try:
            return float(distribution[state])
        except KeyError as exc:
            raise ValueError(
                f"missing state {state!r} in CPT row for node {node_name!r}"
            ) from exc

    def _require_node(self, node_name: str) -> DiscreteBayesNode:
        try:
            return self.nodes[node_name]
        except KeyError as exc:
            raise ValueError(f"unknown node: {node_name}") from exc

    def _validate(self) -> None:
        seen_names: set[str] = set()
        for node in self._nodes_in_order:
            for parent in node.parents:
                if parent not in self.nodes:
                    raise ValueError(f"node {node.name!r} references missing parent {parent!r}")
                if parent not in seen_names:
                    raise ValueError(
                        f"node {node.name!r} must appear after parent {parent!r} in topological order"
                    )

            if not node.states:
                raise ValueError(f"node {node.name!r} must have at least one state")

            expected_parent_rows = 1
            for parent_name in node.parents:
                expected_parent_rows *= len(self.nodes[parent_name].states)

            if len(node.cpt) != expected_parent_rows:
                raise ValueError(
                    f"node {node.name!r} expected {expected_parent_rows} CPT rows "
                    f"but found {len(node.cpt)}"
                )

            for parent_key, distribution in node.cpt.items():
                if len(parent_key) != len(node.parents):
                    raise ValueError(
                        f"node {node.name!r} has invalid parent key length for {parent_key}"
                    )

                total = 0.0
                for state in node.states:
                    probability = float(distribution.get(state, -1.0))
                    if probability < 0.0:
                        raise ValueError(
                            f"node {node.name!r} has invalid probability for state {state!r}"
                        )
                    total += probability

                if abs(total - 1.0) > 1e-9:
                    raise ValueError(
                        f"node {node.name!r} CPT row {parent_key} sums to {total}, expected 1.0"
                    )

            seen_names.add(node.name)
