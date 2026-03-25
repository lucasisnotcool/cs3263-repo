from __future__ import annotations

import json
import math
import random
import sys
from collections import Counter
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Iterator

# Allow `python eWOM/helpfulness/train_test_splitter.py` to resolve package imports.
if __package__ in {None, ""}:
    sys.path.append(str(Path(__file__).resolve().parents[2]))

from eWOM.helpfulness.dataset_loader import AmazonReviewsLoader


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data"

# Edit these constants directly before running the script.
REVIEW_PATH = Path("data/raw/amazon-reviews-2023/Electronics.jsonl")
OUTPUT_DIR = PROJECT_ROOT / "data" / "helpfulness"
TRAIN_OUTPUT_PATH = OUTPUT_DIR / "train.jsonl"
TEST_OUTPUT_PATH = OUTPUT_DIR / "test.jsonl"
SUMMARY_OUTPUT_PATH = OUTPUT_DIR / "split_summary.json"

TEST_SIZE = 0.2
POSITIVE_THRESHOLD = 3
DROP_MIDDLE = False
MIN_REVIEW_WORDS = 0
MAX_ROWS = None
RANDOM_STATE = 42
OVERWRITE_OUTPUT = True
LOG_EVERY_ROWS = 50_000
SHUFFLE_BUFFER_SIZE = 256


@dataclass
class ScanStats:
    raw_rows_seen: int = 0
    eligible_rows: int = 0
    skipped_rows: int = 0
    label_counts: Counter[int] = field(default_factory=Counter)


@dataclass
class SplitPlan:
    test_targets: dict[int, int]
    train_targets: dict[int, int]


@dataclass
class AssignmentStats:
    train_label_counts: Counter[int] = field(default_factory=Counter)
    test_label_counts: Counter[int] = field(default_factory=Counter)


class BufferedJsonlWriter:
    def __init__(self, path: Path, seed: int, buffer_size: int):
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._file = self.path.open("w", encoding="utf-8")
        self._rng = random.Random(seed)
        self._buffer_size = max(0, buffer_size)
        self._buffer: list[dict[str, Any]] = []

    def write(self, record: dict[str, Any]) -> None:
        if self._buffer_size <= 1:
            self._write_record(record)
            return

        self._buffer.append(record)
        if len(self._buffer) > self._buffer_size:
            index = self._rng.randrange(len(self._buffer))
            self._write_record(self._buffer.pop(index))

    def close(self) -> None:
        while self._buffer:
            index = self._rng.randrange(len(self._buffer))
            self._write_record(self._buffer.pop(index))
        self._file.close()

    def _write_record(self, record: dict[str, Any]) -> None:
        json.dump(record, self._file, ensure_ascii=False)
        self._file.write("\n")


def log(message: str) -> None:
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{timestamp}] {message}", flush=True)


def resolve_path(path: Path) -> Path:
    if path.is_absolute():
        return path
    return PROJECT_ROOT / path


def resolve_review_path() -> Path:
    configured_path = resolve_path(REVIEW_PATH)
    if configured_path.exists():
        return configured_path

    candidates = sorted(
        path
        for pattern in ("*.jsonl", "*.jsonl.gz")
        for path in DATA_DIR.rglob(pattern)
        if OUTPUT_DIR not in path.parents
    )
    if len(candidates) == 1:
        log(f"Configured review path not found. Auto-detected source file: {candidates[0]}")
        return candidates[0]

    candidate_lines = "\n".join(f"  - {path}" for path in candidates[:10]) or "  - none found"
    raise FileNotFoundError(
        "Review file not found.\n"
        f"Configured path: {configured_path}\n"
        f"Project root: {PROJECT_ROOT}\n"
        "Available JSONL candidates under data/:\n"
        f"{candidate_lines}\n"
        "Update REVIEW_PATH at the top of train_test_splitter.py to your actual source JSONL file."
    )


def ensure_output_dir(overwrite: bool) -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    generated_paths = [TRAIN_OUTPUT_PATH, TEST_OUTPUT_PATH, SUMMARY_OUTPUT_PATH]
    existing_paths = [path for path in generated_paths if path.exists()]
    if existing_paths and not overwrite:
        existing_str = ", ".join(str(path) for path in existing_paths)
        raise FileExistsError(
            "Output files already exist. Set OVERWRITE_OUTPUT = True to replace them: "
            f"{existing_str}"
        )

    for path in existing_paths:
        path.unlink()

    existing_fold_dirs = sorted(OUTPUT_DIR.glob("fold_*"))
    if existing_fold_dirs:
        log("Existing fold_* directories detected. This script leaves them untouched.")


def coerce_int(value: Any, default: int = 0) -> int:
    if value is None:
        return default
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        if math.isnan(value):
            return default
        return int(value)

    text = str(value).strip()
    if not text:
        return default
    try:
        return int(float(text))
    except ValueError:
        return default


def coerce_float(value: Any) -> float | None:
    if value is None:
        return None
    if isinstance(value, bool):
        return float(value)
    if isinstance(value, (int, float)):
        if isinstance(value, float) and math.isnan(value):
            return None
        return float(value)

    text = str(value).strip()
    if not text:
        return None
    try:
        return float(text)
    except ValueError:
        return None


def coerce_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return False
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in {"true", "1", "yes", "y"}:
            return True
        if lowered in {"false", "0", "no", "n", ""}:
            return False
    return bool(value)


def prepare_record(row: dict[str, Any]) -> dict[str, Any] | None:
    title = str(row.get("title") or "")
    text = str(row.get("text") or "")
    helpful_votes = coerce_int(row.get("helpful_votes", 0), default=0)
    review_len_words = len(text.split())

    if DROP_MIDDLE and 0 < helpful_votes < POSITIVE_THRESHOLD:
        return None
    if review_len_words < MIN_REVIEW_WORDS:
        return None

    return {
        "rating": coerce_float(row.get("rating")),
        "title": title,
        "text": text,
        "verified_purchase": coerce_bool(row.get("verified_purchase", False)),
        "helpful_votes": helpful_votes,
        "asin": row.get("asin"),
        "parent_asin": row.get("parent_asin"),
        "user_id": row.get("user_id"),
        "timestamp": row.get("timestamp"),
        "label": int(helpful_votes >= POSITIVE_THRESHOLD),
        "review_len_words": review_len_words,
    }


def iter_prepared_records(review_path: Path) -> Iterator[dict[str, Any] | None]:
    loader = AmazonReviewsLoader(review_path, max_rows=MAX_ROWS)
    for row in loader.iter_rows():
        yield prepare_record(row)


def scan_dataset(review_path: Path) -> ScanStats:
    stats = ScanStats()
    log(f"Scanning dataset from {review_path}")

    for row in iter_prepared_records(review_path):
        stats.raw_rows_seen += 1
        if row is None:
            stats.skipped_rows += 1
        else:
            label = int(row["label"])
            stats.eligible_rows += 1
            stats.label_counts[label] += 1

        if stats.raw_rows_seen % LOG_EVERY_ROWS == 0:
            log(
                "Scan progress: "
                f"seen {stats.raw_rows_seen:,} rows | "
                f"eligible {stats.eligible_rows:,} | "
                f"skipped {stats.skipped_rows:,}"
            )

    if stats.eligible_rows == 0:
        raise ValueError("No rows remain after filtering.")

    log(
        "Finished scan: "
        f"seen {stats.raw_rows_seen:,} rows | "
        f"eligible {stats.eligible_rows:,} | "
        f"skipped {stats.skipped_rows:,}"
    )
    return stats


def allocate_test_targets(label_counts: Counter[int], test_size: float) -> dict[int, int]:
    total_rows = sum(label_counts.values())
    total_test_rows = int(round(total_rows * test_size))

    exact_targets = {
        label: label_counts[label] * test_size
        for label in label_counts
    }
    targets = {
        label: int(math.floor(exact_targets[label]))
        for label in label_counts
    }

    remaining = total_test_rows - sum(targets.values())
    if remaining > 0:
        ranked_labels = sorted(
            label_counts,
            key=lambda label: (exact_targets[label] - targets[label], label_counts[label]),
            reverse=True,
        )
        for label in ranked_labels[:remaining]:
            targets[label] += 1

    return targets


def build_split_plan(stats: ScanStats) -> SplitPlan:
    if not 0 < TEST_SIZE < 1:
        raise ValueError("TEST_SIZE must be between 0 and 1.")
    if len(stats.label_counts) < 2:
        raise ValueError("Splitting requires at least two label classes.")

    test_targets = allocate_test_targets(stats.label_counts, TEST_SIZE)
    train_targets = {
        label: stats.label_counts[label] - test_targets[label]
        for label in stats.label_counts
    }

    if any(count <= 0 for count in train_targets.values()):
        raise ValueError(
            "TEST_SIZE leaves no training rows for at least one label. "
            "Reduce TEST_SIZE or use more data."
        )

    return SplitPlan(test_targets=test_targets, train_targets=train_targets)


def choose_test_row(remaining_total: int, remaining_test: int, rng: random.Random) -> bool:
    if remaining_test <= 0:
        return False
    if remaining_test >= remaining_total:
        return True
    return rng.randrange(remaining_total) < remaining_test


def assign_and_write(review_path: Path, stats: ScanStats, plan: SplitPlan) -> AssignmentStats:
    train_writer = BufferedJsonlWriter(
        TRAIN_OUTPUT_PATH,
        seed=RANDOM_STATE + 1_001,
        buffer_size=SHUFFLE_BUFFER_SIZE,
    )
    test_writer = BufferedJsonlWriter(
        TEST_OUTPUT_PATH,
        seed=RANDOM_STATE + 1_002,
        buffer_size=SHUFFLE_BUFFER_SIZE,
    )

    remaining_total_by_label = {
        label: int(count)
        for label, count in stats.label_counts.items()
    }
    remaining_test_by_label = {
        label: int(count)
        for label, count in plan.test_targets.items()
    }
    remaining_train_by_label = {
        label: int(count)
        for label, count in plan.train_targets.items()
    }
    test_rngs = {
        label: random.Random(RANDOM_STATE + 10_000 + label)
        for label in stats.label_counts
    }

    assignment_stats = AssignmentStats()
    eligible_rows_written = 0
    log("Starting write pass")

    try:
        for row in iter_prepared_records(review_path):
            if row is None:
                continue

            label = int(row["label"])
            should_go_to_test = choose_test_row(
                remaining_total=remaining_total_by_label[label],
                remaining_test=remaining_test_by_label[label],
                rng=test_rngs[label],
            )
            remaining_total_by_label[label] -= 1
            eligible_rows_written += 1

            if should_go_to_test:
                remaining_test_by_label[label] -= 1
                test_writer.write(row)
                assignment_stats.test_label_counts[label] += 1
            else:
                remaining_train_by_label[label] -= 1
                train_writer.write(row)
                assignment_stats.train_label_counts[label] += 1

            if eligible_rows_written % LOG_EVERY_ROWS == 0:
                log(
                    "Write progress: "
                    f"assigned {eligible_rows_written:,}/{stats.eligible_rows:,} eligible rows"
                )
    finally:
        train_writer.close()
        test_writer.close()

    if any(value != 0 for value in remaining_total_by_label.values()):
        raise RuntimeError("Split assignment finished with unassigned label totals.")
    if any(value != 0 for value in remaining_test_by_label.values()):
        raise RuntimeError("Split assignment finished with unfilled test quotas.")
    if any(value != 0 for value in remaining_train_by_label.values()):
        raise RuntimeError("Split assignment finished with unfilled train quotas.")

    log(f"Finished write pass: assigned {eligible_rows_written:,} eligible rows")
    return assignment_stats


def counter_to_dict(counter: Counter[int]) -> dict[str, int]:
    return {str(label): int(count) for label, count in sorted(counter.items())}


def positive_rate(counter: Counter[int]) -> float | None:
    total = sum(counter.values())
    if total == 0:
        return None
    return counter.get(1, 0) / total


def build_summary(
    review_path: Path,
    stats: ScanStats,
    plan: SplitPlan,
    assignment_stats: AssignmentStats,
) -> dict[str, Any]:
    return {
        "config": {
            "review_path": str(review_path),
            "output_dir": str(OUTPUT_DIR),
            "train_output_path": str(TRAIN_OUTPUT_PATH),
            "test_output_path": str(TEST_OUTPUT_PATH),
            "summary_output_path": str(SUMMARY_OUTPUT_PATH),
            "test_size": TEST_SIZE,
            "positive_threshold": POSITIVE_THRESHOLD,
            "drop_middle": DROP_MIDDLE,
            "min_review_words": MIN_REVIEW_WORDS,
            "max_rows": MAX_ROWS,
            "random_state": RANDOM_STATE,
            "shuffle_buffer_size": SHUFFLE_BUFFER_SIZE,
        },
        "scan": {
            "raw_rows_seen": int(stats.raw_rows_seen),
            "eligible_rows": int(stats.eligible_rows),
            "skipped_rows": int(stats.skipped_rows),
            "label_counts": counter_to_dict(stats.label_counts),
            "positive_rate": positive_rate(stats.label_counts),
        },
        "split_plan": {
            "test_targets": {str(label): int(count) for label, count in sorted(plan.test_targets.items())},
            "train_targets": {str(label): int(count) for label, count in sorted(plan.train_targets.items())},
        },
        "assignment": {
            "train_rows": int(sum(assignment_stats.train_label_counts.values())),
            "test_rows": int(sum(assignment_stats.test_label_counts.values())),
            "train_label_counts": counter_to_dict(assignment_stats.train_label_counts),
            "test_label_counts": counter_to_dict(assignment_stats.test_label_counts),
            "train_positive_rate": positive_rate(assignment_stats.train_label_counts),
            "test_positive_rate": positive_rate(assignment_stats.test_label_counts),
        },
        "notes": {
            "method": "Two-pass streamed, label-stratified split with buffered shuffle on write.",
            "folds": "No fold files are created by this script. Existing fold_* directories are left untouched.",
        },
    }


def write_summary(summary: dict[str, Any]) -> None:
    with SUMMARY_OUTPUT_PATH.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
        f.write("\n")


def main() -> None:
    review_path = resolve_review_path()
    ensure_output_dir(overwrite=OVERWRITE_OUTPUT)

    stats = scan_dataset(review_path)
    plan = build_split_plan(stats)
    assignment_stats = assign_and_write(review_path, stats, plan)

    summary = build_summary(review_path, stats, plan, assignment_stats)
    write_summary(summary)

    log(
        "Split complete: "
        f"train_rows={sum(assignment_stats.train_label_counts.values()):,} | "
        f"test_rows={sum(assignment_stats.test_label_counts.values()):,}"
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
