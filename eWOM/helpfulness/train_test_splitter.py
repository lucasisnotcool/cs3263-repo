from __future__ import annotations

import argparse
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

DEFAULT_REVIEW_PATH = Path("data/raw/amazon-reviews-2023/Electronics.jsonl")
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "data" / "helpfulness"
DEFAULT_TRAIN_OUTPUT_FILENAME = "train.jsonl"
DEFAULT_VAL_OUTPUT_FILENAME = "val.jsonl"
DEFAULT_TEST_OUTPUT_FILENAME = "test.jsonl"
DEFAULT_SUMMARY_OUTPUT_FILENAME = "split_summary.json"

DEFAULT_VAL_SIZE = 0.1
DEFAULT_TEST_SIZE = 0.2
DEFAULT_POSITIVE_THRESHOLD = 1
DEFAULT_DROP_MIDDLE = False
DEFAULT_MIN_REVIEW_WORDS = 0
DEFAULT_MAX_ROWS = None
DEFAULT_RANDOM_STATE = 42
DEFAULT_OVERWRITE_OUTPUT = True
DEFAULT_LOG_EVERY_ROWS = 50_000
DEFAULT_SHUFFLE_BUFFER_SIZE = 256


@dataclass
class ScanStats:
    raw_rows_seen: int = 0
    eligible_rows: int = 0
    skipped_rows: int = 0
    label_counts: Counter[int] = field(default_factory=Counter)


@dataclass
class SplitPlan:
    train_targets: dict[int, int]
    val_targets: dict[int, int]
    test_targets: dict[int, int]


@dataclass
class AssignmentStats:
    train_label_counts: Counter[int] = field(default_factory=Counter)
    val_label_counts: Counter[int] = field(default_factory=Counter)
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


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Create label-stratified train/val/test helpfulness splits from the raw Amazon Reviews file.",
    )
    parser.add_argument(
        "--review-path",
        default=str(DEFAULT_REVIEW_PATH),
        help="Path to the raw review JSONL or JSONL.GZ file.",
    )
    parser.add_argument(
        "--output-dir",
        default=str(DEFAULT_OUTPUT_DIR),
        help="Directory where train.jsonl, val.jsonl, test.jsonl, and split_summary.json will be written.",
    )
    parser.add_argument(
        "--val-size",
        type=float,
        default=DEFAULT_VAL_SIZE,
        help="Validation split size as a fraction of eligible rows.",
    )
    parser.add_argument(
        "--test-size",
        type=float,
        default=DEFAULT_TEST_SIZE,
        help="Test split size as a fraction of eligible rows.",
    )
    parser.add_argument(
        "--positive-threshold",
        type=int,
        default=DEFAULT_POSITIVE_THRESHOLD,
        help="Minimum helpful-vote count required to label a review as helpful.",
    )
    parser.add_argument(
        "--drop-middle",
        action=argparse.BooleanOptionalAction,
        default=DEFAULT_DROP_MIDDLE,
        help="Whether to drop rows with helpful votes strictly between 0 and the positive threshold.",
    )
    parser.add_argument(
        "--min-review-words",
        type=int,
        default=DEFAULT_MIN_REVIEW_WORDS,
        help="Minimum review word count required for a row to remain eligible.",
    )
    parser.add_argument(
        "--max-rows",
        type=int,
        default=DEFAULT_MAX_ROWS,
        help="Optional cap on how many raw rows to scan before splitting.",
    )
    parser.add_argument(
        "--random-state",
        type=int,
        default=DEFAULT_RANDOM_STATE,
        help="Random seed used for split assignment and buffered shuffling.",
    )
    parser.add_argument(
        "--shuffle-buffer-size",
        type=int,
        default=DEFAULT_SHUFFLE_BUFFER_SIZE,
        help="Buffered shuffle size used while writing the split files.",
    )
    parser.add_argument(
        "--log-every-rows",
        type=int,
        default=DEFAULT_LOG_EVERY_ROWS,
        help="Progress logging interval during scan and write passes.",
    )
    parser.add_argument(
        "--overwrite-output",
        action=argparse.BooleanOptionalAction,
        default=DEFAULT_OVERWRITE_OUTPUT,
        help="Whether to overwrite existing split files in the output directory.",
    )
    return parser


def log(message: str) -> None:
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{timestamp}] {message}", flush=True)


def resolve_path(path: str | Path) -> Path:
    path = Path(path)
    if path.is_absolute():
        return path
    return PROJECT_ROOT / path


def build_output_paths(output_dir: Path) -> dict[str, Path]:
    return {
        "train": output_dir / DEFAULT_TRAIN_OUTPUT_FILENAME,
        "val": output_dir / DEFAULT_VAL_OUTPUT_FILENAME,
        "test": output_dir / DEFAULT_TEST_OUTPUT_FILENAME,
        "summary": output_dir / DEFAULT_SUMMARY_OUTPUT_FILENAME,
    }


def resolve_review_path(review_path: Path, output_dir: Path) -> Path:
    configured_path = resolve_path(review_path)
    if configured_path.exists():
        return configured_path

    candidates = sorted(
        path
        for pattern in ("*.jsonl", "*.jsonl.gz")
        for path in DATA_DIR.rglob(pattern)
        if output_dir not in path.parents
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
        "Pass --review-path with the correct source JSONL file."
    )


def ensure_output_dir(output_dir: Path, generated_paths: list[Path], overwrite: bool) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    existing_paths = [path for path in generated_paths if path.exists()]
    if existing_paths and not overwrite:
        existing_str = ", ".join(str(path) for path in existing_paths)
        raise FileExistsError(
            "Output files already exist. Re-run with --overwrite-output to replace them: "
            f"{existing_str}"
        )

    for path in existing_paths:
        path.unlink()

    existing_fold_dirs = sorted(output_dir.glob("fold_*"))
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


def prepare_record(
    row: dict[str, Any],
    *,
    positive_threshold: int,
    drop_middle: bool,
    min_review_words: int,
) -> dict[str, Any] | None:
    title = str(row.get("title") or "")
    text = str(row.get("text") or "")
    helpful_votes = coerce_int(row.get("helpful_votes", 0), default=0)
    review_len_words = len(text.split())

    if drop_middle and 0 < helpful_votes < positive_threshold:
        return None
    if review_len_words < min_review_words:
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
        "label": int(helpful_votes >= positive_threshold),
        "review_len_words": review_len_words,
    }


def iter_prepared_records(
    review_path: Path,
    *,
    max_rows: int | None,
    positive_threshold: int,
    drop_middle: bool,
    min_review_words: int,
) -> Iterator[dict[str, Any] | None]:
    loader = AmazonReviewsLoader(review_path, max_rows=max_rows)
    for row in loader.iter_rows():
        yield prepare_record(
            row,
            positive_threshold=positive_threshold,
            drop_middle=drop_middle,
            min_review_words=min_review_words,
        )


def scan_dataset(
    review_path: Path,
    *,
    max_rows: int | None,
    positive_threshold: int,
    drop_middle: bool,
    min_review_words: int,
    log_every_rows: int,
) -> ScanStats:
    stats = ScanStats()
    log(f"Scanning dataset from {review_path}")

    for row in iter_prepared_records(
        review_path,
        max_rows=max_rows,
        positive_threshold=positive_threshold,
        drop_middle=drop_middle,
        min_review_words=min_review_words,
    ):
        stats.raw_rows_seen += 1
        if row is None:
            stats.skipped_rows += 1
        else:
            label = int(row["label"])
            stats.eligible_rows += 1
            stats.label_counts[label] += 1

        if log_every_rows > 0 and stats.raw_rows_seen % log_every_rows == 0:
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


def allocate_targets(label_counts: Counter[int] | dict[int, int], split_size: float) -> dict[int, int]:
    label_counts = Counter({int(label): int(count) for label, count in label_counts.items()})
    if split_size <= 0:
        return {label: 0 for label in label_counts}

    total_rows = sum(label_counts.values())
    total_split_rows = int(round(total_rows * split_size))

    exact_targets = {
        label: label_counts[label] * split_size
        for label in label_counts
    }
    targets = {
        label: int(math.floor(exact_targets[label]))
        for label in label_counts
    }

    remaining = total_split_rows - sum(targets.values())
    if remaining > 0:
        ranked_labels = sorted(
            label_counts,
            key=lambda label: (exact_targets[label] - targets[label], label_counts[label]),
            reverse=True,
        )
        for label in ranked_labels[:remaining]:
            targets[label] += 1

    return targets


def build_split_plan(stats: ScanStats, *, val_size: float, test_size: float) -> SplitPlan:
    if not 0 <= val_size < 1:
        raise ValueError("val_size must be between 0 and 1.")
    if not 0 <= test_size < 1:
        raise ValueError("test_size must be between 0 and 1.")
    if val_size + test_size >= 1:
        raise ValueError("val_size + test_size must be less than 1.")
    if len(stats.label_counts) < 2:
        raise ValueError("Splitting requires at least two label classes.")

    test_targets = allocate_targets(stats.label_counts, test_size)
    remaining_after_test = {
        label: stats.label_counts[label] - test_targets[label]
        for label in stats.label_counts
    }
    val_relative_size = 0.0 if val_size == 0 else val_size / (1.0 - test_size)
    val_targets = allocate_targets(remaining_after_test, val_relative_size)
    train_targets = {
        label: remaining_after_test[label] - val_targets[label]
        for label in stats.label_counts
    }

    if any(count <= 0 for count in train_targets.values()):
        raise ValueError(
            "Configured split sizes leave no training rows for at least one label. "
            "Reduce val_size/test_size or use more data."
        )

    return SplitPlan(
        train_targets=train_targets,
        val_targets=val_targets,
        test_targets=test_targets,
    )


def choose_split_row(
    remaining_total: int,
    remaining_split_targets: dict[str, int],
    rng: random.Random,
) -> str:
    reserved_targets = {
        split_name: int(count)
        for split_name, count in remaining_split_targets.items()
        if count > 0
    }
    reserved_total = sum(reserved_targets.values())
    if reserved_total <= 0:
        return "train"

    draw = rng.randrange(remaining_total)
    if draw >= reserved_total:
        return "train"

    cumulative = 0
    for split_name in ("test", "val"):
        cumulative += reserved_targets.get(split_name, 0)
        if draw < cumulative:
            return split_name
    return "train"


def assign_and_write(
    review_path: Path,
    stats: ScanStats,
    plan: SplitPlan,
    *,
    output_paths: dict[str, Path],
    max_rows: int | None,
    positive_threshold: int,
    drop_middle: bool,
    min_review_words: int,
    random_state: int,
    shuffle_buffer_size: int,
    log_every_rows: int,
) -> AssignmentStats:
    writers = {
        "train": BufferedJsonlWriter(
            output_paths["train"],
            seed=random_state + 1_001,
            buffer_size=shuffle_buffer_size,
        ),
        "val": BufferedJsonlWriter(
            output_paths["val"],
            seed=random_state + 1_002,
            buffer_size=shuffle_buffer_size,
        ),
        "test": BufferedJsonlWriter(
            output_paths["test"],
            seed=random_state + 1_003,
            buffer_size=shuffle_buffer_size,
        ),
    }

    remaining_total_by_label = {
        label: int(count)
        for label, count in stats.label_counts.items()
    }
    remaining_val_by_label = {
        label: int(count)
        for label, count in plan.val_targets.items()
    }
    remaining_test_by_label = {
        label: int(count)
        for label, count in plan.test_targets.items()
    }
    remaining_train_by_label = {
        label: int(count)
        for label, count in plan.train_targets.items()
    }
    split_rngs = {
        label: random.Random(random_state + 10_000 + label)
        for label in stats.label_counts
    }

    assignment_stats = AssignmentStats()
    eligible_rows_written = 0
    log("Starting write pass")

    try:
        for row in iter_prepared_records(
            review_path,
            max_rows=max_rows,
            positive_threshold=positive_threshold,
            drop_middle=drop_middle,
            min_review_words=min_review_words,
        ):
            if row is None:
                continue

            label = int(row["label"])
            chosen_split = choose_split_row(
                remaining_total=remaining_total_by_label[label],
                remaining_split_targets={
                    "test": remaining_test_by_label[label],
                    "val": remaining_val_by_label[label],
                },
                rng=split_rngs[label],
            )
            remaining_total_by_label[label] -= 1
            eligible_rows_written += 1

            if chosen_split == "test":
                remaining_test_by_label[label] -= 1
                assignment_stats.test_label_counts[label] += 1
            elif chosen_split == "val":
                remaining_val_by_label[label] -= 1
                assignment_stats.val_label_counts[label] += 1
            else:
                remaining_train_by_label[label] -= 1
                assignment_stats.train_label_counts[label] += 1

            writers[chosen_split].write(row)

            if log_every_rows > 0 and eligible_rows_written % log_every_rows == 0:
                log(
                    "Write progress: "
                    f"assigned {eligible_rows_written:,}/{stats.eligible_rows:,} eligible rows"
                )
    finally:
        for writer in writers.values():
            writer.close()

    if any(value != 0 for value in remaining_total_by_label.values()):
        raise RuntimeError("Split assignment finished with unassigned label totals.")
    if any(value != 0 for value in remaining_test_by_label.values()):
        raise RuntimeError("Split assignment finished with unfilled test quotas.")
    if any(value != 0 for value in remaining_val_by_label.values()):
        raise RuntimeError("Split assignment finished with unfilled val quotas.")
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
    output_dir: Path,
    output_paths: dict[str, Path],
    stats: ScanStats,
    plan: SplitPlan,
    assignment_stats: AssignmentStats,
    *,
    val_size: float,
    test_size: float,
    positive_threshold: int,
    drop_middle: bool,
    min_review_words: int,
    max_rows: int | None,
    random_state: int,
    shuffle_buffer_size: int,
) -> dict[str, Any]:
    return {
        "config": {
            "review_path": str(review_path),
            "output_dir": str(output_dir),
            "train_output_path": str(output_paths["train"]),
            "val_output_path": str(output_paths["val"]),
            "test_output_path": str(output_paths["test"]),
            "summary_output_path": str(output_paths["summary"]),
            "val_size": val_size,
            "test_size": test_size,
            "positive_threshold": positive_threshold,
            "drop_middle": drop_middle,
            "min_review_words": min_review_words,
            "max_rows": max_rows,
            "random_state": random_state,
            "shuffle_buffer_size": shuffle_buffer_size,
        },
        "scan": {
            "raw_rows_seen": int(stats.raw_rows_seen),
            "eligible_rows": int(stats.eligible_rows),
            "skipped_rows": int(stats.skipped_rows),
            "label_counts": counter_to_dict(stats.label_counts),
            "positive_rate": positive_rate(stats.label_counts),
        },
        "split_plan": {
            "train_targets": {
                str(label): int(count) for label, count in sorted(plan.train_targets.items())
            },
            "val_targets": {
                str(label): int(count) for label, count in sorted(plan.val_targets.items())
            },
            "test_targets": {
                str(label): int(count) for label, count in sorted(plan.test_targets.items())
            },
        },
        "assignment": {
            "train_rows": int(sum(assignment_stats.train_label_counts.values())),
            "val_rows": int(sum(assignment_stats.val_label_counts.values())),
            "test_rows": int(sum(assignment_stats.test_label_counts.values())),
            "train_label_counts": counter_to_dict(assignment_stats.train_label_counts),
            "val_label_counts": counter_to_dict(assignment_stats.val_label_counts),
            "test_label_counts": counter_to_dict(assignment_stats.test_label_counts),
            "train_positive_rate": positive_rate(assignment_stats.train_label_counts),
            "val_positive_rate": positive_rate(assignment_stats.val_label_counts),
            "test_positive_rate": positive_rate(assignment_stats.test_label_counts),
        },
        "notes": {
            "method": "Two-pass streamed, label-stratified train/val/test split with buffered shuffle on write.",
            "folds": "No fold files are created by this script. Existing fold_* directories are left untouched.",
        },
    }


def write_summary(summary_path: Path, summary: dict[str, Any]) -> None:
    with summary_path.open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)
        handle.write("\n")


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    output_dir = resolve_path(args.output_dir)
    output_paths = build_output_paths(output_dir)
    review_path = resolve_review_path(Path(args.review_path), output_dir)

    ensure_output_dir(
        output_dir,
        generated_paths=list(output_paths.values()),
        overwrite=args.overwrite_output,
    )

    stats = scan_dataset(
        review_path,
        max_rows=args.max_rows,
        positive_threshold=args.positive_threshold,
        drop_middle=args.drop_middle,
        min_review_words=args.min_review_words,
        log_every_rows=args.log_every_rows,
    )
    plan = build_split_plan(
        stats,
        val_size=args.val_size,
        test_size=args.test_size,
    )
    assignment_stats = assign_and_write(
        review_path,
        stats,
        plan,
        output_paths=output_paths,
        max_rows=args.max_rows,
        positive_threshold=args.positive_threshold,
        drop_middle=args.drop_middle,
        min_review_words=args.min_review_words,
        random_state=args.random_state,
        shuffle_buffer_size=args.shuffle_buffer_size,
        log_every_rows=args.log_every_rows,
    )

    summary = build_summary(
        review_path,
        output_dir,
        output_paths,
        stats,
        plan,
        assignment_stats,
        val_size=args.val_size,
        test_size=args.test_size,
        positive_threshold=args.positive_threshold,
        drop_middle=args.drop_middle,
        min_review_words=args.min_review_words,
        max_rows=args.max_rows,
        random_state=args.random_state,
        shuffle_buffer_size=args.shuffle_buffer_size,
    )
    write_summary(output_paths["summary"], summary)

    log(
        "Split complete: "
        f"train_rows={sum(assignment_stats.train_label_counts.values()):,} | "
        f"val_rows={sum(assignment_stats.val_label_counts.values()):,} | "
        f"test_rows={sum(assignment_stats.test_label_counts.values()):,}"
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
