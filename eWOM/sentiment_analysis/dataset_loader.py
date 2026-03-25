from __future__ import annotations

import json
import logging
import random
import sys
from pathlib import Path
from typing import Any

import pandas as pd


LABEL_TEXT_BY_ID = {
    0: "negative",
    1: "positive",
}
LOGGER = logging.getLogger(__name__)


LOCAL_DEPS_DIR = Path(__file__).resolve().parents[2] / ".deps"
if LOCAL_DEPS_DIR.exists():
    sys.path.append(str(LOCAL_DEPS_DIR))


class AmazonPolarityLoader:
    """Loads Amazon Polarity splits saved with Hugging Face `save_to_disk`."""

    REQUIRED_COLUMNS = ("text", "label", "label_text")

    def __init__(self, data_dir: str | Path, random_state: int = 42):
        self.data_dir = Path(data_dir)
        self.random_state = random_state
        self._dataset_dict = None

    def load_split(self, split_name: str, max_rows: int | None = None) -> pd.DataFrame:
        if max_rows is not None and max_rows <= 0:
            raise ValueError("max_rows must be a positive integer.")

        dataset_dict = self._load_dataset_dict()
        if dataset_dict is not None:
            if split_name not in dataset_dict:
                raise ValueError(
                    f"Split '{split_name}' not found under {self.data_dir}. "
                    f"Available splits: {list(dataset_dict.keys())}"
                )

            split = dataset_dict[split_name]
            missing_columns = [
                column for column in self.REQUIRED_COLUMNS if column not in split.column_names
            ]
            if missing_columns:
                raise ValueError(
                    f"Split '{split_name}' is missing required columns: {missing_columns}"
                )

            split = split.select_columns(list(self.REQUIRED_COLUMNS))
            if max_rows is not None:
                sample_rows = min(int(max_rows), len(split))
                split = split.shuffle(seed=self.random_state).select(range(sample_rows))

            df = split.to_pandas()
        else:
            df = self._load_split_with_pyarrow(split_name, max_rows=max_rows)

        df["text"] = df["text"].fillna("").astype(str)
        df["label"] = df["label"].astype(int)
        df["label_text"] = (
            df["label_text"]
            .fillna(df["label"].map(LABEL_TEXT_BY_ID))
            .astype(str)
        )
        return df.loc[:, list(self.REQUIRED_COLUMNS)]

    def load_train_test(
        self,
        max_train_rows: int | None = None,
        max_test_rows: int | None = None,
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        train_df = self.load_split("train", max_rows=max_train_rows)
        test_df = self.load_split("test", max_rows=max_test_rows)
        return train_df, test_df

    def load_dataset_info(self, split_name: str = "train") -> dict[str, Any]:
        info_path = self.data_dir / split_name / "dataset_info.json"
        if not info_path.exists():
            raise FileNotFoundError(f"Dataset metadata not found at {info_path}")
        with info_path.open("r", encoding="utf-8") as f:
            return json.load(f)

    def _load_dataset_dict(self):
        if self._dataset_dict is not None:
            return self._dataset_dict

        if not self.data_dir.exists():
            raise FileNotFoundError(f"Dataset directory not found: {self.data_dir}")

        try:
            from datasets import load_from_disk
        except ImportError:
            LOGGER.info("`datasets` is unavailable; falling back to direct Arrow loading.")
            return None

        LOGGER.info("Loading Amazon Polarity from Hugging Face disk format at %s", self.data_dir)
        self._dataset_dict = load_from_disk(str(self.data_dir))
        return self._dataset_dict

    def _load_split_with_pyarrow(
        self,
        split_name: str,
        max_rows: int | None = None,
    ) -> pd.DataFrame:
        try:
            import pyarrow as pa
            import pyarrow.ipc as ipc
        except ImportError as exc:
            raise ImportError(
                "Either `datasets` or `pyarrow` is required to load Amazon Polarity from disk."
            ) from exc

        split_dir = self.data_dir / split_name
        state_path = split_dir / "state.json"
        if not state_path.exists():
            raise FileNotFoundError(f"Split state not found at {state_path}")

        with state_path.open("r", encoding="utf-8") as f:
            state = json.load(f)

        arrow_files = [
            split_dir / file_info["filename"]
            for file_info in state.get("_data_files", [])
        ]
        if not arrow_files:
            raise FileNotFoundError(f"No Arrow shard files listed in {state_path}")

        if max_rows is None:
            LOGGER.info("Loading full '%s' split with pyarrow fallback", split_name)
            tables = []
            for arrow_file in arrow_files:
                with arrow_file.open("rb") as f:
                    reader = ipc.open_stream(f)
                    tables.append(reader.read_all().select(list(self.REQUIRED_COLUMNS)))
            table = pa.concat_tables(tables)
            return table.to_pandas()

        target_rows = int(max_rows)
        LOGGER.info(
            "Sampling %s rows from '%s' via pyarrow fallback with seed=%s",
            target_rows,
            split_name,
            self.random_state,
        )
        rng = random.Random(self.random_state)
        reservoir: list[dict[str, Any]] = []
        seen_rows = 0

        for arrow_file in arrow_files:
            with arrow_file.open("rb") as f:
                reader = ipc.open_stream(f)
                for batch in reader:
                    batch_df = batch.select(list(self.REQUIRED_COLUMNS)).to_pandas()
                    for row in batch_df.to_dict(orient="records"):
                        seen_rows += 1
                        if len(reservoir) < target_rows:
                            reservoir.append(row)
                            continue
                        replacement_idx = rng.randint(0, seen_rows - 1)
                        if replacement_idx < target_rows:
                            reservoir[replacement_idx] = row

        return pd.DataFrame(reservoir, columns=list(self.REQUIRED_COLUMNS))
