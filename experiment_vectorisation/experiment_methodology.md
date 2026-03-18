# Experiment Methodology

## Notebook
- Entry point: `experiment_vectorisation/experiment_vectorisation.ipynb`

## Pipeline
1. Install required Python packages (`sentence-transformers`, `umap-learn`, `plotly`, `pyarrow`, etc.).
2. Resolve dataset path from current/parent directories for robust execution context.
3. Read CSV schema and auto-resolve key columns:
   - ID: `PRODUCT_ID` (if present)
   - Text: `TITLE`, `BULLET_POINTS`, `DESCRIPTION`
   - Optional label/color: `PRODUCT_TYPE_ID`
4. Build `document_text` per row:
   - `Title: ... | Bullet Points: ... | Description: ...`
5. Determine sampling plan:
   - `population_rows = min(MAX_ROWS, total_rows)` when `MAX_ROWS` is set
   - exact random sample based on `SAMPLE_FRAC` with `RANDOM_STATE`
6. Encode sampled rows in streaming chunks (`CHUNK_SIZE`) with `SentenceTransformer`.
7. Persist:
   - embeddings to float32 memmap
   - row metadata to CSV
   - run metadata to `run_info.json`
8. Fit 3D UMAP on sampled embeddings and store coordinates.
9. Render interactive Plotly 3D scatter (with browser fallback if inline renderer deps are missing).

## Configuration (current defaults)
- `MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"`
- `MAX_ROWS = None`
- `SAMPLE_FRAC = 0.01`
- `CHUNK_SIZE = 10000`
- `BATCH_SIZE = 128`
- `UMAP_SAMPLE_SIZE = 50000`
- `PLOT_MAX_POINTS = 5000`
- `RANDOM_STATE = 42`

## Reproducibility Notes
- Sampling is deterministic for a fixed `RANDOM_STATE`.
- Model/version and output paths are logged in `run_info.json`.
