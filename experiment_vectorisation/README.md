# Experiment Definition: Product Vectorisation (RAG Prep)

## Objective
Build a reproducible experiment that:
1. Loads the Amazon product dataset.
2. Creates product-level embedding text from `TITLE`, `BULLET_POINTS`, and `DESCRIPTION`.
3. Generates dense vectors suitable for retrieval (RAG).
4. Stores vectors + metadata as experiment artifacts.
5. Visualises semantic structure with 3D UMAP.

## Dataset
- Source CSV: `data/raw/amazon-product-data/dataset/train.csv`
- Total rows in CSV: `2,249,698`

## Current Scope
- Sampling mode: enabled
- Sample fraction: `1%` (`SAMPLE_FRAC = 0.01`)
- Target embedded rows: `~22,496`

## Embedding Setup
- Model: `sentence-transformers/all-MiniLM-L6-v2`
- Embedding dimension: `384`
- Chunked processing for scalability
- Embeddings are L2-normalized (`normalize_embeddings=True`)

## Primary Outputs
- `experiment_vectorisation/artifacts/product_embeddings.float32.memmap`
- `experiment_vectorisation/artifacts/product_metadata.csv`
- `experiment_vectorisation/artifacts/run_info.json`
- `experiment_vectorisation/artifacts/umap_3d_sample.parquet`
- `experiment_vectorisation/artifacts/umap_sample_embeddings.npy`
- `experiment_vectorisation/artifacts/umap_sample_metadata.parquet`
