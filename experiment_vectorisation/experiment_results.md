# Experiment Results

## Status
Completed for sampled vectorisation run (`SAMPLE_FRAC = 0.01`).

## Run Summary
- Dataset: `/Users/lohzh/Desktop/cs3263-repo/data/raw/amazon-product-data/dataset/train.csv`
- Rows in CSV: `2,249,698`
- Population rows considered: `2,249,698`
- Sample fraction: `0.01`
- Rows embedded: `22,496`
- Embedding model: `sentence-transformers/all-MiniLM-L6-v2`
- Embedding dimension: `384`

## Artifacts Generated
- `experiment_vectorisation/artifacts/product_embeddings.float32.memmap` (~34.6 MB)
- `experiment_vectorisation/artifacts/product_metadata.csv` (~33.7 MB)
- `experiment_vectorisation/artifacts/umap_sample_embeddings.npy` (~34.6 MB)
- `experiment_vectorisation/artifacts/umap_sample_metadata.parquet` (~20.9 MB)
- `experiment_vectorisation/artifacts/umap_3d_sample.parquet`
- `experiment_vectorisation/artifacts/run_info.json`

## Interpretation
- The vectorisation pipeline is operational and produces reusable retrieval artifacts.
- Sampling reduced runtime while preserving enough points for UMAP structure inspection.
- Current outputs are suitable for:
  - nearest-neighbor search experiments,
  - clustering/category sanity checks,
  - downstream RAG indexing.

## Limitations
- This run is sampled, not full-corpus indexing.
- UMAP is computed on sampled embeddings for tractability; it is not a full dataset manifold.

## Next Experiments
1. Run multiple seeds (`RANDOM_STATE`) to assess stability of cluster structure.
2. Compare embedding models (e.g., larger sentence-transformers models) on retrieval quality.
3. Build FAISS or similar ANN index from `product_embeddings.float32.memmap` + metadata.
