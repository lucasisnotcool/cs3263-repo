# Experiment Notes

## Run Notes
- Dataset discovered at: `/Users/lohzh/Desktop/cs3263-repo/data/raw/amazon-product-data/dataset/train.csv`
- Initial full-run plan was reduced to sampled execution for faster iteration.
- Current run used `SAMPLE_FRAC = 0.01` (~1% sample).

## Issues Encountered
1. Dataset path mismatch when notebook executed from subfolder.
   - Fix: notebook now searches current directory + parent directories.
2. Kaggle credential path mismatch between repo-local `.kaggle/` and `~/.kaggle/`.
   - Fix: dataset setup notebook supports both.
3. `plotly` inline rendering error (`nbformat>=4.2.0` missing).
   - Fix: install cell includes `nbformat` and `ipykernel`; plotting cell has browser fallback.
4. Original vectorisation run only processed `10,000` rows.
   - Cause: `MAX_ROWS = 10000`.
   - Fix: changed to `MAX_ROWS = None` and added configurable `SAMPLE_FRAC`.

## Operational Guidance
- For quicker tests, reduce `SAMPLE_FRAC` (e.g., `0.001`).
- For larger-scale runs, increase `CHUNK_SIZE` only if memory allows.
- Keep `RANDOM_STATE` fixed when comparing experiment variants.
