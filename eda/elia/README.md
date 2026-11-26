# families/elia – data exploration, cleaning and validation setup

Run the notebooks **in this order** (each one reads the files written by the previous ones):

| notebook | what it does | writes |
|---|---|---|
| `00_eda.ipynb` | exploratory data analysis: classes, pixel statistics, test shift, aliens, embeddings, UMAP, adversarial validation, linear probe | `outputs/meta.csv`, `outputs/pixel_stats.csv`, `outputs/emb_dinov2_s.npy`, `outputs/emb_dinov2_s_ids.csv`, `outputs/umap2d.npy` |
| `01_clean_dataset.ipynb` | finds the brightened / alien copies (star pattern), resolves ambiguous groups, flags artefacts and suspicious labels | `outputs/copies.csv`, **`splits/clean_train.csv`** |
| `02_validation_setup.ipynb` | fixed folds on the clean set, calibration of the shifted (test-like) validation, protocol test | `splits/folds.csv`, `splits/val_shift_plan.csv`, `outputs/core_chroma.csv`, `experiments_log.csv` |

`common.py` holds the shared helpers (paths, folds, shifted validation, evaluation). The data folder is found automatically (`data/` at the repo root).

- `outputs/` = caches and large files, ignored by git.
- `splits/` and `experiments_log.csv` = small files that define the clean set and the validation: **commit them**.
- `02_validation_setup` is **provisional** until the team agrees on `splits/clean_train.csv`.

Key finding: 996 training images are brightened copies of other training images with a swapped label (2↔3, 5↔6) and 332 are copies with a pasted alien and a random label. The clean training set has 2,995 images, identical to the independent cleaning proposal of Davide (`eda/01_audit_davide.py`).
