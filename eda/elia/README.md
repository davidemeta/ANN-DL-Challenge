# eda/elia – data exploration, cleaning and validation setup

Run the notebooks **in this order** (each one reads the files written by the previous ones):

| notebook | what it does | writes |
|---|---|---|
| `00_eda.ipynb` | exploratory data analysis: classes, pixel statistics, test shift, aliens, embeddings, UMAP, adversarial validation, linear probe | `outputs/meta.csv`, `outputs/pixel_stats.csv`, `outputs/emb_dinov2_s.npy`, `outputs/emb_dinov2_s_ids.csv`, `outputs/umap2d.npy` |
| `01_clean_dataset.ipynb` | finds the brightened / alien copies (star pattern), resolves ambiguous groups, flags artefacts and suspicious labels | `outputs/copies.csv`, **`splits/clean_train.csv`** |
| `02_test_shift_study.ipynb` | how the test differs from the clean training set: colour (hue rotation of about ±120° on ~28% of the images) and blur (~20%, sigma 0.6–1.0); parameters of the shifted validation (agreed by the team) | `outputs/core_rgb.csv` |
| `03_validation_setup.ipynb` | fixed 5 folds on the 2,995 clean images, shifted-validation plan (hue ±100–140° on 28%, blur sigma 0.6–1.0 on 20%), check against the test, protocol test | **`splits/folds.csv`**, **`splits/val_shift_plan.csv`**, `experiments_log.csv` |

`common.py` holds the shared helpers (paths, folds, shifted validation, evaluation). The data folder is found automatically (`data/` at the repo root).

- `outputs/` = caches and large files, ignored by git.
- `splits/` and `experiments_log.csv` = small files that define the clean set and the validation: committed.
- `03_validation_setup` was run **once**: do not re-run it, everybody reads the same `splits/` files.

## Key findings
- 996 training images are brightened copies of other training images with a swapped label (2↔3, 5↔6) and 332 are copies with a pasted alien and a random label. The clean training set has **2,995 images**, identical to the independent cleaning proposal of Davide (`eda/01_audit_davide.py`).
- The test differs from the clean training set by a **hue rotation** of about ±120° (~28% of the images) and a **mild blur** (~20%). The shifted validation reproduces both.

## For the team: how to use the shared split

```python
import sys; sys.path.insert(0, "eda/elia")   # path from the repo root
import pandas as pd
import common as C

folds = pd.read_csv(C.SPLITS / "folds.csv")                      # 2,995 clean images, column `fold` = 0..4
plan = pd.read_csv(C.SPLITS / "val_shift_plan.csv").set_index("id")

train_df, val_df = C.get_split(folds, k=0)                       # never recompute the split
img_clean = C.load_image(val_df.id[0])                           # clean validation image
img_shifted = C.load_shifted(val_df.id[0], plan)                 # same image, shifted like the test

res_clean = C.evaluate(val_df.label, preds_clean)                # weighted F1, per-class F1, confusion matrix
res_shifted = C.evaluate(val_df.label, preds_shifted)
```

Rules:
1. read the split from `splits/folds.csv`; **never** recompute it;
2. **never filter the validation**: experiments only change the training set (e.g. `train_df[~train_df.review]`);
3. evaluate every model on **both** validations and log the result;
4. **selection rule:** the shifted validation decides (improvement ≥ ~0.01); the clean one must not drop by more than ~0.01; the F1 of class 4 must not collapse;
5. explore on fold 0; confirm important decisions on all 5 folds and with a Kaggle submission;
6. for 5-fold runs, save the **out-of-fold** probabilities (2,995 × 10) so that models of different team members can be compared and combined.
