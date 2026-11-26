"""Shared helpers used by every notebook (paths, folds, shifted validation, evaluation).

The rules live in ONE place: if something changes here, it changes for every
experiment in the same way.
"""
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image, ImageFilter
from sklearn.metrics import confusion_matrix, f1_score
from sklearn.model_selection import StratifiedKFold

# --------------------------------------------------------------------------- #
# Paths: works on the Mac (../an2dl2627) and in the repo on the PC (data/)     #
# --------------------------------------------------------------------------- #
HERE = Path(__file__).resolve().parent
for _cand in [HERE.parent / "an2dl2627", HERE / "data", HERE.parent.parent / "data"]:
    if (_cand / "train.csv").exists():
        DATA = _cand
        break
else:
    raise FileNotFoundError("data folder not found (expected train.csv in ../an2dl2627, ./data or ../../data)")
OUT = HERE / "outputs"        # caches and large files (ignored by git)
SPLITS = HERE / "splits"      # small files to commit (clean train set, folds, shift plan)
OUT.mkdir(exist_ok=True)
SPLITS.mkdir(exist_ok=True)

CLASSES = {
    0: "disturbed", 1: "merging", 2: "round smooth", 3: "in-between smooth",
    4: "cigar", 5: "barred spiral", 6: "tight spiral", 7: "loose spiral",
    8: "edge-on no bulge", 9: "edge-on bulge",
}
N_CLASSES = 10

# images with no usable content (found in the EDA): removed from everything
EMPTY_IDS = ["train_03848"]


def load_image(img_id):
    """PIL RGB image from its id ('train_00012' or 'test_00345')."""
    split = img_id.split("_")[0]
    return Image.open(DATA / split / f"{img_id}.png").convert("RGB")


# --------------------------------------------------------------------------- #
# 1. Fixed folds (on the agreed clean training set)                           #
# --------------------------------------------------------------------------- #
def make_folds(clean, n_splits=5, seed=42):
    """Assign a fold (0..n_splits-1) to every image of the clean training set.

    clean: splits/clean_train.csv (notebook 01). Only rows with keep=True are used.
    Copies were removed during cleaning, so every image is a distinct galaxy and a
    plain StratifiedKFold (same class proportions in every fold) is enough.
    """
    keep = clean[clean.keep].reset_index(drop=True).copy()
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    keep["fold"] = -1
    for k, (_, val_idx) in enumerate(cv.split(keep, keep.label)):
        keep.loc[val_idx, "fold"] = k
    return keep[["id", "label", "fold", "review", "artifact"]]


def get_split(folds, k):
    """Training and validation images for round k.

    train: every image outside fold k (experiments may filter it further,
           e.g. train[~train.review]).
    val:   every image of fold k. It is ALWAYS the same for every experiment:
           never filter it by hand.
    """
    return folds[folds.fold != k].reset_index(drop=True), folds[folds.fold == k].reset_index(drop=True)


# --------------------------------------------------------------------------- #
# 2. Shifted validation                                                       #
# --------------------------------------------------------------------------- #
# Proportions and strength measured on the test set (notebook 02_validation_setup):
#  - ~20% of the images are blurred, with a MILD blur (sigma 0.5-0.9);
#  - ~20% have altered colours: the model that best reproduces the test is a
#    different gain on each R, G, B channel (e.g. lower green -> pink galaxy).
P_BLUR, P_COLOR = 0.20, 0.20
BLUR_SIGMA = (0.5, 0.9)
CHANNEL_GAIN = (0.5, 1.0)


def make_shift_plan(ids, seed=1234):
    """Decide ONCE AND FOR ALL which alteration each validation image receives.

    Returns a DataFrame (id, blur_sigma, gain_r, gain_g, gain_b):
    blur_sigma = 0 and gains = 1 mean "no alteration".
    Saved to disk, it guarantees that every experiment sees the same shifted validation.
    The two alterations are drawn independently (some images get both).
    """
    rng = np.random.default_rng(seed)
    n = len(ids)
    blur = np.where(rng.random(n) < P_BLUR, rng.uniform(*BLUR_SIGMA, n), 0.0)
    color_on = rng.random(n) < P_COLOR
    gains = rng.uniform(*CHANNEL_GAIN, (n, 3))
    gains = gains / gains.max(1, keepdims=True)          # the strongest channel stays at 1
    gains[~color_on] = 1.0
    return pd.DataFrame({"id": list(ids), "blur_sigma": blur.round(3),
                         "gain_r": gains[:, 0].round(3), "gain_g": gains[:, 1].round(3), "gain_b": gains[:, 2].round(3)})


def apply_shift(img, blur_sigma=0.0, gains=(1.0, 1.0, 1.0)):
    """Apply the alteration to a PIL image: colour first, then blur.

    Note: training augmentations should use torchvision transforms with different
    parameters/forms, so that we do not "study the exam questions".
    """
    if tuple(gains) != (1.0, 1.0, 1.0):
        a = np.asarray(img, np.float32) * np.asarray(gains, np.float32)
        img = Image.fromarray(np.clip(a, 0, 255).astype(np.uint8))
    if blur_sigma > 0:
        img = img.filter(ImageFilter.GaussianBlur(float(blur_sigma)))   # float(): PIL rejects np.float64
    return img


def load_shifted(img_id, plan):
    """Shifted version of a validation image, following the saved plan (indexed by id)."""
    r = plan.loc[img_id]
    return apply_shift(load_image(img_id), r.blur_sigma, (r.gain_r, r.gain_g, r.gain_b))


# --------------------------------------------------------------------------- #
# 3. Evaluation                                                               #
# --------------------------------------------------------------------------- #
def evaluate(y_true, y_pred):
    """Competition metric (weighted F1) + per-class F1 + confusion matrix."""
    labels = list(range(N_CLASSES))
    return {
        "f1_weighted": f1_score(y_true, y_pred, average="weighted"),
        "f1_per_class": pd.Series(f1_score(y_true, y_pred, average=None, labels=labels, zero_division=0),
                                  index=[f"{c} {CLASSES[c]}" for c in labels]),
        "confusion": confusion_matrix(y_true, y_pred, labels=labels),
    }


def summary_row(name, clean, shifted):
    """One row for the experiment log."""
    return {"experiment": name,
            "f1_clean": round(clean["f1_weighted"], 4),
            "f1_shifted": round(shifted["f1_weighted"], 4),
            "f1_cigar_clean": round(clean["f1_per_class"].iloc[4], 4),
            "f1_cigar_shifted": round(shifted["f1_per_class"].iloc[4], 4)}
