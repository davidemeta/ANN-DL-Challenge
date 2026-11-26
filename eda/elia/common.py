"""Shared helpers used by every notebook (paths, folds, shifted validation, evaluation).

The rules live in ONE place: if something changes here, it changes for every
experiment in the same way.
"""
import os
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image, ImageFilter
from sklearn.metrics import confusion_matrix, f1_score
from sklearn.model_selection import StratifiedKFold

# --------------------------------------------------------------------------- #
# Paths: works on the Mac (../an2dl2627), in the repo on the PC (data/ at the  #
# repo root) and on Kaggle (competition data under /kaggle/input/...)          #
# --------------------------------------------------------------------------- #
HERE = Path(__file__).resolve().parent
KAGGLE_INPUT = Path("/kaggle/input")

_cands = [HERE.parent / "an2dl2627", HERE / "data", HERE.parent.parent / "data"]
if KAGGLE_INPUT.exists():                       # on Kaggle: look for train.csv a few levels below /kaggle/input
    _cands += [p.parent for depth in ("*", "*/*", "*/*/*") for p in KAGGLE_INPUT.glob(f"{depth}/train.csv")]
for _cand in _cands:
    if (_cand / "train.csv").exists() and (_cand / "train").is_dir():
        DATA = _cand
        break
else:
    raise FileNotFoundError("data folder not found (expected train.csv + train/ in ../an2dl2627, ./data, "
                            "../../data or under /kaggle/input)")

SPLITS = HERE / "splits"      # small shared files (clean train set, folds, shift plan): read from here
# caches and large files: next to this file when it is writable (Mac / PC repo),
# otherwise in the Kaggle working directory (Kaggle datasets are read-only)
OUT = HERE / "outputs" if os.access(HERE, os.W_OK) else Path("/kaggle/working/outputs")
OUT.mkdir(parents=True, exist_ok=True)
if os.access(HERE, os.W_OK):
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
# Proportions and strength measured on the test set (notebook 02_test_shift_study):
#  - ~28% of the images have their HUE ROTATED by about +-120 degrees
#    (drawn in +-[100, 140], half in each direction): yellow galaxies become
#    purple / green, blue ones become red / green;
#  - ~20% of the images are blurred, with a MILD blur (sigma 0.6-1.0).
# The two alterations are drawn independently.
P_COLOR, HUE_DEG = 0.28, (100, 140)
P_BLUR, BLUR_SIGMA = 0.20, (0.6, 1.0)


def make_shift_plan(ids, seed=1234):
    """Decide ONCE AND FOR ALL which alteration each validation image receives.

    Returns a DataFrame (id, hue_deg, blur_sigma): hue_deg = 0 and blur_sigma = 0
    mean "no alteration". Saved to disk, it guarantees that every experiment sees
    the same shifted validation.
    """
    rng = np.random.default_rng(seed)
    n = len(ids)
    color_on = rng.random(n) < P_COLOR
    hue = rng.uniform(*HUE_DEG, n) * rng.choice([-1, 1], n)
    blur_on = rng.random(n) < P_BLUR
    sigma = rng.uniform(*BLUR_SIGMA, n)
    return pd.DataFrame({"id": list(ids),
                         "hue_deg": np.where(color_on, hue, 0.0).round(1),
                         "blur_sigma": np.where(blur_on, sigma, 0.0).round(3)})


def rotate_hue(img, deg):
    """Rotate the hue of a PIL image by deg degrees (saturation and brightness unchanged).

    PIL's HSV mode stores the hue in 0-255, so deg is converted to that scale.
    """
    h, s, v = img.convert("HSV").split()
    shift = int(round(deg / 360 * 256))
    h = h.point(lambda x: (x + shift) % 256)
    return Image.merge("HSV", (h, s, v)).convert("RGB")


def apply_shift(img, blur_sigma=0.0, hue_deg=0.0):
    """Apply the alteration to a PIL image: colour first, then blur.

    Note: training augmentations should use different implementations / random
    parameters (e.g. torchvision ColorJitter), so that the model does not learn
    the exact alteration of this validation ("studying the exam questions").
    """
    if hue_deg:
        img = rotate_hue(img, hue_deg)
    if blur_sigma > 0:
        img = img.filter(ImageFilter.GaussianBlur(float(blur_sigma)))   # float(): PIL rejects np.float64
    return img


def load_shifted(img_id, plan):
    """Shifted version of a validation image, following the saved plan (indexed by id)."""
    r = plan.loc[img_id]
    return apply_shift(load_image(img_id), r.blur_sigma, r.hue_deg)


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
