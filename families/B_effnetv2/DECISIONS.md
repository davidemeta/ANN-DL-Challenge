# Family B - EfficientNetV2-S: study and decisions

Notebook: `B_effnetv2_kaggle.ipynb` (Kaggle, inputs `an2dl2627-data` + `batchsize4-shared-split`).
Data, folds and shifted validation are the team's shared ones (Elia, `eda/elia/`, branch
`elia/exploration`); nothing here recomputes them.

## 1. What the data is

- 4,328 train + 14,736 test images, 256x256 RGB, 10 galaxy classes. After removing the 1,328 copies
  (996 brightened with swapped labels 2<->3 / 5<->6, 332 with a pasted alien and a random label),
  the empty and the broken image: **2,995 clean images**. 3,000 non-copy train images + 14,736 test
  = 17,736, the size of **Galaxy10 DECaLS**: same image source and the same 10 classes.
- Clean class shares (%): 0 disturbed 6.1, 1 merging 10.5, 2 round smooth 14.9, 3 in-between 11.5,
  **4 cigar 1.9**, 5 barred spiral 11.6, 6 tight spiral 10.3, 7 loose spiral 14.8,
  8 edge-on no bulge 8.0, 9 edge-on bulge 10.6. Weighted F1 -> no class re-weighting.
- 126 clean images (4.2%) have a suspicious label (`review`, linear-probe disagreement), mostly
  classes 0 and 7. Kept; label smoothing 0.1 is the only noise defence.

## 2. How the test differs (Elia's notebooks 00/02 + checks done here)

| difference | evidence | in the shifted validation? |
|---|---|---|
| **hue rotation** about +-120 deg on ~28% of the test | 99.7% of tinted cores restored by a hue rotation; fit of the full hue histogram (Elia 02) | yes (28%, +-100-140 deg) |
| **mild blur** sigma 0.6-1.0 on ~20% | fit of the sharpness distribution (Elia 02) | yes (20%) |
| galaxies look ~6% larger | **explained by the blur**: the same measure on the shifted copy of the clean set reproduces the test quantiles | - (no real size shift: no zoom augmentation) |
| **more off-centre targets** (75th pct offset 8.2 px vs 3.8; 95th 20 vs 16) | not reproduced by the shifted copy (3.9 / 16.7): a real difference | **no** |
| **wider sky level / brightness range** (sky 5th-95th pct 14-34 vs 20-28; mean 5th pct 27 vs 35) | not reproduced by the shifted copy | **no** |
| slightly grainier sky on a few % of the test | upper tail of sky noise 8.3 vs 7.7 | no |

Colour carries some class information (core colour alone: weighted F1 0.22 vs 0.04 for the
majority class; shape features alone 0.39, together 0.47), so colour cannot simply be thrown away.

A hue rotation of exactly +-120 deg **is a cyclic shift of the RGB channels** (checked: max pixel
difference 0.0; +120 = roll channels by +1).

Reference result (Elia, ConvNeXt-Tiny 256, flips/90-degree rotations only, 20 epochs): OOF weighted
F1 **0.844 clean / 0.820 shifted**, Kaggle public **0.832**. The public score falls between the two:
the shifted validation is a slightly pessimistic but good proxy. Weakest classes: 0 disturbed (0.47),
7 loose spiral (0.75), 6 tight spiral (0.77), 4 cigar (0.77).

## 3. Literature (context only: general-purpose weights, no external labels)

- Galaxy10 DECaLS papers report ~87-90% accuracy for fine-tuned EfficientNet / ResNet and up to
  95.5% for D16-equivariant CNNs, but they train on ~14k labelled images; with 2,995 these numbers
  are not a target.
- Rotation equivariance helps on this dataset (D16 > D4 > plain CNN): any-angle rotation in
  training + D4 TTA is the cheap version of it.
- CNNs are not robust to hue shifts; hue augmentation recovers most of it without hurting clean data.

## 4. Decisions

- **Backbone** `tf_efficientnetv2_s.in21k_ft_in1k` (pure CNN, MBConv + SE): different design from the
  ConvNeXt families -> different errors for the team ensemble. Native 256 px.
- **Recipe**: AdamW lr 3e-4, wd 0.05, warmup 1 epoch + cosine, 30 epochs, batch 32, label smoothing
  0.1, drop 0.2 / drop-path 0.1, zero-initialised head, EMA 0.998 with warmup, last-epoch weights
  (no best-epoch picking), fp16 on T4.
- **Augmentation chosen by measurement, not by taste** (stage 1 of the notebook, fold 0): start from
  any-angle rotation + flip, then add one step at a time and keep it only if it passes the team rule:
  1. hue rotation (p 0.5, full circle): keep if shifted F1 +>=0.01 and clean -<=0.01;
  2. Gaussian blur (p 0.3, sigma 0.3-1.3): same rule;
  3. photometric (brightness, contrast, sky offset, light noise) + translation +-4%: these target
     shifts that the shifted validation does **not** contain, so it cannot reward them; keep if they
     cost <=0.01 on both validations ("no harm").
  Differences below ~0.01 on one fold are noise (Elia's folds spread 0.02 clean / 0.03 shifted).
- **Hue-view TTA**: average over the image and its two channel shifts (= +-120 deg hue). It makes the
  prediction exactly invariant to a +-120 deg rotation, at the price of part of the colour cue.
  Per-view logits are saved and three weightings are compared on OOF. Caveat: the shifted validation
  uses the same mechanism, so it flatters this TTA. Both submissions are written (plain D4 and the
  best hue-TTA); the **Kaggle public score decides**.
- **Outputs for the team**: `oof.csv` (Elia's format: `p{c}_clean`, `p{c}_shifted`), `test_probs.npy`
  (folds, 14736, 10), raw per-view logits, `experiments_log.csv` rows in Elia's columns.

## 5. Run 1 results (2026-11-27, T4, 125 min)

Notebook `B_effnetv2_kaggle.ipynb`, logs in `experiments_log.csv` / `ablation_run1.csv`.

| fold-0 step | F1 clean | F1 shifted | decision |
|---|---|---|---|
| control (rotation + flip) | 0.826 | 0.796 | - |
| + hue rotation | 0.828 | 0.807 | kept (+0.011) |
| + blur | 0.853 | 0.847 | kept (+0.040) |
| + photometric + translation | 0.845 | 0.838 | kept as "no harm" (-0.0095) |

5-fold OOF (D4 TTA): **0.827 clean / 0.8285 shifted** (Elia ConvNeXt-T: 0.844 / 0.820). The shift is
handled (shifted ≈ clean); clean is ~0.017 below the ConvNeXt baseline.

Error analysis on the OOF:
- validation F1 peaks at epochs 18-22, then drifts down -> 30 epochs is too many;
- hue-view TTA: +0.002 for 3x test time -> not worth it once hue augmentation is in training;
- main confusions: tight <-> loose spiral (85), disturbed <-> loose (56), barred <-> loose (52),
  disturbed -> in-between smooth (22): fine morphology;
- class 0 is under-predicted (4.5% predicted vs 6.1% true), but nested per-class multipliers gain only
  +0.002 -> no post-processing;
- confident learning on the OOF (p(label) < 0.05, other class > 0.7): **128 images (4.3%)**, 40 of
  them *disturbed* (22% of the class), 31 *loose spiral*; a visual check shows mostly real label errors
  (smooth ellipticals labelled *disturbed*, an edge-on labelled *loose spiral*). List:
  `label_noise_run1.csv`. The OOF probabilities come from models that never saw the image; using them
  to drop training images leaks fold labels only at second order (standard practice).

## 6. Run 2 (`B_effnetv2_run2_kaggle.ipynb`)

20 epochs, no hue-view TTA, run-1 recipe as control. Fold-0 greedy study with the team rule, then the
4 other folds:
1. **resolution 320** (upsampled from 256): EfficientNetV2-S is pretrained at 300-384 px and the
   remaining errors are about small structures (arms, bars);
2. **drop the 128 likely-mislabelled images from training** (validation untouched);
3. **MixUp 0.2**: regularisation against late-epoch overfitting and label noise.

Speed: in run 1, reading the PNGs took 17 of the 125 minutes (468 s for the clean set + shifted copy,
572 s for the test, ~13-26 images/s: `/kaggle/input` is latency-bound). Run 2 reads with 32 threads,
reads each clean image once and builds the shifted copy from it (pixel-identical to `C.load_shifted`,
asserted in the notebook).

Not in run 2, on purpose:
- **pseudo-labelling the test set**: the playbook (Part 0, §0.5b) makes it conditional on a forum
  answer about unlabelled test use, and schedules it at the end (Dec 4-5) with per-fold
  cross-family teachers, calibrated thresholds and soft labels;
- **ensemble with Elia's ConvNeXt** (different errors, likely the largest gain): needs his
  `oof.csv` / `test_probs.npy`, done in the ensemble stage;
- rotation-by-45° TTA: same reasoning as hue TTA, small gains once rotation is in training.

Later: EfficientNetV2-M, two-pass cleaning with the cross-family OOF, de-tinting the test images.

Sources: Elia's notebooks `eda/elia/00-03` and `families/elia/10_baseline.ipynb`;
[Scaling Laws for Galaxy Images](https://arxiv.org/abs/2404.02973);
[E(2) Equivariant Neural Networks for Robust Galaxy Morphology Classification](https://arxiv.org/abs/2311.01500);
[Learning Color Equivariant Representations](https://arxiv.org/abs/2406.09588).
