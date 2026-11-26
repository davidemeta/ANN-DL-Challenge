# EDA guide (Davide): how to read every output file

This guide covers every file that `eda/01_audit_davide.py`, `eda/02_embeddings_davide.py` and `eda/03_test_distortions_davide.py` write to `eda/outputs/davide/`: what it contains,
what is on the axes and in which unit, and what to look for. The findings, the decisions and the
conclusions are in the notebook `eda/00_eda_davide.ipynb`.

`eda/outputs/davide/` is gitignored. It is **deleted and rebuilt** on every run of the audit script
(about 2.5 minutes), except `06_embeddings/` and `07_test_distortions/`, which belong to the other two scripts. Never edit it by hand.

```bash
source .venv/bin/activate
```
```bash
python eda/01_audit_davide.py
```
```bash
python eda/02_embeddings_davide.py
```
The second script needs torch, timm and umap-learn. The first run takes about 30 minutes on a Mac (19,064 images × 8 views);
the embeddings are then cached in `eda/outputs/embeddings_cache/` and later runs take about a minute.
```bash
python eda/03_test_distortions_davide.py
```

## Contents

0. [Reading basics](#0-reading-basics)
1. [Per-image measures (glossary)](#1-per-image-measures-glossary)
2. [Categories and thresholds](#2-categories-and-thresholds)
3. [`01_overview/`](#3-01_overview)
4. [`02_class_sheets/`](#4-02_class_sheets)
5. [`03_corruptions/aliens/`](#5-03_corruptionsaliens)
6. [`03_corruptions/grainy/`](#6-03_corruptionsgrainy)
7. [`03_corruptions/duplicates/` and `lines/`](#7-03_corruptionsduplicates-and-lines)
8. [`04_test/`](#8-04_test)
9. [`05_shift/`](#9-05_shift)
10. [`csv/`](#10-csv)
10b. [`06_embeddings/`](#10b-06_embeddings)
10c. [`07_test_distortions/`](#10c-07_test_distortions)
11. [`images/`](#11-images)
12. [The notebook](#12-the-notebook)
13. [Suggested reading order](#13-suggested-reading-order)

---

## 0. Reading basics

**Images.**
- Every image is 256 × 256 pixels, RGB.
- Each channel of each pixel is an integer from 0 (black) to 255 (full intensity). This guide calls that unit **"level 0–255"**.
- Grey (luminance) = 0.299·R + 0.587·G + 0.114·B, also in levels 0–255.
- Galaxies are centred; the border is almost always dark sky (about 30–35 levels) with a few stars.

**Regions used by the measures.**
```
+----------------------------+
|  outer ring (16 px)        |  <- "border": the sky
|   +--------------------+   |
|   |    +--------+      |   |
|   |    | centre |      |   |  <- "center"/"core": central 64x64 square = galaxy core
|   |    +--------+      |   |
|   +--------------------+   |
+----------------------------+
```
Noise is measured in a slightly wider 24 px ring, so the galaxy stays out of it.

**Histograms.**
- **x axis:** the value of the measure, with its unit in `[ ]` in the title or axis label.
- **y axis "density":** each histogram is normalised so that the area under the bars is 1. Groups of
  very different size can then be compared, for example 14,736 test images vs 996 grainy ones.
  Compare shapes; the y value is not a number of images.
- **y axis "number of images (log scale)":** a real count, where each tick is 10× the previous one.
- Overlapping bars are semi-transparent. A dashed vertical line is the threshold of a detector.

**Boxplots.** The box spans the 25th–75th percentile and the orange line is the median.
Whiskers reach the last value within 1.5 × the box height. Outliers are not drawn.

**Contact sheets (grids of thumbnails).**
- Title at the top.
- Caption above each thumbnail: usually the last 5 digits of the id (`00811` = `train_00811`) and
  `y=` + the label (0–9; `y=-1` in test).
- **Frames:** orange = grainy, red = alien, black = blank, grey = broken, blue = wrong-label duplicate, none = clean.
- **Pair sheets** are read two by two (1st with 2nd, 3rd with 4th, …).

---

## 1. Per-image measures (glossary)

All of them are in `csv/per_image.csv`, one row per image, rounded to 4 decimals. Column names are the same as in the charts.

| column | what it measures | unit / range | notes |
|---|---|---|---|
| `sky_mean` | mean brightness of the sky: mean of `border_r/g/b` | level 0–255 | grain rule: > 52 (with `lum_noise` > 10) |
| `mean_r/g/b` | mean of each channel over the whole image | level 0–255 | |
| `std_r/g/b` | standard deviation of each channel | level 0–255 | high = more contrast |
| `border_r/g/b` | mean of the sky ring (16 px) | level 0–255 | normal sky ≈ 30–40; grainy ≈ 60–90 |
| `center_r/g/b` | mean of the central 64×64 square | level 0–255 | |
| `gray_std` | standard deviation of the grey image | level 0–255 | ≈ 2 = blank image |
| `lap_var` | variance of the **Laplacian** (a filter that responds to sharp edges and grain) | level² | high = sharp **or** noisy; low = blurred |
| `sat_hi_frac` | fraction of pixels with a channel at 255 (burnt) | 0–1 | |
| `colorfulness` | Hasler–Süsstrunk colourfulness index | levels (≈ 0–100) | |
| `corr_rg`, `corr_gb` | Pearson correlation between channels R–G and G–B | −1…1 | near 1 = similar channels |
| `hf_energy` | share of the Fourier energy at **high frequency** (fine detail and noise) | 0–1 | |
| `chroma_noise` | **colour grain of the sky**: std of (R−G) after removing the smooth background (5×5 high-pass) in the 24 px ring | level 0–255 | clean train ≈ 17, grainy ≈ 35, test ≈ 14 |
| `lum_noise` | same, on luminance instead of colour difference | level 0–255 | very smooth test ≈ 2 |
| `skin_blob` | fraction of the image covered by the **largest pale-skin blob** (HSV colour range) | 0–1 | aliens ≥ 0.065, galaxies ≤ 0.05 |
| `core_r/g/b` | core **chromaticity**: share of R, G, B in the brightest 20% of the centre, i.e. R/(R+G+B) | 0–1, sum = 1 | a white core ≈ 0.33 / 0.33 / 0.33 |
| `gal_area` | galaxy area: pixels connected to the centre and brighter than the sky by > 3σ | fraction 0–1 | |
| `gal_axis_ratio` | minor / major axis of the galaxy | 0–1 | 1 = round, ≈ 0.2 = edge-on |
| `gal_offset` | distance of the galaxy centroid from the image centre | pixels | high = off-centre or double |
| `dark_flat` | share of 16×16 blocks that are dark (mean < 100) and flat (std < 3.5): missing data | 0–1 | broken train image if > 0.05; blank = 1 |
| `line_score` | largest jump of a row or column mean (per channel) with respect to its ±3 neighbours: straight-line artifacts | level 0–255 | typical ≈ 6; flagged > 25 |
| `alien`, `grainy`, `wrong_label_dup`, `broken`, `blank`, `saturated`, `smooth`, `line` | detector flags (§2) | true/false | |
| `tint` | core colour cast | `purple` / `green` / empty | |
| `category` | final category (§2) | text | |
| `p_test` | estimated probability that the image comes from the test (adversarial validation, all train vs all test, out-of-fold, §9) | 0–1 | near 0 = "train-like" |

---

## 2. Categories and thresholds

The detectors are **the same on train and test**. Thresholds are in the `CONFIG` block at the top of the script.

| flag | rule | threshold | how it was chosen |
|---|---|---|---|
| `blank` | `gray_std` < threshold | 4.0 | the 3 blank images are at 2.01; the first real image is at 8.1 |
| `alien` | `skin_blob` > threshold | 0.06 | train has a gap between 0.05 (galaxies) and 0.065 (aliens) |
| `grainy` | `chroma_noise` > 24 **or** (sky > 52 **and** `lum_noise` > 10) | 24 / 52 / 10 | clean train max ≈ 23.7; sky brightness is bimodal with a gap at 46–55 |
| `tint` | `core_g` outside the 1–99% range of clean train | [0.3228 0.3627, ] | by construction only 2% of clean train falls outside |
| `smooth` | `lum_noise` < 1st percentile of clean train | ≈ 5.8 | test smoother than almost all train |
| `broken` (train only) | `dark_flat` > threshold and not blank | 0.05 | rule from the teammate's EDA: one train image at 0.145, all others ≤ 0.004. Not used on test: blurred skies are flat too |
| `saturated` | `sat_hi_frac` > 0.05 | 5% | informative only |
| `line` | `line_score` > threshold | 25 | informative only (lines exist in train and test) |

Corrections decided by eye are in the `MANUAL` block of the script and applied on every run.
They are listed in notebook §7.

**Final category** = first true rule in this order:
- **train:** `blank` → `broken` → `alien` → `wrong_label_dup` → `grainy` → `clean`;
- **test:** `blank` → `tinted` → `smooth` → `grainy_like` → `alien_candidate` → `clean`.

In test, `grainy_like` and `alien_candidate` are **false alarms**: images above the threshold that,
by eye, are not the train corruption. They have different names on purpose.

---

## 3. `01_overview/`

**`label_distribution.png`**
- Train images per class, counted three ways.
- **x:** class (number + pasta name). **y:** number of train images.
- **Legend:** grey `raw` = all images; light blue `no_alien` = without aliens; dark blue `clean` = without aliens and grainy images.
- **Look for:** classes 2, 3, 5, 6 drop when going to `clean` (that is where the grainy images were);
  class 4 goes from 83 to **56**, the rarest class.

**`class_overview_clean.png`**
- One row per class (0 at the top), 12 random **clean** images each.
- The first caption of each row is the class number and name; the others are id digits.
- The fastest way to learn what each class looks like.

**`mean_image_fft.png`**
- **Top row:** pixel-wise mean of all images of each class (`c0`…`c9`) and of the test, brightened to be visible.
  `n=` is the number of images; corrupted images are included. A fixed overlay (text, frame, pattern)
  would show up here. There is only a blurred galaxy in the centre.
- **Bottom row:** mean log|FFT| (Fourier spectrum). Centre = low frequencies (large shapes), edges = high
  frequencies (detail, noise). Colour scale `magma`: dark = weak, yellow = strong. Isolated bright dots or
  sharp crosses would mean periodic noise. There are none.
- No axes: these are 256×256 images.

**`stats_by_class_clean.png`**
- 8 boxplots, one per measure: the test (first box) and classes 0–9 (**clean train only**).
- **y:** the value of the measure, unit in `[ ]` in the title.
- **Panels:**
  - `gal_area`: classes 5 and 6 are large.
  - `gal_axis_ratio`: class 2 ≈ 0.93 (round); 8 ≈ 0.21 and 9 ≈ 0.30 (edge-on); 4 ≈ 0.43.
  - `gal_offset`: class 1 (merging) is often off-centre.
  - `core_r/g/b`: class 9 has the reddest core (the bulge).
  - `colorfulness`.
  - `lap_var`: the test median is lower, i.e. smoother.
- Purpose: check that the clean labels make physical sense. They do.
  The medians behind the boxes are printed in notebook §3.

---

## 4. `02_class_sheets/`

- **Files:** `class<N>_<name>_p<page>of<total>.png`, e.g. `class3_ravioli_in_between_smooth_p05of07.png`.
  There are 47 pages of 100 images (10×10 grid, 128 px thumbnails).
- **Content:** **all** 4,328 train images by class, ordered clean → wrong-label duplicate → grainy → alien → broken → blank, then by id.
- **Title:** class, page and count per category.
- **Caption:** id digits + a letter: `g` grainy, `a` alien, `x` broken, `b` blank, `w` wrong-label duplicate, none = clean. Frames as in §0.
- **Use:** browse the first pages of each class to judge the labelling. The orange-framed images of class 3
  are **round**, i.e. really class 2: that is the label swap.

---

## 5. `03_corruptions/aliens/`

| file | content |
|---|---|
| `aliens_all_train.png` | all 332 train aliens, 20 per row, 80 px, sorted by increasing score; caption `id y=label` |
| `aliens_lowest_score.png` | the 30 aliens with the lowest `skin_blob` (caption = score). **Check:** all must be aliens, otherwise the threshold is too low |
| `non_aliens_highest_score.png` | the 30 train **non**-aliens with the highest score. **Check:** no alien (they are yellowish galaxies) |
| `test_highest_score.png` | the 30 test images with the highest score: all galaxies, so no aliens in test |
| `alien_score_hist.png` | **x:** `skin_blob` (0–1); **y:** number of images (log). Blue = train, orange = test, dashed = threshold 0.06. Train has a second hump at 0.07–0.2 (the aliens); test stops around 0.08 |

---

## 6. `03_corruptions/grainy/`

**`grain_hist.png`**
- **x:** `chroma_noise` (level 0–255). **y:** density. Dashed line = threshold 24.
- **Legend:** blue = clean + alien train, orange = grainy train, grey = test.
- Blue and orange are almost separate. The few orange bars below 24 come from the second rule (bright sky + noise).
  Grey sits to the **left** of blue: the test is less noisy than train.

**`identity_calibration.png`** proves that a grainy image and its "source" are **the same galaxy**.
- **Method:** mask the galaxy, compare the **stars** around it (their positions are a fingerprint), and take
  the correlation, best of the 8 rotations/flips.
- **x:** identity score, −1…1 (1 = identical star field). **y:** density.
- **Blue:** every clean image vs the most similar *other* clean image, i.e. how high **different** galaxies can score.
- **Orange:** every grainy image vs its best clean candidate.
- **Lines:** dashed at 0.7 = confirmed; dotted at 0.5 = below this, no source.
- Blue stays almost entirely below 0.5; orange is packed between 0.8 and 1, so most grainy images really are copies.

**`label_swap_matrix.png`**
- Cross-table over the 890 confirmed pairs.
- **Rows (y):** label of the clean source. **Columns (x):** label of the grainy copy.
- **Cell number:** count of pairs. Darker = more.
- Correct labels would sit on the diagonal. Instead everything is **off-diagonal**: 2→3 (265), 3→2 (245),
  5→6 (199), 6→5 (181). This is the systematic label swap.

**Pair sheets** (grainy first with an orange frame `G id y=label`, then the source `S id y=label id=score`):

| file | content |
|---|---|
| `grainy_vs_source_strongest30.png` | the 30 pairs with the highest identity score: same galaxy, different labels |
| `grainy_vs_source_weakest30.png` | the 30 confirmed pairs with the lowest score. **Check:** still the same galaxy? Yes |
| `grainy_vs_source_uncertain.png` | pairs between 0.5 and 0.7 not confirmed by eye (now 2: `00851`, `01368`) |
| `grainy_vs_test_top20.png` | each grainy image next to its best **test** candidate (`T id id=score`); scores are low, so the grainy images do not come from the test |

**Other sheets**
- `grainy_no_source_found.png`: the 104 grainy images with no candidate in train, sorted by label (mostly large bright galaxies). Dropped like the others.
- `grainy_train_first200.png`: the first 200 grainy images by label, to check that no alien hides under the noise.

---

## 7. `03_corruptions/duplicates/` and `lines/`

All sheets are **pair sheets**; frames show the category.

| file | content |
|---|---|
| `near_dups_train_top40.png` | the 40 most similar train–train pairs; almost all grainy–grainy (several degraded copies of one galaxy) |
| `near_dups_label_conflict.png` | the 3 pairs of **clean** images that are near-identical but have **different labels** |
| `near_dups_involving_test.png` | the 5 pairs involving the test (3 train–test, 2 test–test) |

The method and the columns are in `csv/near_duplicates.csv` (§10).

**`lines/lines_top15_train_test.png`**
- The 15 train images (caption `tr`) and then the 15 test images (`te`) with the highest `line_score`; caption = id digits, split, score. Frames = category.
- Straight horizontal/vertical lines (satellite trails, coloured rows or columns) appear in **both** train and test, so they are kept.

---

## 8. `04_test/`

| file | content | caption |
|---|---|---|
| `test_random64.png` | 64 random test images (8×8): the "average" test | id |
| `test_tinted_purple_most60.png` | the 60 most **purple/pink** test images (of 1,959) | `g=` `core_g` (lower = more purple) |
| `test_tinted_purple_median30.png` | 30 typical purple ones (middle of the ranking): the tint is not just a few extremes | same |
| `test_tinted_green_most60.png` | the 60 most **green** (of 1,361) | `g=` (higher = greener) |
| `test_tinted_green_median30.png` | 30 typical green ones | same |
| `clean_train_outside_tint_bounds.png` | **reference:** clean train images outside the same colour bounds. They have natural colours (blue spirals, yellow ellipticals), unlike the test | `g=` |
| `test_smooth_most60.png` | the 60 **smoothest** test images: no grain in the sky, blurred look | `n=` `lum_noise` |
| `clean_train_smoothest30.png` | **reference:** the 30 smoothest clean train images, which still show fine grain | `n=` |
| `test_grainy_like_top60.png` | test above the grain threshold: ordinary galaxies with some noise and a **dark** sky, **not** the train corruption | `n=` `chroma_noise` |
| `blank_images_train_and_test.png` | the 3 blank images: `train_03848` (y=8), `test_13385`, `test_13646` | full id |

**`core_colour_scatter.png`**
- Each dot is an image, placed by the **colour of its core**.
- **Left panel:** x = `core_g` (share of green, 0–1), y = `core_b` (share of blue). Dotted lines = tint bounds 0.3228 0.3627 and .
- **Right panel:** x = `core_r`, y = `core_g`.
- **Legend:** blue = clean train, orange = test.
- Clean train forms a narrow band. The test has the same band plus **diagonal arms**: the images with altered colour channels.
  Isolated far dots are images with a pure-colour core (e.g. a bright green star in the centre).

---

## 9. `05_shift/`

**`stats_clean_grainy_test.png`**
- 27 histograms, one per measure (§1). **x:** the measure, unit in `[ ]`. **y:** density.
- **Legend** (first panel): blue = clean train, orange = grainy train, grey = test.
- Each panel is clipped to the 0.1–99.9th percentile; more extreme values pile up in the last bar
  (e.g. `center_r` at 255: burnt cores of grainy images).
- Orange is far from blue and grey in almost every measure: the grainy images are another population.
  Grey (test) is shifted **left** of blue in `chroma_noise`, `lum_noise`, `lap_var`, `hf_energy`: the test is smoother.
  The orange spike at 0.333 in `core_*` is saturated white cores (R = G = B = 255).

**`clean_train_least_testlike.png`**
- The 40 clean train images that a classifier finds **least like the test**.
- Caption `id y=label p=…`, where p = estimated probability of coming from the test (0–1).
- **Adversarial validation:** a classifier (HistGradientBoosting, 5-fold out-of-fold) learns to tell train from
  test using only the measures of §1. If it succeeds, the two sets differ. The 4 variants and their AUC are in
  notebook §6: AUC ≈ 0.94 even for clean train vs clean test (0.5 = indistinguishable); noise and sharpness separate them most.
- **Note:** the `p` on this sheet comes from the *clean train vs all test* variant. The `p_test` column of
  `csv/per_image.csv` comes from *all train vs all test*, the only variant covering every image.
  The two values can differ for the same image.

---

## 10. `csv/`

Open them with Excel, Numbers or pandas. To compare with the teammate, join on `id`.

| file | rows | content |
|---|---|---|
| `cleaning_proposal_davide.csv` | 4,328 (train) | `id, label, category, action` (`keep` / `drop`), `reason` in plain words (2,995 keep, 1,333 drop), `dup_group` = smallest id of the group of copies of the same galaxy (near duplicates + grainy→source pairs), for `StratifiedGroupKFold`. A personal proposal, not the team's `shared/cleaning_ids.csv` |
| `per_image.csv` | 19,064 (train + test) | `id, split, label` (−1 in test), `category`, flags, all measures of §1 (incl. `dark_flat`, `line_score`), `p_test` |
| `grainy_matches.csv` | 996 (one per grainy image) | see below |
| `near_duplicates.csv` | 151 pairs | see below |

**`grainy_matches.csv`**

| column | meaning |
|---|---|
| `id`, `label` | the grainy image and its (swapped) label |
| `src_clean_id`, `src_clean_label` | the clean source found and its label |
| `ident_clean` | star-field identity score (−1…1) |
| `best_test_id`, `ident_test` | best candidate in the test and its identity score |
| `source_status` | `confirmed` (≥ 0.7 or checked by eye), `uncertain` (0.5–0.7), `none_found` (< 0.5) |
| `confirmed_by` | `identity_score` or `eye` |

**`near_duplicates.csv`**

| column | meaning |
|---|---|
| `id_a`, `split_a`, `label_a`, `cat_a` (and `_b`) | the two images of the pair |
| `dhash_min_d4` | distance between the dHash fingerprints (256 bits): number of differing bits, minimum over the 8 rotations/flips. 0 = identical; threshold 24 |
| `mad_64_min_d4` | mean absolute difference of the two images shrunk to 64×64 grey (level 0–255); threshold 6 |
| `pair_type` | `train-train`, `train-test`, `test-test` |
| `label_conflict` | true for a train pair with different labels |

---

## 10b. `06_embeddings/`

**The method.** A pretrained CNN (`convnextv2_tiny.fcmae_ft_in22k_in1k`, the playbook's family D backbone) is used frozen,
as a feature extractor: each image becomes a vector of 768 numbers (its *embedding*). Each image is passed 8 times
(4 rotations × flip) and the 8 vectors are averaged, so the embedding does not depend on orientation.
Vectors are L2-normalised, so the similarity of two images is the cosine of their vectors (1 = identical).
Images that look alike to the CNN have close embeddings.

| file | content |
|---|---|
| `umap_by_class.png` | **UMAP** map: the 768 dimensions squeezed to 2 so that close images stay close. Each dot is an image; clean train coloured by class (legend), test in light grey. The axes have **no unit** and no meaning in themselves: only distances and clusters matter |
| `umap_by_category.png` | same map coloured by category (legend with counts): test clean / blurred / tinted, train clean / grainy / alien |
| `confusion_knn_probe.png` | two confusion matrices on the clean train. **Rows** = given class, **columns** = predicted class, cell = share of the row (0–1, darker = more). Left: **k-NN** (each image gets the majority class of its 10 nearest clean neighbours, itself excluded). Right: **linear probe** (logistic regression on the embeddings, 5-fold out-of-fold). The title gives the weighted F1. Off-diagonal cells = classes the embeddings confuse |
| `outlier_score_by_category.png` | boxplot of the **outlier score** (1 − mean cosine similarity to the 10 nearest clean train images; 0 = identical to known images, higher = more unusual) for each category and split |
| `possible_label_errors.png` | clean train images whose 10 nearest neighbours mostly (≥ 70%) belong to ONE other class and at most 2 share its label. Caption `id y=given nn=neighbour class share`. **Candidates** to look at, not proven errors |
| `clean_train_outliers.png` | the 40 clean train images with the highest outlier score (caption = score) |
| `test_far_from_train.png` | the 40 test images farthest from clean train (frame = category, e.g. purple = tinted) |
| `embedding_analysis.csv` | one row per image: `id, split, label, category`, `umap_x, umap_y`, `knn_pred` (majority class of the 10 nearest clean train), `knn_agree` (share of those neighbours with the same label, train only), `outlier`, `probe_pred`, `probe_conf` (probability of the predicted class; out-of-fold for clean train) |

The raw embeddings are in `eda/outputs/embeddings_cache/emb_convnextv2_tiny_{train,test}.npy` (+ `_ids.csv` for the row order):
the same input family D needs.

---

## 10c. `07_test_distortions/`

Measures the two test distortions so that the training augmentations reproduce them.

| file | content |
|---|---|
| `hue_shift_hist.png` | **x:** hue of the galaxy core minus the clean-train mean (degrees, −180…180; 0 = natural colour, ±180 = opposite colour, e.g. yellow → blue). **y:** % of images per 10° bin. Blue bars = clean train (packed around 0), orange = test, black line = the part of the test that is rotated (test minus the smallest share of "train-like" images that keeps every bin non-negative). That share is a **lower bound** on the rotated images (≥ 35%) |
| `tint_undo_examples.png` | rows 1 and 3: typical purple and green test images; rows 2 and 4: the same images rotated back in HSV hue by the measured angle (caption). Natural colours come back, which confirms that the cast is a hue rotation |
| `blur_calibration.png` | **left:** sky noise (`lum_noise`, level 0–255) of clean train images blurred with a known Gaussian σ (x, pixels at 256 px); band = 25–75%; dashed lines = medians of blurred and clean test. **Centre:** average log power spectrum of the test relative to clean train, by spatial frequency (cycles/pixel; 0 = same as train, negative = less detail), with Gaussian fits (dotted, σ in the legend). **Right:** % of images per estimated σ, clean train vs test |
| `simulated_vs_real.png` | 4 clean train images: original, hue rotated by −100°, +70°, 180°, blurred σ = 0.9, blurred + rotated; last row: real test images (2 purple, 2 green, 2 blurred) for comparison |
| `test_distortions.csv` | one row per test image: `id, category, tint`, `core_hue_shift` (degrees), `blur_sigma_est` (pixels; ≈ 0 = no blur, from the calibration curve) |
| `augmentation_params.json` | measured shares and the recommended augmentation: hue rotation p = 0.4 with angle uniform in [−180°, 180°] (`ColorJitter(hue=0.5)`); Gaussian blur p = 0.35 with σ in [0.4, 1.2] px at 256 px; applied independently |

---

## 11. `images/`

Single PNGs, named by their original id. They are **APFS clones** of the dataset files: no extra disk space
(even if Finder shows ≈ 1 GB), and editing one does not change the dataset.

```
images/
├── train_by_class/
│   ├── 0_spaghetti_disturbed/       clean/ (182)  alien/ (30)  wrong_label_dup/ (1)
│   ├── 1_gemelli_merging/           clean/ (313)  alien/ (32)
│   ├── 2_gnocchi_round_smooth/      clean/ (446)  grainy/ (259)  alien/ (38)  wrong_label_dup/ (1)
│   ├── 3_ravioli_in_between_smooth/ clean/ (343)  grainy/ (302)  alien/ (40)
│   ├── 4_penne_cigar/               clean/ (56)   alien/ (27)
│   ├── 5_fusilli_barred_spiral/     clean/ (346)  grainy/ (203)  alien/ (35)
│   ├── 6_fusilli_tight_spiral/      clean/ (309)  grainy/ (232)  alien/ (24)
│   ├── 7_riccioli_loose_spiral/     clean/ (444)  alien/ (32)  wrong_label_dup/ (1)
│   ├── 8_linguine_edge_on_no_bulge/ clean/ (239)  alien/ (35)  broken/ (1)  blank/ (1)
│   └── 9_lasagna_edge_on_bulge/     clean/ (317)  alien/ (39)
├── test/
│   ├── random/            300 random test images
│   ├── tinted_purple/     the 300 most purple
│   ├── tinted_green/      the 300 most green
│   ├── smooth/            the 300 smoothest
│   ├── grainy_like/       all 448 above the grain threshold (false alarms)
│   ├── alien_candidates/  the 30 with the highest alien score (no real alien)
│   └── blank/             the 3 blank images (including the train one)
└── review/                pairs to compare: same leading number = same pair
    ├── grainy_vs_source/  890 confirmed pairs, e.g.
    │                       001_a_grainy_train_00811_y3.png  <- grainy copy, label 3
    │                       001_b_source_train_01671_y2.png  <- clean original, label 2
    ├── near_duplicates/   151 pairs: 001_a_<id>_y<label>.png / 001_b_...
    └── label_conflicts/   3 clean pairs with different labels
```

**Tip:** in Finder use large icons, or select all and press Space (Quick Look). In `review/`, the `_a_` and
`_b_` files of a pair sit next to each other because they start with the same number.

---

## 12. The notebook

`eda/00_eda_davide.ipynb` tells the whole EDA from start to finish, already executed (outputs saved inside).
- It reads `eda/outputs/davide/`, recomputes the key tables (integrity, categories, class counts and medians,
  label-swap matrix, duplicates, adversarial validation, cleaning proposal), draws the main charts and shows the key sheets downscaled.
- It contains the decisions taken by eye and ends with the conclusions for training.
- In VS Code, pick the `.venv` kernel.
- With `RUN_AUDIT = True` in the first code cell (or if the outputs are missing) it reruns the script (about 2.5 minutes).
- The heavy work stays in the script because its parallel workers do not run well from a notebook on macOS.

`eda/outputs/davide/README.md` is the short folder index, regenerated by the script.

---

## 13. Suggested reading order

0. `eda/00_eda_davide.ipynb`: the whole story in one file; then the details below.
1. `01_overview/class_overview_clean.png`: learn the classes.
2. `01_overview/label_distribution.png`: how many images are really left.
3. `02_class_sheets/` of class 3: orange frames around round galaxies.
4. `03_corruptions/grainy/label_swap_matrix.png` and `grainy_vs_source_strongest30.png`: the proof of the swap.
5. `03_corruptions/aliens/aliens_all_train.png`: the intruders.
6. `04_test/test_random64.png`, `test_tinted_*_most60.png`, `core_colour_scatter.png`: how the test differs.
7. `04_test/test_smooth_most60.png` next to `clean_train_smoothest30.png`: the blurred test.
8. `05_shift/stats_clean_grainy_test.png`: the full picture of the differences.
8b. `06_embeddings/umap_by_class.png` and `confusion_knn_probe.png`: which classes overlap.
8c. `07_test_distortions/tint_undo_examples.png` and `augmentation_params.json`: what to put in the training augmentation.
9. `csv/` when comparing numbers with the teammate's EDA.
