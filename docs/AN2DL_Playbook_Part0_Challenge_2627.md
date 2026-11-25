# AN2DL Challenge Playbook (v7.2) — Part 0: Challenge Brief 2026-27 ("Flying SPAGHETTI")

Challenge-specific layer on top of the generic playbook. Where this file and Part I disagree, **this file wins** for the 2026-27 challenge.

- **Part 0 — Challenge Brief (this file):** rules, deliverables, calendar, and the technique plan tailored to this task.
- **Part I — Strategy:** [AN2DL_Playbook_Part1_Strategy.md](./AN2DL_Playbook_Part1_Strategy.md) (generic; relevant sections are §2, §3.2, §4, §5).
- **Part II — Code:** [AN2DL_Playbook_Part2_Code.md](./AN2DL_Playbook_Part2_Code.md).

---

## 0.1 Fact Sheet

| Item | Value |
| --- | --- |
| Task | **Galaxy morphology classification**, 10 classes (integer labels `0–9`) |
| Input | **256 × 256 RGB PNG** (`uint8`, 0–255), one galaxy per image (see §0.1b) |
| Train | **4,328** images `train/train_00000.png … train_04327.png` + `train.csv` (`id`, `label`) |
| Test | **14,736** images `test/test_00000.png … test_14735.png` — **3.4× larger than train** |
| Size | 2.69 GB total; `sample_submission.csv` lists every test id |
| Class balance | **Imbalanced** ("some pasta shapes are much rarer than others") — exact counts from EDA |
| Validation | **None provided** → build our own folds |
| Data quirks announced | "unexplained visual artifacts, bizarre color distortions, suspiciously systematic disturbances", "identify corrupted observations", intruders ("Area 51 files"); Data tab: *"Do not assume that the training archive and the new observations look alike, or that every entry in the archive can be trusted"* → **train/test shift + label noise / corrupted train entries** |
| Metric | **Weighted F1** (per-class F1 weighted by test support `n_c / N`) — *not* Macro-F1 |
| Leaderboard | Public LB = subset of test; **final ranking on a separate private subset** |
| Submission | CSV with header `id,label`, e.g. `test_00000,3` |
| Dates | **Nov 25 00:00 → Dec 8 23:59** (CET) |
| Team | 4 members, registered in the Kaggle "Team" tab; Kaggle nickname = Google-form nickname |
| External data | ❌ **Forbidden** — only the provided dataset |
| Pretrained models | ✅ **Allowed** (ImageNet / DINOv2 / CLIP backbones via `timm`, `torchvision`, HF) |
| Individual requirement | ⚠️ **Every member must make ≥ 1 submission**, otherwise that member gets **0** |
| Code sharing between teams | ❌ Forbidden (identical submissions → 40/40, 60/0 or 0/0 split) |
| Grade | LB + report → **0–6 pts** (same for the team), +0.5 for exceptional work, then **individual oral** |
| Daily submission limit | check the Submit page (not stated in Overview/Data) |
| GPU | Colab / Kaggle free GPUs (T4/P100) or local |

### 0.1b Classes (galaxy morphology, "pasta" names)

| Label | Pasta name | Morphology | Likely confusions |
| --- | --- | --- | --- |
| 0 | Spaghetti alla rinfusa | Disturbed / irregular | 1 (merging), 7 |
| 1 | Gemelli | Merging pair | 0 |
| 2 | Gnocchi | Round smooth (elliptical) | 3 |
| 3 | Ravioli | In-between round smooth | 2, 4 |
| 4 | Penne | Cigar-shaped smooth | 3, 8, 9 |
| 5 | Fusilli con la sbarra | Barred spiral | 6, 7 |
| 6 | Fusilli stretti | Unbarred tight spiral | 5, 7 |
| 7 | Riccioli | Unbarred loose spiral | 6, 0 |
| 8 | Linguine | Edge-on, no bulge | 9, 4 |
| 9 | Lasagna | Edge-on with bulge | 8 |

The taxonomy matches the public **Galaxy10 DECaLS** dataset (Galaxy Zoo volunteer labels, 256×256 g/r/z composites). Consequences:
- **Do NOT download Galaxy10 / Galaxy Zoo data** — external data is forbidden, and the test set very likely comes from it (would also be flagged as cheating). The same applies to checkpoints **fine-tuned on Galaxy Zoo** (e.g., Zoobot): ask on the forum before even considering them; default = generic ImageNet/DINOv2/CLIP weights only.
- Labels come from volunteer votes → **intrinsic label noise** even before the organizers' corruption, especially within {2,3}, {5,6,7}, {8,9}, {0,1}. Ceiling accuracy is well below 100% (published Galaxy10 DECaLS results are roughly in the 85–90% range with fine-tuned CNNs).
- Galaxies are **centered** in the frame and have **no preferred orientation** → full rotation/flip invariance (D4 + arbitrary angles) is label-preserving for all 10 classes (an edge-on disk stays edge-on when rotated). Arbitrary-angle rotation needs care with corners (rotate then center-crop, or reflect padding).
- **Color carries information** (ellipticals are redder, spirals bluer arms) → photometric augmentations must be mild.
- Fine details (bar, arm tightness, bulge) matter → avoid aggressive downscaling/crops; prefer native 256 or 224 center-ish crops.

### Answers to the Part I §1.1 rules checklist

- Pretrained weights: **yes**. External data: **no** — so no Galaxy10/Galaxy Zoo/SDSS/DECaLS images, no web images, no "external" pseudo-labeled sets.
- Non-neural models: not forbidden, but the course asks to use "the frameworks discussed in practical sessions" → keep the core in PyTorch/Keras; a k-NN/logistic head on frozen embeddings is fine as a diversity member.
- Unlabeled test usage (pseudo-labeling, TTA statistics, self-supervised pretraining on test images): not explicitly regulated → **ask on the Discussion forum on Day 0**. With test 3.4× larger than train *and* an announced train/test shift, this can be a strong lever, but it is **optional**: the rest of the plan does not depend on it. It is not leakage in itself (only test *images* are used, never test labels), but the course may consider it outside the spirit of the challenge → get a clear answer early. Protocol in §0.5b.
- Delivery is a **CSV file**, not code → no inference time limit, so large ensembles + heavy TTA are feasible (Plan B of Part I §5.3 not needed).

---

## 0.2 Deliverables (Do Not Lose Points Here)

1. **Kaggle submissions** — final selection of **2 submissions** (best CV blend + conservative fallback). Every member submits at least once (do it on Day 1–2 with the baseline!).
2. **Report** — PDF, **max 3 pages excluding references**, **official Overleaf template unmodified** (any change or extra page → penalty). Content: members, development process, ideas behind the final solution, evidence (tables of ablations, CV vs LB).
3. **Notebooks** — a `.zip` with one or more **fully executed notebooks** (all outputs visible) that reproduce the models in the report: model creation, training, local performance. Code commented and reproducible.
   - Our repo keeps logic in `src/` + `families/*/train.py`; **plan from Day 1** a thin `final/` notebook per family that imports the code (or inlines it) and keeps the training log outputs. Re-running everything on Dec 7–8 is not realistic: save executed notebooks as you train the final models.
4. **Email** — sent by **one member** to the course address by **Dec 8 23:59**, subject `AN2DL2627 - Challenge`, body *exactly* the template (GROUP_NAME, STUDENT_1..4 NAME/ID/EMAIL, REPORT_FILENAME, ZIP_FILENAME), no extra text. Prepare the draft on Day 10.

---

## 0.3 Calendar (Real Dates)

| Day | Date | Phase | Must-have |
| --- | --- | --- | --- |
| 0 | Tue Nov 25 | Rules, data download, repo, smoke test | Download data (2.69 GB) to the shared drive, everyone joins the team, ask the forum about test usage |
| 1–2 | Nov 26–27 | EDA + **corruption audit** (§0.4) | Baseline 5-fold, **first submission by every member** |
| 3 | Fri Nov 28 | **Freeze gate** | `train_folds.csv`, `cleaning_ids.csv`, `metric.py` frozen |
| 3–9 | Nov 28 – Dec 4 | Family exploration | Ensemble dry-run on Dec 1 |
| 9–12 | Dec 4 – Dec 7 | Convergence | Final configs on all folds; **executed notebooks saved**; report draft started Dec 5 |
| 12 | Sun Dec 7 | Training cutoff | Report ~final, email draft ready |
| 13 | Mon Dec 8 | Blend + submit + send | Final 2 submissions selected, PDF + zip emailed **well before 23:59** |

Compared to the generic schedule, the report and notebooks eat ~1.5 days: the training cutoff is effectively **Dec 6 evening**.

---

## 0.4 The Core Problem: Corrupted Observations

The story and the Data tab (*"not every entry in the archive can be trusted"*) say explicitly that the **training archive is contaminated**. Expect some mix of:

| Contamination type | What it looks like | How to detect | What to do |
| --- | --- | --- | --- |
| **Out-of-domain intruders** ("Area 51") | Images that are not galaxies at all (cartoons, objects, memes) | Embedding outliers (DINOv2/CLIP k-NN distance), near-duplicate clusters, visual grid inspection | **Remove from train**; check if the same kind appears in test |
| **Exact / near duplicates** | Same image repeated, possibly with different labels | Perceptual hash (pHash/dHash) + cosine similarity of embeddings | Keep in the same fold; inspect label conflicts manually |
| **Color distortions** | Channel swaps, tints, inverted colors, saturation shifts | Per-image channel mean/std histograms, hue statistics, outlier scores | Either remove or **train with matching augmentations** if test contains them too |
| **Systematic artifacts** | Overlaid patterns, stripes, grids, watermarks, constant patches, periodic noise | 2D FFT spectrum peaks, row/column mean profiles, pixel-wise median over images, Grad-CAM focusing away from the central galaxy | Mask/inpaint if fixed location; augmentation; **check for shortcut learning** |
| **Label noise** | Correct-looking image, wrong class | High-loss OOF samples, confident learning (cleanlab-style) | Re-label is not allowed → drop from train or use noise-robust loss |
| **Degraded quality** | Blur, Gaussian/salt-pepper noise, low-contrast, black/blank images | Laplacian variance, image std ≈ 0, saturation fraction | Drop blanks; keep mild ones (they are test-like) |

**Key decision rule — compare with the test set first.** Run the same detectors on test images (no labels needed):
- if a corruption type appears **only in train** → remove it from training (pure noise);
- if it appears **also in test** → keep it (or a cleaned copy) and make the model robust to it (augmentations, TTA), because the test predictions for those images still count in the weighted F1;
- if intruders appear in test, they still need a label in the CSV: predict them with the model anyway (they are scored on whatever ground truth the organizers set — check the forum).

**Shortcut warning.** "Systematic disturbances" may correlate with labels in train (e.g., an artifact present mostly in one class). The model will learn the artifact, and the private LB will punish it if the correlation differs. Always run **Grad-CAM** (Part II H4) on top-confidence predictions per class and look at the adversarial-validation AUC (Part I §2.3).

### Day 1–2 audit notebook (`eda/01_audit.ipynb`) — checklist

- [ ] Check all files open and are `(256, 256, 3)` `uint8` (some corrupted PNGs may be grayscale/RGBA or a different size); label histogram → class imbalance level (rare classes: probably 4 *Penne* and the edge-on/disturbed ones — verify).
- [ ] Grid of 64 random images per class + 64 random test images.
- [ ] Per-image stats: mean/std per channel, Laplacian variance, fraction of saturated pixels; plot histograms and look at the tails.
- [ ] pHash duplicates within train, within test, and **train ↔ test** (train↔test duplicates: their label is a free hint for the corresponding test prediction, but do not let them distort CV).
- [ ] DINOv2 (or `convnext`/CLIP) embeddings for all images → UMAP/t-SNE coloured by class and by train/test; k-NN outlier score.
- [ ] Mean image and FFT magnitude averaged per class (reveals fixed overlays / periodic noise).
- [ ] Adversarial validation on embeddings (train vs test AUC).
- [ ] Output: `shared/cleaning_ids.csv` with columns `id, reason, action` (`drop` / `keep_flag`).

## 0.4b Train ↔ Test Shift ("the archive and the new observations do not look alike")

The test set (14,736) is the "new observations"; the train (4,328) is the "archive" that had "a rough couple of weeks". Most likely scenario: **train contains corruptions that test does not** (or a different mix), plus label noise. Treat it as a small **domain adaptation** problem:

1. **Measure the shift on Day 1**: adversarial validation train vs test on DINOv2 embeddings and on simple image stats (per-channel mean/std, sharpness, noise level). AUC ≈ 0.5 → no shift; AUC ≫ 0.6 → find which images drive it (top train images most "un-test-like" = prime cleaning candidates).
2. **Make validation look like test**: once corrupted train images are identified, evaluate CV on the *clean-looking* validation subset (the one resembling test) as the primary metric, the full fold as secondary. Check that it tracks the public LB from the first submissions.
3. **Make train look like test**: remove or repair corruptions that never appear in test (e.g., undo channel swaps / color casts if the transformation is invertible — repairing keeps precious labeled samples, train is small).
4. **Make test look like train only if needed**: if test has degradations train lacks (e.g., noise, blur, different color calibration), add them as augmentations.
5. **Exploit the unlabeled test set** (only if allowed by the forum answer): self-supervised/continued pretraining or BN-statistics adaptation on test images, then per-fold pseudo-labeling (§0.5b).

---

## 0.5 Technique Plan for This Task

### Validation

- **Stratified 5-fold** on the cleaned train set (near-duplicate groups in the same fold → `create_and_save_folds(..., group_col='dup_group')`).
- Report CV on the **full validation folds** (intruders/corrupted kept) *and* on cleaned validation; use the one that best tracks the public LB.
- Metric everywhere: `src/metric.py::score` (weighted F1). Weighted F1 ≈ accuracy-like behaviour dominated by large classes, so **class multipliers usually give little**; keep them only through nested CV (`full_nested_postprocessing(..., average='weighted')`).

### Families (suggested split for 4 people)

| Family | Model | Why |
| --- | --- | --- |
| **Lead** | Data audit, cleaning, shift analysis, folds, metric; later ensemble | Contamination + shift handling is the highest-leverage work here |
| **A — Modern CNN** | `convnext_tiny` / `convnext_small` (IN-22k), alt. `efficientnetv2_s` | Strong fine-tuning at 224–256 px on a T4; good at fine texture (arms, bars) |
| **B — Self-supervised ViT** | DINOv2 ViT-S/14 → ViT-B/14 fine-tuned with LLRD (input 224 or 252) | Robust features on non-ImageNet domains, natural fit for embeddings/outlier detection and test-set adaptation |
| **C — Robust / cheap diversity** | `resnet50`/`regnety_032` with heavy rotation aug + noise-robust loss, **or** logistic/k-NN head on frozen DINOv2/CLIP embeddings (+ rotation-averaged embeddings) | Fast, robust to label noise, different errors |

With only 4,328 train images, a 5-fold × 3 families run is cheap (≈ 3.5k images/fold): one ConvNeXt-T epoch at 224 on a T4 takes well under a minute. The bottleneck is PNG decoding → **pre-decode once to a `uint8` `.npy` array** (4,328×256×256×3 ≈ 850 MB; test ≈ 2.9 GB, keep on disk with `np.load(mmap_mode='r')` or resize to 224 first).

### Input pipeline

- Native **256×256**: try (i) train 224 random-resized-crop (scale 0.8–1.0, the galaxy is centered and fine structure matters) / test 224 center crop or 256 resize; (ii) native 256 for CNNs. For ViT/DINOv2 use 224 or 252 (= 18×14 patches) and keep the same size for TTA.
- Center crop is a cheap trick: outer pixels are mostly background sky/neighbors; a 192–224 center crop resized to 224 may *help*. Ablate.
- Normalize with the **backbone's** pretraining mean/std.

### Augmentations (ablate by family, Part I §4.3)

- **Geometric (core, label-preserving for galaxies):** h/v flips, 90° rotations, **arbitrary-angle rotation** (reflect padding or rotate-then-crop), small shift (±8 px), mild zoom (0.9–1.1). This is the highest-value augmentation for galaxy morphology.
- **Photometric (mild):** brightness/contrast ±10–20%, small gamma; **hue jitter off or tiny** (color separates ellipticals from spirals). Ablate saturation separately.
- **Corruption-robustness**, *only for the corruption types found in test* (§0.4b): Gaussian noise, blur, JPEG, random erasing on the border area (not the center).
- **MixUp:** worth a try (light, α≈0.2). **CutMix: probably harmful** — the galaxy is a single centered object, a pasted patch can delete the bar/bulge that defines the class.

### Training recipe (starting point)

- AdamW, lr `3e-4` head / LLRD 0.75 (ViT) or ~1e-4 full-model (ConvNeXt), wd 0.05, warmup 5% + cosine, **30–50 epochs** (small dataset), label smoothing 0.1, EMA, AMP `fp16` on T4, batch 32–64.
- Small data → test **two-stage transfer** (2–3 epochs frozen backbone, then full unfreeze, Part II H3).
- **Noise-robust training** (label noise is announced + volunteer labels): label smoothing 0.1–0.2, Symmetric/Generalized CE, small-loss selection (after warmup, ignore the top 5–10% loss samples per batch), or the classic two-pass: train → rank OOF losses → drop/flag worst train samples → retrain (train split only, Part I §2.2).
- **Imbalance:** weighted F1 favors frequent classes, so start **without reweighting**; try class-balanced sampling / logit adjustment for rare classes only if nested CV improves. Stratified folds are mandatory (rare classes may have < 100 samples).

### Inference & ensembling

- Fold averaging (5 models) + **rotation TTA**: D4 (8 views), optionally + 45° rotations (16 views). Select the view set on CV; same TTA on OOF and test (`predict_with_tta` currently does flips only → extend with `torch.rot90`).
- Calibrate (temperature) → Caruana on log-loss → nested post-processing with `average='weighted'`.
- Final picks: (1) best nested-CV blend, (2) simple mean of the best model per family without multipliers.

### "Nice to have" if time allows

- **Pseudo-labeling on test** (if allowed): see the dedicated protocol in §0.5b.
- Rotation-equivariant tricks: averaging embeddings over 8 rotations before a linear head (family C), or group-equivariant CNNs (more report value than LB value).
- Knowledge distillation of the ensemble into one model.
- CLIP zero-shot as a sanity check for intruder detection ("a photo of a galaxy" vs anything else).

---

## 0.5b Pseudo-Labeling Protocol (Optional, Only if Allowed)

**What it is.** Use the trained models to predict the unlabeled test images, keep only the predictions that are reliably correct, and add those images (with the predicted "pseudo-labels") to the training set of a new model (the *student*). It never uses test labels, so it is not leakage by itself — but it can make **CV lie** and can make the model **reinforce its own mistakes** if done carelessly. Rules below prevent both.

**Go / no-go.** Only if the forum confirms that unlabeled test usage is allowed. It is kept **only if it clearly improves CV** (gain > 2× seed std); otherwise it is dropped without regrets.

### Rules

1. **When: at the end, not at the start.** Pseudo-labels are generated on **Day 9–10 (Dec 4–5)** from the best available models; students are retrained on **Days 10–12**. Early models are too weak to produce useful pseudo-labels.
2. **Per-fold teachers (never the full ensemble).** The pseudo-labels used by the student evaluated on fold *k* must come only from models that never saw fold *k*:

   | Student validated on | Pseudo-labels generated by | Student trained on |
   | --- | --- | --- |
   | F1 | blend of **A1 + B1 + C1** (cross-family ensemble is fine) | F2–F5 + test pseudo-labeled by teacher 1 |
   | F2 | blend of **A2 + B2 + C2** | F1, F3–F5 + test pseudo-labeled by teacher 2 |
   | … | … | … |

   Using the blend of all folds as teacher leaks fold-*k* labels into the student evaluated on fold *k* (label of F1 → models trained on F1 → pseudo-labels → student → evaluated on F1) and inflates CV.
   **The final run stays per-fold too.** A "strongest possible" all-fold teacher for the final models is *not* used: marginal gain over a 3-family per-fold teacher, unverifiable CV, and students that copy the same errors (less ensemble diversity). The final submission is the ensemble of the per-fold students.
3. **Never validate on pseudo-labeled images.** Validation folds contain only train images with true labels. Pseudo-labeled test images are training data only.
4. **Calibrate, then choose per-class thresholds on OOF.** A raw "97%" does not mean 97% correct (networks are overconfident):
   - temperature-scale each teacher on OOF logits (`fit_temperature`);
   - on the teacher's OOF predictions, for each class find the lowest confidence threshold at which **measured precision ≥ 95%**; classes that never reach it get **no pseudo-labels**;
   - be conservative: test does not look like train (§0.4b), so real test precision is likely a bit lower than OOF precision;
   - cap the number per class (e.g., at most 2–3× the class's train count) so easy classes (round smooth) do not swamp the rare ones.
5. **Soft labels with reduced weight.** Store the teacher's calibrated **probability vector**, not the argmax, and weight the pseudo-label loss at ~0.3 (`compute_semi_supervised_loss`, Part II H1). Uncertain teacher → soft target → the student does not learn a hard mistake.
6. **One round (two at most).** Teacher → student is one round. A second round (student becomes teacher) is allowed only if round 1 clearly helped and there is time before the Dec 6–7 cutoff. Online methods (Mean Teacher, FixMatch) are out of scope.
7. **No human labeling of test images.** Labeling or correcting test images by hand is hand-labeling the test set (cheating). Human inspection is welcome **on the train set** (cleaning, §0.4).

### Checklist

- [ ] Forum answer: allowed? (if no → skip this section)
- [ ] Per-fold teachers A_k + B_k + C_k calibrated (temperature on OOF of the K−1 training folds)
- [ ] Per-class thresholds from OOF precision ≥ 95%, per-class caps, count of pseudo-labels per class logged
- [ ] Soft labels saved as `pseudo_soft_fold{k}.npy` `(N_test, 10)` + mask of selected ids
- [ ] Students trained per fold, validated on true-label folds only
- [ ] CV with vs without pseudo-labels compared (same seeds); kept only if gain > 2× seed std
- [ ] Report: protocol, threshold table, gain (or why it was dropped)

---

## 0.6 Report Plan (3 pages)

Collect material during the challenge, not at the end:
- **Dataset & cleaning** (½ page): contamination types found, counts, examples grid, decision rule (train vs test).
- **Method** (1 page): validation, families, augmentation, training recipe.
- **Experiments** (1 page): ablation table from `runs/*.json` (change → CV → LB), CV vs public LB plot, final ensemble composition.
- **Conclusions / limitations** (½ page).
Keep one shared folder of figures (`report/figures/`) and update the ablation table daily.

---

## 0.7 Open Questions for the Forum

- [ ] **Is unlabeled test usage allowed** (pseudo-labeling, self-supervised pretraining / BN adaptation on test images)? ← ask on Day 0.
- [ ] Are checkpoints pretrained on astronomical data (e.g., Galaxy Zoo / Zoobot) considered "external data"? (Default: avoid.)
- [ ] Daily submission limit and number of final submissions selectable.
- [ ] Are corrupted images also present in the test set, and are they scored?
- [ ] Is Keras/TensorFlow required, or is PyTorch + `timm` acceptable in the notebooks?
