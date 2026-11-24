# AN2DL Challenge Playbook (v6.1) — Part I: Strategy and Techniques

- **Part I — Strategy and Techniques (this file):** Guiding principles, Day 0 setup, data and validation, model family selection, training priorities and techniques, inference/ensembling, error catalog, and operational checklists.
- **Part II — Code ([AN2DL_Playbook_Part2_Code.md](./AN2DL_Playbook_Part2_Code.md)):** Verification status, module map for [an2dl_toolkit.py](../src/an2dl_toolkit.py) and [smoke_test.py](../tests/smoke_test.py), and standalone blocks (D and H).

---

## Table of Contents

**Part I — Strategy and Techniques**

0. [Guiding Principles](#0-guiding-principles)
1. [Day 0 and Team Organization](#1-day-0-and-team-organization)
2. [Days 1–3: Data, Cleaning, and Anti-Leakage Validation](#2-days-13-data-cleaning-and-anti-leakage-validation)
3. [Days 1–9: Problem Map (Classical to Advanced)](#3-days-19-problem-map-classical-to-advanced)
4. [Days 3–12: Training Techniques and Exploration Priorities](#4-days-312-training-techniques-and-exploration-priorities)
5. [Days 9–14: Inference, TTA, Ensembling, and Submission](#5-days-914-inference-tta-ensembling-and-submission)
6. [Error Catalog](#6-error-catalog)
7. [Individual Checklists](#7-individual-checklists)

**Part II — Code:** see [AN2DL_Playbook_Part2_Code.md](./AN2DL_Playbook_Part2_Code.md) (core code in [an2dl_toolkit.py](../src/an2dl_toolkit.py) and [smoke_test.py](../tests/smoke_test.py))

---

# PART I — STRATEGY AND TECHNIQUES

## 0. Guiding Principles

1. **Local validation (CV) matters more than any architecture.** If CV does not correlate with the leaderboard, every subsequent decision is noise and final shake-up will drop your rank.
2. **Simple baseline first, complexity later.** A working baseline on Day 2 is the yardstick for everything else. Test every change in isolation (one-variable-at-a-time ablation).
3. **Order of importance** (indicative, task-dependent):
   data & validation → baseline & model family choice → pretraining / backbone → augmentation & loss → ensembling & TTA → "advanced" techniques (EMA, SAM, pseudo-labeling, threshold tuning).
4. **Diversity beats quantity** in ensembles: three models from distinct families (e.g., CNN + Transformer + non-neural model) outperform ten variants of the same architecture.
5. **Read the rules before writing code.** Pretrained weights, external data, non-neural libraries, test-set pseudo-labeling, ensemble limits, and submission quotas change the entire plan.
6. **Honest expectations.** Many paper "SOTA" techniques give small, dataset-dependent gains. Validate them in CV and keep them only if the gain exceeds noise (e.g., seed variance).

---

## 1. Day 0 and Team Organization

### 1.1 Rules Checklist (Day 0)

- Are pretrained weights allowed (`timm`, HuggingFace, DINOv2)? Can they be downloaded during evaluation or must they be uploaded as datasets (offline environment)?
- Are non-neural models allowed (LightGBM, CatBoost, MiniRocket via `aeon`/`sktime`), or must submitted code be pure PyTorch/Keras?
- Is using the unlabeled test set allowed (pseudo-labeling, statistics, adaptation)?
- Constraints: model size limit, daily submission limit, inference time limit, official metric, submission format (CSV file vs. executable code).
- Native data resolution and format; available GPU (free T4, paid L4/A100).

### 1.2 Core Idea and Team Roles

The workflow is a **funnel**: a shared frozen base, parallel exploration, then convergence into a single ensemble.

- **Everything shared across the team is built once and frozen:** cleaned data, folds, metric, prediction format. Changing folds after Day 3 invalidates all OOFs and breaks ensembling.
- **Work is split by model family, not by hyperparameter.** Ensemble value comes from diversity (§0.4): four people tuning variants of the same backbone yield effectively one model.
- **Seeds are not a feature to tune.** Fix seeds for ablations; use multiple seeds only to measure variance (early) and strengthen the final ensemble (late).

| Role                                                      | Owner                                                         | Responsibilities                                                                                                          |
| --------------------------------------------------------- | ------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------- |
| **Data & Validation Lead** (later **Ensemble Lead**)      | **\_\_\_\_**                                                  | EDA, cleaning, folds, official metric, adversarial validation; from mid-challenge onward collects OOFs, blends, and tunes post-processing |
| **Family A**                                              | **\_\_\_\_** (e.g., fine-tuned pretrained backbone)           | Explores and trains assigned model family                                                                                 |
| **Family B**                                              | **\_\_\_\_** (e.g., alternative architecture / self-supervised) | Same                                                                                                                    |
| **Family C**                                              | **\_\_\_\_** (e.g., classical/non-neural: GBDT, Rocket)       | Same; provides ensemble diversity                                                                                         |

Assign **roles** on **Day 0**; confirm the **3 specific families** on **Day 1** right after reviewing the rules (§1.1) and initial data inspection (§3). Two people must not work on the same family.

### 1.3 Shared Repository (GitHub) and Day 0 Setup

A **private** repository is the single source of truth for code and frozen artifacts: it maintains a single `an2dl_toolkit.py` (with the critical fixes from §6.1), distributes `train_folds.csv`, `cleaning_ids.csv`, and the metric without version drift, and logs the commit hash of every run (§1.4). One person sets it up on Day 0 (~1 hour) and immediately verifies on Colab/GPU that **`smoke_test.py` passes** (catching PyTorch/`timm`/`torchvision` version mismatches on Day 0 rather than Day 3). If some teammates avoid Git, the minimum shared core is `src/` + `shared/` + `runs/`.

**What does NOT go into the repository**

- **Raw data and images:** keep in a shared drive (e.g., Google Drive); competition rules may also restrict redistribution.
- **Weights and checkpoints** (`.pt`, `.pth`): too large and change every run.
- **Tokens, keys, and credentials** (Kaggle, Codabench, HuggingFace): never commit; use Colab Secrets.
- **OOF and test logits** (`.npy`): store in the shared drive using the naming contract in §1.4. Keep only run metadata in the repo.

**Structure**

```
an2dl/
├── README.md              # run instructions, team rules, link to Playbook
├── PLAYBOOK.md
├── requirements.txt       # pinned versions (torch, timm, lightgbm, ...)
├── .gitignore             # data/, outputs/, *.pt, *.pth, *.npy, .env
│
├── src/                   # shared: modified only by the lead
│   ├── an2dl_toolkit.py
│   ├── metric.py          # official metric (single implementation)
│   ├── data.py            # shared dataset and loaders
│   └── submit.py          # generates submission + checks from §5.4
│
├── shared/                # small files FROZEN on Day 3
│   ├── train_folds.csv
│   └── cleaning_ids.csv
│
├── eda/                   # EDA and cleaning notebooks (lead)
│
├── families/              # one folder per person/family
│   ├── A_<name>/          # train.py + config (yaml) + optional notebook
│   ├── B_<name>/
│   └── C_<name>/
│
├── runs/                  # one small json per run, e.g. A_convnext_s42.json
│
├── ensemble/              # blend.py, post-processing, OOF vs test report
└── tests/
    └── smoke_test.py
```

`outputs/` (OOFs, test logits, checkpoints) lives locally, is gitignored, and points to the shared drive.

**Operating rules**

1. **Only the lead edits `src/` and `shared/`.** Teammates open issues or pull requests so the toolkit does not diverge.
2. **Day 3 freeze gate:** create tag `freeze-day3`. Modifying `shared/` after this tag requires unanimous agreement.
3. **One branch per person** (`famA/...`); merge into `main` only what others need.
4. **Every run logs its commit** (`git rev-parse HEAD`). Runs with uncommitted changes are marked `"dirty": true` and excluded from the final ensemble.
5. **Configs in files, not notebook cells:** store hyperparameters in YAML or CLI flags for reproducibility.

**Notebooks and Colab**

- Notebooks cause merge conflicts and bloat git history with outputs: use `nbstripout`, or keep logic in `train.py` and use the notebook only as a launcher.
- One notebook per person; never have two people edit the same notebook.
- On Colab: run `git clone` or `git pull` at session start (personal token in Colab Secrets), read data from shared drive, and commit/push before closing (Colab disk is ephemeral).

**What not to do**

- Do not spend Day 0 over-engineering repo structure: the skeleton above (~1 hour) is enough.
- Do not add CI pipelines, branch protection, or mandatory PR reviews for a 14-day sprint; run `smoke_test.py` manually before modifying `src/`.

### 1.4 File Contract and Run Registry

Everyone must output files with the exact same names and shapes for ensembling to work.

| File                    | Contents                                                                                 |
| ----------------------- | ---------------------------------------------------------------------------------------- |
| `train_folds.csv`       | Shared fold split (single file, frozen on Day 3)                                         |
| `cleaning_ids.csv`      | Removed or flagged sample IDs with reasons                                               |
| `oof_logits_{run}.npy`  | Shape `(N, C)`; OOF logits ordered like `train_folds.csv`, using the same TTA as test    |
| `test_logits_{run}.npy` | Shape `(K, N_test, C)`; one model per fold, **not pre-averaged**                         |
| `runs/{run}.json`       | ID, family, change description, per-fold & OOF CV, leaderboard, seed, runtime, commit    |

`{run}` follows a fixed pattern, e.g., `family_model_seed` (`A_convnext_s42`).

**Run registry (`runs/` + aggregated table):** maintain a shared summary tracking ID, model, change vs. previous run, CV (single fold and full OOF), public leaderboard, seed, and runtime. Decisions are based on CV; the public leaderboard is a consistency check, not the target. Storing **one `.json` file per run in `runs/`** avoids merge conflicts across 4 teammates; the lead aggregates them into a table via script. Example:

```json
{
  "run": "A_convnext_s42",
  "family": "A",
  "owner": "name",
  "change": "resolution 64 -> 128 vs A_convnext_base",
  "seed": 42,
  "cv_folds": [0.812, 0.805, 0.82, 0.809, 0.815],
  "cv_oof": 0.812,
  "leaderboard": 0.8,
  "time_min": 38,
  "commit": "<git rev-parse HEAD>",
  "dirty": false
}
```

### 1.5 Four-Phase Schedule (Days 0 → 14)

| Days       | Phase                                  | Deliverables                                                                                            |
| ---------- | -------------------------------------- | ------------------------------------------------------------------------------------------------------- |
| 0–1        | Rules, roles, and setup                | §1.1 checklist completed, repo + `smoke_test.py` verified, families assigned on Day 1                   |
| 1–3        | Shared base                            | EDA + adversarial validation, `train_folds.csv`, `metric.py`, 5-fold baseline, OOF cleaning, 1st submit |
| **3**      | **Freeze gate**                        | Folds, metric, and `cleaning_ids.csv` frozen (`freeze-day3`); changes require unanimous agreement       |
| 3–9        | Parallel family exploration            | Run registry, single-variable ablations, **ensemble dry-run (Day 5–6)** to verify OOF alignment         |
| 9–12       | Convergence (+ pseudo-label/distill.)  | 1–2 final configs per family on all folds with TTA + multi-seed, teacher blend (D9–10), optional pseudo-labeling/distillation (D10–12) |
| **12**     | **Training cutoff**                    | No new models enter unless fixing a critical bug                                                        |
| 12–14      | Ensembling and submission              | Calibration, Caruana, nested post-processing, pre-submission checks (§5.4), 2 submissions selected on CV |

#### Phase 0 — Days 0–1: Rules, Setup, and Roles

Complete the §1.1 checklist together (pretrained weights, non-neural models, unlabeled test usage, submission and inference limits, delivery format). These answers shape the plan: e.g., a code submission with tight runtime limits makes Plan B (§5.3) immediately relevant. On Day 0, set up the repository (§1.3) and verify `smoke_test.py` on GPU; on Day 1, after inspecting the data (§3), confirm the 3 model families.

#### Phase 1 — Days 1–3: Shared Base

Executed once in strict order to prevent rework and leakage:

1. **Structural EDA and pre-split cleaning (§2.2, part 1):** corrupted samples (`NaN`/`Inf`, variance $\approx 0$), anomalous lengths/resolutions, groups, and near-duplicate detection (duplicates must stay in the same fold).
2. **Adversarial validation (§2.3):** check for *covariate shift* between train and test before locking folds, so you can stratify by test-similarity or drop shift-inducing features.
3. **Folds (`train_folds.csv`) and official metric (`metric.py`) (§2.4):** generate the shared split using the right strategy (groups, time blocks with purge gap, etc.) and unit-test the metric on a known case.
4. **Simple 5-fold baseline and first submission:** verify immediately that local CV tracks the public leaderboard.
5. **Post-baseline OOF cleaning (§2.2, part 2):** use the baseline's highest-loss OOF samples to manually inspect potential label errors or outliers; record decisions in `cleaning_ids.csv` (applied **only to training folds**, keeping validation intact).

**Day 3 freeze gate (`freeze-day3`).** Folds, metric, and cleaning list are frozen. Resolve any CV ↔ leaderboard mismatch now, not on Day 12.

#### Phase 2 — Days 3–9: Parallel Exploration

Each owner explores their assigned family under shared rules:

- **Measure noise first.** Run the family baseline with 3 seeds on the same fold and record seed standard deviation. Adopt a change only if its gain clearly exceeds this noise (roughly $> 2\times$ seed std).
- **One-variable-at-a-time ablations** on 1–2 folds with reduced epochs or resolution to save GPU. Confirm top candidates on all 5 folds, since single folds are noisy.
- **Output contract (§1.4):** every kept run saves OOF and test logits in the standard format, using identical TTA on OOF and test.
- **Mid-challenge ensemble dry-run (Days 5–6):** as soon as each family has a full 5-fold baseline, the ensemble lead runs `full_nested_postprocessing` end-to-end to catch shape, ID-order, or scale mismatches early.
- **Shared run registry (§1.4):** log every run in `runs/{run}.json`.
- **Short daily sync** (10–15 min or written update): what was tried, what worked, what is blocking. Share generalizable wins (augmentations, resolution, loss) immediately across families.
- **Leaderboard is a consistency check,** not a target (§0.1). Log CV–leaderboard pairs; if they diverge, diagnose why (split, shift) before proceeding.

#### Phase 3 — Days 9–12: Convergence (and Optional Pseudo-Labeling / Distillation)

- Each family owner selects **at most 1–2 final configurations** and trains them on **all folds with full TTA** at full resolution/epochs.
- Only now add multiple seeds to top configurations to strengthen the ensemble.
- The ensemble lead verifies OOF alignment (same folds, IDs, metric) and checks **error correlation** across models (highly correlated models add little).
- **Initial teacher blend by Days 9–10:** run Caruana to quantify each family's contribution. If using **per-fold pseudo-labeling (§4.7)** or **distillation under time/VRAM limits (§5.3)**, this blend serves as the *teacher* to generate per-fold soft labels, leaving **Days 10–12** for *student* retraining before the Day 12 cutoff.

#### Phase 4 — Days 12–14: Freeze and Submission

- **Day 12:** training stops.
- **Day 13:** calibration, Caruana selection, and nested post-processing (`full_nested_postprocessing`), plus pre-submission checks (§5.4).
- **Day 14:** select final submissions with buffer time. Pick **two distinct submissions**: the best CV blend and a conservative version (simple average without class multipliers) to hedge against overfitting.

---

## 2. Days 1–3: Data, Cleaning, and Anti-Leakage Validation

### 2.1 Reproducibility and DataLoader

Fixing seeds (Python, NumPy, PyTorch, DataLoader workers) makes runs *comparable*, not bit-for-bit identical: some GPU operations remain non-deterministic. Full deterministic mode slows training and should be reserved for exact reproduction.

**DataLoader.** With `num_workers > 0`, workers respawn every epoch unless persistent workers are enabled (which saves seconds to tens of seconds per epoch on image datasets). Use `pin_memory` on GPU and `drop_last=True` in training (a trailing 1-sample batch breaks BatchNorm and MixUp). Tune `num_workers` to the machine (2 is a reasonable starting point on 2-vCPU Colab free). For small-resolution datasets that fit comfortably in GPU VRAM as `uint8` tensors (e.g., 32×32 or 64×64 images), keeping images on GPU and running normalization/vectorized augmentations per batch eliminates CPU DataLoader bottlenecks.

### 2.2 Data Inspection and Cleaning (Pre-Split and Post-Baseline OOF)

Dataset quality checks happen at **two distinct points** in Phase 1:

1. **Pre-split (Day 1, before creating folds):**
   - **Corrupted samples:** check for near-zero variance (flat/constant images or series), `NaN`/`Inf`, anomalous resolutions or sequence lengths.
   - **Duplicates and near-duplicates:** detect via perceptual hashing or backbone embeddings. Duplicates with **conflicting labels** are not automatically synthetic noise—they may be genuinely ambiguous cases and must be **inspected manually**, not auto-dropped. Keep all remaining duplicates in the same fold (§2.4).
2. **Post-baseline (Days 2–3, after running the initial 5-fold baseline):**
   - **Embedding outliers and highest-confidence OOF errors:** rank training samples by distance to class centroids / test set in embedding space (e.g., DINOv2 or MiniRocket) and inspect the baseline's highest-loss OOF errors (plus per-class accuracy and top confusion pairs). For vision tasks, run **CAM / Grad-CAM** on high-loss or suspiciously easy samples to check whether the model relies on spurious background artifacts, borders, or watermarks instead of the object. Use this strictly as a **ranking for manual inspection**, never as a blind filter (which can mistake domain shift for garbage).

**CV caution:** removing outliers from validation folds makes CV optimistic and unrepresentative of a test set that may contain similar samples. **Clean only the training split** of each fold (logging IDs in `cleaning_ids.csv`) and report validation metrics on both cleaned and full validation (using full validation as the reference).

### 2.3 Adversarial Validation

Train a classifier to distinguish train from test **before finalizing the split**. If AUC is clearly above chance (roughly > 0.6), *covariate shift* is present. Mitigations:

- inspect top distinguishing features (sequential IDs, timestamps, sensor offsets) and remove or normalize them;
- build a validation split that better matches test (stratify on "similarity to test");
- as a last resort, weight training samples by the density ratio (corrected for the train/test size ratio).

For images and time series, feed backbone embeddings rather than raw pixels/samples.

### 2.4 Building the Split (`train_folds.csv`) and Local Metric

Leakage rarely comes from the scaler; it comes from the split.

| Scenario                                                                | Correct Split                                                                                                                                    |
| ----------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------ |
| Independent samples                                                     | Stratified K-Fold on target class                                                                                                                |
| Multiple samples per subject / patient / device / acquisition           | **Group** K-Fold (stratified + grouped)                                                                                                          |
| Sliding windows extracted from long series                              | Split parent series into **contiguous time blocks** with a *purge gap* ≥ window length first, then extract windows inside each split            |
| Forecasting                                                             | **Rolling-origin** (expanding train, future validation) with gap ≥ forecast horizon                                                             |
| Near-duplicate images                                                   | Detect duplicates (perceptual hash or backbone embeddings) and place them in the same fold                                                       |

Related rules:

- **One shared fold file** (`train_folds.csv`) across the team so OOF predictions from different models align and combine cleanly.
- **Fit preprocessing only on the train fold** (scalers, quantile transformers, target encoding, imputation).
- **Local metric equals official metric:** implement on Day 1 and use everywhere.
- **Always save OOF and test predictions** (logits, not hard labels; save test logits per fold with shape `(K, N_test, C)`).

---

## 3. Days 1–9: Problem Map (Classical to Advanced)

Reference for choosing families on Day 1 (§1.2) and iterating during Days 3–9. For each task: **(a)** classical baselines, **(b)** strong modern default, **(c)** advanced additions, **(d)** common pitfalls.

### 3.1 Tabular Data — Classification and Regression

**(a) Classical.**

- *Linear/Logistic Regression* (L1/L2): interpretable baseline, requires scaled features; useful sanity check and ensemble member.
- *k-NN, SVM*: sensitive to scale and dimensionality; rarely winning today, useful on small datasets.
- *Random Forest*: robust, minimal tuning, solid nonlinear baseline.
- *MLP*: works, but rarely beats trees on tabular data without specialized components.

**(b) Strong default.** Tree-based Gradient Boosting (LightGBM, CatBoost, XGBoost) with sensible feature engineering and moderate tuning. Remains the tabular benchmark; CatBoost handles categoricals well.

**(c) Advanced.**

- Modern tabular networks for ensemble diversity: periodic numerical embeddings (PLR), architectures such as TabM/RealMLP, gated blocks (SwiGLU). Value comes primarily from *diversity* relative to trees.
- Pretrained tabular models (TabPFN family) on small datasets (few thousand rows): verify version and constraints.
- Feature Gaussianization (`QuantileTransformer`) for neural nets: optional, not mandatory; can hurt on small sample sizes (cap `n_quantiles`). Decide via CV.

**Regression specifics.**

- Match loss to metric: MSE for RMSE, MAE/Huber when outliers exist.
- For positive skewed targets, test log-transform (and invert at inference).
- Inspect residuals and error across target ranges; clip predictions to plausible bounds.
- Ensemble diverse model families via averaging or simple stacking.

**(d) Pitfalls.** Target leakage in features; fitting scalers on full data; computing categorical target encodings outside the fold; optimizing the wrong metric.

### 3.2 Images — Classification

**(a) Classical (and from-scratch rules).** CNNs from scratch (VGG-style 3×3 blocks → ResNet / pre-activation WideResNet, DenseNet, ResNeXt, MobileNet/MBConv+SE) using **Global Average Pooling (GAP)** instead of heavy Flatten+MLP heads, BatchNorm, dropout, and basic augmentations. Good for understanding the problem, fast baselines on small images, or competitions that forbid pretrained weights:

- on natively small images (e.g., 32×32 / 64×64), prefer a 3×3 stride-1 stem without aggressive early downsampling;
- **do not pick blocks by parameter count or MACs alone:** depthwise-expanded or concatenated blocks (MBConv, DenseNet) can have few parameters yet large activation memory (VRAM) and lower real GPU throughput than simpler residual blocks—benchmark actual images/sec with `torch.cuda.synchronize()`.

**(b) Strong default.** *Transfer learning* from pretrained backbones (ResNet/EfficientNet, ConvNeXt/ConvNeXt V2, Swin, ViT, DINOv2) with fine-tuning and a well-chosen learning rate:

- test direct fine-tuning (AdamW, warmup + cosine decay, label smoothing) and, on small datasets, **two-stage transfer learning** (brief head warm-up with frozen backbone, then unfreezing the last stages or full backbone at a 10–100× lower LR — see §4.4);
- domain-appropriate augmentations across geometric, photometric, and random-erasing families (exclude label-altering transforms such as flips on digits or asymmetric anatomy);
- resolution: moderate upscaling on natively small images (e.g., 64→128/224) often helps by matching pretraining scale. It adds no information, so **measure** in CV and avoid extreme resolutions that waste VRAM;
- **explainability check:** inspect **CAM / Grad-CAM** overlays on validation errors and top-confidence predictions to catch shortcut learning (background, borders, scanner stamps) early.

**(c) Advanced.**

- *Layer-wise LR decay* (LLRD): lower learning rates for early layers. Most useful for ViTs and large backbones on limited data.
- *Mixup / CutMix*: strong regularization, helpful with many epochs or small data; less effective when classes depend on fine local details.
- *Weight EMA*, *TTA*, *diverse architecture ensembles* (pure CNN + hybrid + self-supervised ViT).
- Self-supervised backbones (DINOv2): often strong on non-ImageNet domains, but domain-dependent; verify in CV.
- *GeM pooling*: replaces average pooling; appropriate on CNNs with non-negative activations (ResNet, some EfficientNets). On ConvNeXt, CAFormer, Swin, and ViT, keep native pooling to preserve pretrained final normalization.

**(d) Pitfalls.** Evaluating with augmentations enabled; normalizing with stats different from the backbone's pretraining stats; picking backbones too large for available VRAM; failing to fix folds across runs.

### 3.3 Images — Semantic Segmentation

**(a) Classical.** U-Net (encoder-decoder with skip connections); FCN. Reliable baseline especially on medical and industrial imagery.

**(b) Strong default.** U-Net/U-Net++/FPN with pretrained encoders (`segmentation_models_pytorch`), or transformer models (SegFormer, UPerNet). Composite loss: Dice + (Cross-Entropy or Focal), because pixel-wise CE alone struggles with small classes.

**(c) Advanced.**

- Lovász-Softmax in the final epochs as a differentiable IoU surrogate.
- Diverse architecture ensembles; flip TTA (if orientation is not semantic).
- Mask post-processing (small-component removal, hole filling), validated in CV.
- **Weakly supervised segmentation / ROI cropping:** when only image-level labels are available, a GAP + linear classifier (equivalently a Network-in-Network 1×1 conv + GAP head) yields **Class Activation Maps (CAM)**, and **Grad-CAM** generalizes to any intermediate stage (`H4` in Part II). Thresholding upsampled maps gives rough localization masks or foreground crops.

**(d) Pitfalls.** Failing to upsample SegFormer logits (1/4 resolution) to mask resolution before loss and blending; incompatible encoder-decoder pairs (stride/stage count); augmenting image without mask.

### 3.4 Time Series — Classification

**(a) Classical.** Hand-crafted features (summary stats, spectral) + classical model; k-NN with DTW distance; RNNs (LSTM/GRU) with mean or last hidden state. Pure RNNs on long noisy sequences are often slow and unstable.

**(b) Strong default.**

- *Rocket / MiniRocket / MultiRocket / HYDRA* + linear classifier: trains in seconds on CPU, often competitive with deep nets on small datasets, and adds strong ensemble diversity (errors differ from neural nets, though not strictly independent).
- *Multi-scale 1D CNNs* (InceptionTime / 1D ResNet families) with global pooling.

**(c) Advanced.**

- Attention pooling and concatenated avg + max pooling instead of temporal mean alone (which dilutes brief events).
- Domain augmentations: jitter, scaling, time masking, time warping, window slicing.
- Rocket + 1D CNN ensemble.
- **Per-series aggregation (when there is one label per series):** if using sliding windows, the official metric remains at the series level; aggregate window logits per series (`aggregate_window_logits`) in validation and test before computing metrics or calibrating. With per-timestep labels, taking the last timestep matches causal/streaming use; offline, you can test majority vote, center timestep, or dropping transition-straddling windows (purity threshold).
- **RevIN in classification:** use only if absolute amplitude and offset do *not* discriminate classes; otherwise it destroys signal (alternatively, pass window mean and std as auxiliary features).

**(d) Pitfalls.** Overlapping windows across train and validation (leakage); ignoring subject/group boundaries; scoring individual windows instead of aggregating logits per series; feeding variable-length sequences without consistent padding/masking.

### 3.5 Time Series — Forecasting and Regression

**(a) Classical.** Naive baselines (last value, seasonal naive), ARIMA/ETS, lag regression. **Always compute naive baselines**: they are often surprisingly hard to beat.

**(b) Strong default.**

- *LightGBM with lag features* (lagged values, rolling means/stds, calendar features).
- Linear decomposition models (DLinear): fast and frequently competitive.
- Well-regularized recurrent or convolutional nets.

**(c) Advanced.**

- *PatchTST* (patching + channel-independent transformer) and newer architectures; performance varies by dataset and horizon—test both DLinear and PatchTST.
- *RevIN* to mitigate window distribution shift; optional, validate in CV.
- Time-series foundation models (Chronos/TimesFM families) as baselines or ensemble members if allowed by rules; check versions and licenses.
- Diverse ensembles (GBDT + linear + neural).

**(d) Pitfalls.** Random splits; insufficient train–val gap (train window targets leak into validation period); normalizing with future statistics; **extracting windows across the full series before splitting** (windows must be built inside each already-separated segment); comparing `[B, H, 1]` predictions with `[B, H]` targets (see §6.1).

### 3.6 Other Problem Types (Brief Notes)

- *Image retrieval / metric learning*: extract L2-normalized pooled backbone embeddings and rank gallery items by cosine similarity, evaluating with `Precision@k` / `mAP`; fine-tune with metric losses if retrieval is the official task. The same normalized embeddings serve §2.2 near-duplicate detection and outlier ranking.
- *Sequences / text*: TF-IDF + linear baseline first, then fine-tuned pretrained RNN/Transformer.
- *Anomaly detection*: reconstruction (autoencoders), feature-space distances, one-class models; validation requires extra care because anomalies are rare.
- *Image regression*: same setup as classification with a regression head, suitable loss (Huber/MAE), and target normalization.

---

## 4. Days 3–12: Training Techniques and Exploration Priorities

### 4.1 Technique Impact vs. Cost (Priority Compass)

Qualitative, *indicative* ratings: actual gains are dataset-dependent and must be verified in CV. Use this table to prioritize work during Days 3–12.

| Technique                                     | Typical Gain                 | Cost              | Risk                     | When to Try                             |
| --------------------------------------------- | ---------------------------- | ----------------- | ------------------------ | --------------------------------------- |
| Proper validation (groups, blocks, purge)     | **Very high** (avoids traps) | Low               | Low                      | Always, Day 1                           |
| Appropriate pretrained backbone               | High                         | Medium            | Low                      | Images, converted signals               |
| GBDT with strong features                     | High (tabular, forecasting)  | Low               | Low                      | As baseline                             |
| Rocket + linear classifier                    | Medium-high (small series)   | Very low          | Low                      | Before deep networks                    |
| Domain augmentation                           | Medium-high                  | Low               | Medium (label change)    | Always, carefully                       |
| Diverse family ensemble                       | Medium                       | Medium            | Low                      | Late stage (dry-run D5–6, blend D9+)    |
| Fold averaging + TTA                          | Medium                       | Low               | Low                      | Always                                  |
| LLRD                                          | Low-medium                   | Low               | Low                      | Fine-tuning large backbones             |
| EMA (with BN buffers)                         | Low-medium                   | Low               | Low                      | Long training runs (many steps)         |
| Mixup/CutMix                                  | Variable                     | Low               | Medium                   | Many epochs, limited data               |
| Temperature scaling + Caruana                 | Low-medium                   | Low               | Low                      | Final blend                             |
| Class multipliers (Macro-F1)                  | Low, sometimes zero          | Medium            | **Medium-high** (overfit)| Only if nested CV gain passes threshold |
| Per-fold pseudo-labeling                      | Low-medium                   | **High**          | Medium (leakage, rules)  | Days 10–12 (before freeze), if allowed  |
| SAM                                           | Low                          | High (2× compute) | Medium                   | Only with spare compute                 |
| "Exotic" activations                          | Negligible                   | Low               | Low                      | Almost never                            |

### 4.2 Optimization

- **AdamW** default; exclude biases and normalization parameters from weight decay.
- **Linear warmup + cosine decay**, with gradient clipping (norm 1.0 is a solid default).
- **Mixed precision:** use `fp16` + `GradScaler` on Turing GPUs (T4); native `bf16` without scaler on Ampere+ (L4, A100). Emulated `bf16` on T4 is slow.
- Batch size interacts with learning rate; changing both at once confounds ablations.
- **Scheduler units.** PyTorch schedulers count `step()` calls, not epochs. When calling `step()` per *optimizer step*, set total duration in steps (`epochs × steps_per_epoch`); passing `T_max = EPOCHS` finishes the cosine cycle after a few batches. Pick one unit and use it consistently across scheduler, warmup, and EMA.
- **Gradient accumulation.** With large backbones on small GPUs (T4, high resolution), micro-batches may be 8–16 samples. Accumulating gradients across micro-batches reduces gradient noise and simulates a larger batch, though MixUp/CutMix mixing and BatchNorm statistics still operate at micro-batch size. With accumulation, scheduler and EMA advance once per optimizer step.

### 4.3 Regularization and Augmentation

- Dropout, weight decay, small label smoothing (e.g., 0.05–0.1).
- **Domain-consistent augmentations:** never apply transforms that alter the label. Organize image transforms by family — **geometric** (flips, affine), **photometric** (color jitter), and **random erasing / cutout** — and ablate them systematically rather than stacking every transform blindly (over-augmenting small datasets can degrade CV).
- Mixup/CutMix require losses supporting 2D soft targets; designed for classification, not directly for segmentation or time series.

### 4.4 Transfer Learning

- Start from pretrained weights; full fine-tuning with a lower LR than training from scratch, or **two-stage transfer learning** (freeze the backbone to warm up the randomly initialized classifier head first, then unfreeze the top stages or full backbone at a 10–100× lower LR) when data is limited.
- **Freezing correctly in PyTorch:** setting `requires_grad_(False)` alone is not enough—calling `model.train()` puts frozen stages back into training mode, mutating `BatchNorm` running statistics and keeping `Dropout`/`StochasticDepth` active. Override `train()` (or re-call `.eval()` on frozen stages after `model.train()`). On small batches or ImageNet-like domains, also test keeping `BatchNorm` frozen (`bn.eval()`) during fine-tuning (`H3` in Part II).
- LLRD to restrict early-layer movement.
- In offline environments, download checkpoints locally and load them as datasets.

### 4.5 Class Imbalance, Loss Functions, and Metrics

Tools, simplest first: stratified folds, balanced sampling, class weights, *logit adjustment*, focal losses.

- They serve different purposes: e.g., PolyLoss is not specifically an imbalance method.
- **Do not double-correct the prior:** if the loss already shifts boundaries toward rare classes, applying class multipliers in post-processing risks over-correcting. Pick one strategy or validate the combination.
- The metric dictates the objective: accuracy favors calibrated probabilities; balanced accuracy and Macro-F1 benefit from rare-class adjustments.
- **Loss vs. metric:** when the metric is non-differentiable (F1, IoU), train with a smooth surrogate loss and optimize the target metric **afterward** in validated post-processing.

### 4.6 Model Stabilization (EMA, Multi-Seed, SWA)

- **Weight EMA:** maintains a moving average of weights for validation and inference. With BatchNorm, also average *buffers* (`running_mean`/`running_var`), otherwise EMA evaluates averaged weights against stale stats. Even time-averaged BN buffers can drift slightly from EMA weights (especially under MixUp/CutMix): recomputing BN stats on one epoch of clean images (*precise BN*, `torch.optim.swa_utils.update_bn`) takes seconds and can be validated in CV (not needed for `LayerNorm` architectures like ConvNeXt/ViT/EVA). With few total steps, lower decay or warm it up.
- **Multi-seed / fold averaging:** averaging the 5 fold models at test time is nearly free and reliable.
- SWA as an end-of-training alternative.

### 4.7 Semi-Supervised Learning and Pseudo-Labeling (Days 10–12)

Useful when rules allow it and unlabeled data is plentiful. Because it requires retraining before the Day 12 cutoff, close the initial *teacher* blend by **Days 9–10** and reserve **Days 10–12** for student retraining.

- **Subtle leakage:** generating pseudo-labels from an ensemble containing models that trained on the target validation fold makes subsequent CV optimistic. Fix: generate **per-fold pseudo-labels** using only models that never saw that fold's validation split.
- A high fixed confidence threshold (e.g., 0.98) selects very few samples poorly under Mixup and label smoothing: prefer per-class selection (top-q) or calibrated *soft labels* with a reduced loss weight.
- Expect modest gains at the cost of a full retraining cycle.

### 4.8 Techniques to Treat with Skepticism

- **"Exotic" activations** (E-Swish, etc.): with a learnable multiplicative parameter, the following layer or normalization absorbs the scale, making them behave like SiLU. Activation choice has minimal leverage; GELU/SiLU/SwiGLU suffice.
- **SAM (Sharpness-Aware Minimization):** doubles training compute, requires a two-pass loop and disabling BN stat updates on the second pass; try only after exhausting higher-leverage options.
- **Thresholds tuned on full OOF:** overfit easily; validate via nested CV.

---

## 5. Days 9–14: Inference, TTA, Ensembling, and Submission

### 5.1 Test-Time Augmentation (TTA)

Average predictions across input views (flips, mild shifts/zooms; on **orientation-invariant domains** such as microscopy or aerial imagery, test 90° rotations and full flip+rotation D4 views). Always **select the TTA view set on validation CV**, use only label-preserving transforms, and do not change resolution on ViT/EVA/DINOv2 (positional embeddings) or CNNs if dimensions break total stride divisibility.

- **OOF ↔ Test consistency:** generate OOFs **with the same TTA used on test** (`predict_with_tta`); applying TTA only at test time creates a logit-scale mismatch against the OOFs used to fit temperatures and multipliers.

### 5.2 Ensembling, Calibration, and Nested Post-Processing

1. **Calibrate** each model (temperature scaling on OOF).
2. **Select weights** by minimizing *log-loss* (smooth) or via Caruana selection (greedy hill-climbing with replacement). Direct gradient optimization of Macro-F1 fails because the objective is a step function.
3. **Post-processing** (e.g., class multipliers for Macro-F1) via coordinate ascent and **nested validation**: fit all steps (temperatures, weights, multipliers) on $K-1$ training folds and evaluate on the held-out fold to avoid optimistic bias.
4. Apply multipliers only if nested CV gain exceeds a noise threshold (`min_f1_gain`).
5. Watch for *scale mismatch*: averaging logits across models or folds reduces logit norm relative to the single-fold OOF logits used for calibration. Save `test_logits_{model}.npy` as `(K, N_test, C)` (one model per fold, unaveraged), apply `mean_temps[m]` to each fold model, and average probabilities afterward.

### 5.3 Plan B for Code Submissions with Strict Time/VRAM Limits

If 3 architectures × 5 folds × TTA exceed the inference timeout, you **cannot** apply Caruana to individual fold checkpoints (each checkpoint is OOF only on its own fold and trained on the remaining 80%). Valid alternatives:

- select architectures via Caruana on full OOFs, then reduce the number of folds per architecture (or retrain one full-data model per architecture) and TTA views at inference, measuring the cost in CV;
- or **distill** the calibrated OOF ensemble probabilities (and unlabeled test if rules permit) into a single fast student model during **Days 10–12** (before training cutoff), verifying in CV.

### 5.4 Pre-Submission Checks

- Format, row count, ID ordering.
- Predicted class distribution vs. expected distribution.
- Compare OOF vs. test mean confidence (`mean(max_c p_c)`) and predicted class balance: investigate large discrepancies before submitting, especially when applying class multipliers.
- Choose final submissions based on CV, never on the public leaderboard alone.

---

## 6. Error Catalog

### 6.1 Silent Technical Bugs Already Encountered

Collected across previous iterations of this playbook; all degrade performance **without raising errors**.

1. **Hand-rolled LLRD** missing `patch_embed`, `cls_token`, `pos_embed`, stem, or final norms, or applying weight decay to biases and norm layers.
2. **LLRD silently ignored by PyTorch schedulers:** `timm`'s optimizer factory stores the multiplier in `lr_scale`, which only `timm` schedulers read; PyTorch schedulers ignore it. Multiply `group['lr']` by `group['lr_scale']` before creating the scheduler.
3. **EMA not updating BatchNorm buffers** (`AveragedModel` without `use_buffers=True`).
4. **Optimizing Macro-F1 with SLSQP or Nelder-Mead:** zero or brittle gradients.
5. **GeM on ConvNeXt/ViT:** truncates negative activations and bypasses final normalization.
6. **`bf16` on T4 GPUs:** emulated and slow; use `fp16` + `GradScaler`.
7. **Pseudo-labels generated by models that trained on the target validation fold.**
8. **`log_loss` crashing** when a rare class is absent from a fold (pass explicit `labels=np.arange(n_classes)`).
9. **Assigning folds with label-based indices instead of `.iloc` positional indices** on filtered DataFrames.
10. **Time-series CV without a purge gap** and calling `torchvision.transforms.v2` Mixup/CutMix with non-existent args (`label_smoothing` belongs to `timm`, not `torchvision.transforms.v2`).
11. **Scheduler stepped in wrong units:** calling `step()` per batch with `T_max` set in epochs (or vice versa).
12. **Silent broadcasting in loss:** `[B, H, 1]` predictions vs. `[B, H]` targets broadcast to `[B, H, H]`, comparing every timestep with all others. PyTorch only emits a warning while the model converges to garbage. Check shapes explicitly.
13. **Wrong tensor layout for model family:** 1D CNNs expect `[B, C, L]`; DLinear and PatchTST expect `[B, L, C]`.
14. **Using the public leaderboard** to pick final submissions.
15. **OOF ↔ test scale mismatch:** saving fold-averaged `test_logits` instead of `(K, N_test, C)`, or applying TTA on test but not OOF.
16. **Dropping outliers from validation folds:** inflates CV and breaks correlation with noisy or ambiguous test samples.
17. **Freezing stages with `requires_grad_(False)` while calling `model.train()`:** frozen `BatchNorm` running stats keep drifting and `Dropout`/`StochasticDepth` stay active in frozen blocks. Explicitly set frozen stages back to `.eval()` after `model.train()`. Similarly, when computing **Grad-CAM** on a frozen stage, set `x.requires_grad_(True)` so autograd builds a graph through the frozen layers.

### 6.2 Common Process Mistakes

- **Everyone using different folds:** OOFs cannot be combined; ensembling is impossible.
- **Spending early days tuning model details** instead of data and validation.
- **Chasing the leaderboard:** with limited daily submissions and solid CV, it is just noise.
- **Leaving ensembling to the last day:** format mismatches are discovered too late.
- **Adding look-alike models:** ten variants of one backbone add less than three distinct families.
- **Settling disagreements by opinion:** when two choices conflict, CV decides.

---

## 7. Individual Checklists

Copy into your workspace and check off items. **[L]** items belong to the Data & Ensemble Lead.

### 7.1 Day 0 — Everyone

- [ ] Read rules and completed §1.1 checklist with the team.
- [ ] Confirmed whether pretrained weights, non-neural models, and unlabeled test data are permitted.
- [ ] Confirmed limits: daily submissions, inference runtime, model size, delivery format (file vs. code).
- [ ] Roles assigned and recorded in the shared doc (confirm specific 3 families on Day 1 after inspecting data).
- [ ] Access verified to GPU, shared repo, and run registry.
- [ ] Private repo created matching §1.3; everyone can `clone` and `pull` (including from Colab).
- [ ] `.gitignore` active: no data, weights, `.npy`, or tokens in repo.
- [ ] `smoke_test.py` passed on Colab/GPU with pinned `requirements.txt` versions.

### 7.2 Days 1–3 — Shared Base

- [ ] Confirmed the 3 model families (A, B, C) based on data (§3) and rules (§1.1).
- [ ] **[L]** Pre-split structural EDA completed: corrupted samples (`NaN`/`Inf`, variance $\approx 0$), anomalous resolutions/lengths, groups, and near-duplicates (§2.2).
- [ ] **[L]** Adversarial validation run and interpreted *before* generating the split (§2.3).
- [ ] **[L]** `train_folds.csv` created with the right strategy (§2.4); duplicates placed in the same fold.
- [ ] **[L]** Official metric (`metric.py`) unit-tested on a known case.
- [ ] Baseline reproduced across all 5 folds using the shared `train_folds.csv`.
- [ ] First submission made; CV ↔ leaderboard correlation recorded.
- [ ] **[L]** Post-baseline cleaning: manually inspected highest-loss OOF errors; logged decisions in `cleaning_ids.csv` applied **only to training** (validation left intact).
- [ ] **Freeze gate:** everyone confirmed in writing the freeze of folds, metric, and cleaning list.
- [ ] **[L]** Committed `train_folds.csv` and `cleaning_ids.csv` in `shared/` and created tag `freeze-day3`.

### 7.3 Days 3–9 — Exploration (Per Family)

- [ ] Family baseline run with 3 seeds on the same fold: seed std measured.
- [ ] One-variable ablations on 1–2 folds, then 5-fold confirmation for top candidates.
- [ ] High-leverage knobs tested before marginal ones (§4.1): backbone, domain augmentation, resolution, loss.
- [ ] My model differs from teammates' by family, not just hyperparameters.
- [ ] **[L] Mid-challenge dry-run (Days 5–6):** ran `full_nested_postprocessing` on the initial 5-fold baselines of all 3 families to verify ID alignment, folds, and `(N, C)` / `(K, N_test, C)` shapes.

### 7.4 Days 9–12 — Convergence (and Optional Pseudo-Labeling / Distillation)

- [ ] Selected 1–2 final configs for my family, justified by CV.
- [ ] Trained on **all folds** with TTA for both OOF and test.
- [ ] Added multiple seeds to top configs.
- [ ] Uploaded files matching §1.4 format to the shared drive.
- [ ] **[L]** Verified OOF alignment (IDs, folds, shapes).
- [ ] **[L]** Computed error correlation across models.
- [ ] **[L]** Initial *teacher* Caruana blend completed by **Days 9–10**.
- [ ] *(Optional, Days 10–12)* Per-fold pseudo-labeling (§4.7) or distillation (§5.3) completed before training cutoff.

### 7.5 Days 12–14 — Submission

- [ ] **Day 12:** training closed; no new models.
- [ ] **[L]** Calibration, Caruana, and nested post-processing run on OOFs.
- [ ] **[L]** Class multipliers applied only if nested gain exceeds `min_f1_gain`.
- [ ] **[L]** 3D test logits `(K, N_test, C)` passed to `full_nested_postprocessing`.
- [ ] Pre-submission checks (§5.4) completed (format, IDs, class distribution, OOF vs. test confidence).
- [ ] Selected **two** final submissions: best CV blend and conservative fallback.
- [ ] Submissions uploaded with time buffer before deadline.
- [ ] If code submission: inference time and memory verified against limits (Plan B from §5.3 if needed).

### 7.6 Before Every Run — Everyone

- [ ] Changing **one variable** relative to a reference run.
- [ ] Seed fixed and logged.
- [ ] Using `train_folds.csv` as-is (no fold redefinition).
- [ ] Pulled latest `main` and working on my own branch.
- [ ] Preprocessing fit strictly on the train fold.
- [ ] Scheduler, EMA, and gradient accumulation counted in the same unit (steps); LLRD applied to `lr` (§6.1).
- [ ] Verified input, output, and target tensor shapes before launch (§6.1, items 12 and 13).

### 7.7 After Every Run — Everyone

- [ ] Metric computed with the official function on single fold and full OOF.
- [ ] Compared against baseline **accounting for seed variance**.
- [ ] OOF and test logits saved in §1.4 format with identical TTA.
- [ ] Run logged in the run registry (change, CV, runtime, commit).
- [ ] `runs/{run}.json` created with commit hash and accurate `dirty` flag; code committed and pushed.
- [ ] Shared any winning insight in the daily sync.

### 7.8 Daily — Everyone

- [ ] Brief sync: tried, worked, blocked.
- [ ] Run registry updated before signing off.
- [ ] No decisions based solely on the public leaderboard.

### 7.9 Arbitration Rule

When two results or choices conflict, **nested CV or full-OOF CV decides**, never opinion or the public leaderboard.

---

**Next:** [AN2DL_Playbook_Part2_Code.md](./AN2DL_Playbook_Part2_Code.md)
