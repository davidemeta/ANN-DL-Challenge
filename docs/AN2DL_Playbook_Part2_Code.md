# AN2DL Challenge Playbook (v6.1) — Part II: Code Reference

Maps the modules implemented in [an2dl_toolkit.py](../src/an2dl_toolkit.py) and [smoke_test.py](../tests/smoke_test.py) (single source of truth to prevent drift) and keeps standalone modules outside the toolkit (**D** and **H**) inline.

- **Part I — Strategy and Techniques:** [AN2DL_Playbook_Part1_Strategy.md](./AN2DL_Playbook_Part1_Strategy.md)
- **Core Toolkit (blocks A, B, C, E, F, G):** [an2dl_toolkit.py](../src/an2dl_toolkit.py)
- **Test Suite (block I):** [smoke_test.py](../tests/smoke_test.py)

---

## Module Index

- [Verification Status](#verification-status)
- [Usage](#usage)
- [Not Included](#not-included)
- [A. Setup, Determinism, and Hardware](#a-setup-determinism-and-hardware) → `an2dl_toolkit.py`
- [B. Anti-Leakage Validation](#b-anti-leakage-validation) → `an2dl_toolkit.py`
- [C. Computer Vision](#c-computer-vision) → `an2dl_toolkit.py`
- [D. Semantic Segmentation (Standalone Block)](#d-semantic-segmentation-standalone-block) → _inline below_
- [E. Time Series](#e-time-series) → `an2dl_toolkit.py`
- [F. Tabular Data](#f-tabular-data) → `an2dl_toolkit.py`
- [G. Ensembling and Nested Post-Processing](#g-ensembling-and-nested-post-processing) → `an2dl_toolkit.py`
- [H. Additional Modules (Outside the Toolkit)](#h-additional-modules-outside-the-toolkit) → _inline below_
- [I. Smoke Test (`smoke_test.py`)](#i-smoke-test-smoke_testpy) → `smoke_test.py`

---

## Verification Status

| Block                                                                                                                    | Status                                                                                                                                                              |
| ------------------------------------------------------------------------------------------------------------------------ | ------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| B (folds, time blocks, forecasting CV, adversarial validation)                                                           | **Executed** on synthetic data (CPU); adversarial validation tested with a scikit-learn wrapper instead of LightGBM                                                 |
| G (temperature, Caruana, multipliers, nested post-processing)                                                            | **Executed** on synthetic data, including missing class in a fold and three identical models                                                                        |
| `FastSlidingWindowDataset`, `build_warmup_cosine_scheduler`, `steps_per_epoch`, `forecast_loss` (numerical logic only)   | **Executed with mocked `torch`**: window indices and boundaries, label alignment, scheduler curve, shape checks. Not tested on real PyTorch                         |
| A, C, E, F (`torch` / `timm` / `torchvision`), `make_loader`, `train_vision_epoch` with accumulation                     | **Not executed**: packages missing in test environment. Run [smoke_test.py](../tests/smoke_test.py) (block I)                                                               |
| D, H (segmentation, pseudo-label loss, logit adjustment, safe freezing, Grad-CAM)                                        | **Not verified**, not covered by the smoke test                                                                                                                     |

First things to check when launching on Colab:

- `v2.RandomChoice([MixUp, CutMix])` works when called directly on `(images, targets)` in the installed `torchvision` version;
- LLRD actually covers stem, `patch_embed`, and `cls_token` on ConvNeXt and ViT (the smoke test only verifies that param-group learning rates differ);
- model and pretrained weight names exist (`timm.list_models(pretrained=True)`).

## Usage

1. Blocks **A, B, C, E, F, G** live directly in [an2dl_toolkit.py](../src/an2dl_toolkit.py) (do not duplicate them in Markdown).
2. Block **I** lives in [smoke_test.py](../tests/smoke_test.py): run it from the repo root (`python tests/smoke_test.py`; it finds `src/an2dl_toolkit.py` automatically) (after `!pip install -q timm lightgbm` on Colab).
3. Blocks **D** and **H** are standalone modules outside the toolkit and remain inline below for copy-paste when needed.

## Not Included

- Complete LightGBM/CatBoost/XGBoost pipelines (the tabular part of the toolkit only provides the PLR + SwiGLU network to pair with trees).
- SAM (requires a two-pass loop and suspending BN stat updates on the second pass).
- TabM, RealMLP, TabPFN, time-series foundation models: integrate from their respective libraries after checking versions and rules.
- Superseded first-draft code (hand-rolled LLRD, `gather`-based PolyLoss, SLSQP on Macro-F1, generic GeM, E-Swish).

---

### A. Setup, Determinism, and Hardware

**Source:** [an2dl_toolkit.py (lines 1–76)](../src/an2dl_toolkit.py#L1-L76) · **Strategy:** [Part I §2.1](./AN2DL_Playbook_Part1_Strategy.md#21-reproducibility-and-dataloader)

Shared imports, GPU detection (`fp16` + `GradScaler` on T4, native `bf16` without scaler on Ampere+), and `make_loader` (persistent workers, `pin_memory`, `drop_last` in training).

**Symbols and signatures:**
- Globals: `SEED = 42`, `DEVICE`, `AMP_DTYPE, USE_SCALER = get_amp_config()`
- `seed_everything(seed=42, deterministic=False)`
- `seed_worker(worker_id)`
- `get_amp_config()`
- `make_loader(dataset, batch_size, is_train, num_workers=2, seed=SEED)`

---

### B. Anti-Leakage Validation

**Source:** [an2dl_toolkit.py (lines 79–149)](../src/an2dl_toolkit.py#L79-L149) · **Strategy:** [Part I §2.3](./AN2DL_Playbook_Part1_Strategy.md#23-adversarial-validation) and [§2.4](./AN2DL_Playbook_Part1_Strategy.md#24-building-the-split-train_foldscsv-and-local-metric)

Stratified and grouped folds (with `reset_index` and `.iloc` to prevent misalignments on filtered DataFrames), contiguous time blocks with purge gap, rolling-origin forecasting CV with gap, and adversarial validation with size-ratio-corrected density ratio.

**Symbols and signatures:**
- `create_and_save_folds(df, target_col, group_col=None, n_splits=5, save_path=None)`
- `create_purged_time_blocks(n_timestamps, n_splits=5, purge_gap=0)`
- `get_forecasting_cv(n_splits=5, pred_len=24)`
- `run_adversarial_validation(X_train, X_test)`

---

### C. Computer Vision

**Source:** [an2dl_toolkit.py (lines 152–333)](../src/an2dl_toolkit.py#L152-L333) · **Strategy:** [Part I §3.2](./AN2DL_Playbook_Part1_Strategy.md#32-images--classification), [§4.2](./AN2DL_Playbook_Part1_Strategy.md#42-optimization), [§4.6](./AN2DL_Playbook_Part1_Strategy.md#46-model-stabilization-ema-multi-seed-swa), [§5.1](./AN2DL_Playbook_Part1_Strategy.md#51-test-time-augmentation-tta)

`timm` LLRD with `lr_scale` explicitly applied to `lr`, offline weights via `pretrained_cfg_overlay`, fp32 `SafeGeM` (ResNet only), EMA updating BN buffers (`use_buffers=True`) with progressive decay, consistent 1D/2D label-smoothing loss, `torchvision.transforms.v2` MixUp/CutMix (no `label_smoothing`), fp16/bf16 training loop, **optimizer-step scheduler** (`build_warmup_cosine_scheduler`, preserving LLRD) and **gradient accumulation** (`accum_steps`), fixed-resolution TTA.

**Symbols and signatures:**
- `SafeGeM(p=3.0, eps=1e-6)`
- `create_timm_model_and_optimizer(model_name, num_classes, lr=3e-4, weight_decay=1e-2, layer_decay=0.75, pretrained=True, local_weights_path=None, use_gem_on_resnet=False, drop_rate=0.2)`
- `WarmupEMA(model, max_decay=0.995, device=None)` (`.update_parameters(model)`)
- `SoftTargetCrossEntropy(label_smoothing=0.05)`
- `build_mixup_cutmix(num_classes)`
- `build_grad_scaler()`
- `steps_per_epoch(loader, accum_steps=1)`
- `build_warmup_cosine_scheduler(optimizer, total_steps, warmup_frac=0.05, min_lr_ratio=0.01)`
- `train_vision_epoch(model, ema_model, loader, optimizer, scheduler, criterion, scaler, mixup_fn=None, accum_steps=1)`
- `predict_with_tta(eval_model, loader, allow_hflip=True, allow_vflip=False)`

---

### D. Semantic Segmentation (Standalone Block)

**Strategy:** [Part I §3.3](./AN2DL_Playbook_Part1_Strategy.md#33-images--semantic-segmentation)

Composite Dice + Focal loss with Lovász enabled only in the final 20% of epochs. **Not included in the toolkit and unverified.** `UnetPlusPlus` requires 5 progressive-stride stages: for ConvNeXt encoders (stride-4 stem, 4 stages), use FPN/UPerNet and run a test forward pass in the installed `smp` version. SegFormer outputs 1/4-resolution logits; `compute_seg_loss` upsamples them to mask resolution.

```python
import torch.nn.functional as F
import segmentation_models_pytorch as smp

dice_loss = smp.losses.DiceLoss(mode='multiclass', from_logits=True)
focal_loss = smp.losses.FocalLoss(mode='multiclass')
lovasz_loss = smp.losses.LovaszLoss(mode='multiclass', from_logits=True)

def compute_seg_loss(logits, mask, current_epoch, total_epochs):
    # SegFormer outputs H/4 x W/4 logits
    if logits.shape[-2:] != mask.shape[-2:]:
        logits = F.interpolate(logits, size=mask.shape[-2:], mode='bilinear', align_corners=False)

    base_loss = 0.5 * dice_loss(logits, mask) + 0.5 * focal_loss(logits, mask)
    # Enable Lovasz only in the final 20% of epochs once weights have stabilized
    if current_epoch >= int(0.8 * total_epochs):
        return 0.7 * base_loss + 0.3 * lovasz_loss(logits, mask)
    return base_loss
```

---

### E. Time Series

**Source:** [an2dl_toolkit.py (lines 336–626)](../src/an2dl_toolkit.py#L336-L626) · **Strategy:** [Part I §3.4](./AN2DL_Playbook_Part1_Strategy.md#34-time-series--classification) and [§3.5](./AN2DL_Playbook_Part1_Strategy.md#35-time-series--forecasting-and-regression)

Vectorized augmentations, InceptionTime with avg + max + attention pooling, calibrated Rocket pipeline, RevIN, DLinear, PatchTST (`assert seq_len >= patch_len`, final `LayerNorm`, end-aligned patching), `forecast_loss` (blocks silent broadcasting between `[B,H,1]` and `[B,H]`), `FastSlidingWindowDataset` (zero-copy strided views built **after** splitting, with `n_series`, `series_ids`, series-level or step-level labels, shape validation on `labels` for 1D/2D series, and `warnings.warn` on dropped short series), and `aggregate_window_logits`. RevIN is optional (`use_revin`) and must be disabled in classification when absolute amplitude discriminates classes.

**Symbols and signatures:**
- `ts_augment_batch(x, jitter_sigma=0.02, scale_range=(0.9, 1.1), mask_ratio=0.08)`
- `InceptionModule1D(in_channels, n_filters=32, kernel_sizes=(9, 19, 39), bottleneck_channels=32)`
- `InceptionTimeClassifier(in_channels, num_classes, n_filters=32, depth=6)`
- `build_calibrated_rocket_pipeline(rocket_transformer)`
- `RevIN(num_features: int, eps=1e-5, affine=True)`
- `MovingAvg1D(kernel_size)`
- `DLinearForecaster(seq_len, pred_len, n_channels, kernel_size=25, use_revin=True)`
- `PatchTSTForecaster(seq_len, pred_len, n_channels, patch_len=16, stride=8, d_model=128, nhead=4, num_layers=3, use_revin=True)`
- `forecast_loss(preds, targets, criterion=None)`
- `FastSlidingWindowDataset(series, window_len, horizon=0, stride=1, labels=None, layout='LC')`
- `aggregate_window_logits(window_logits, series_ids, n_series=None)`

---

### F. Tabular Data

**Source:** [an2dl_toolkit.py (lines 629–675)](../src/an2dl_toolkit.py#L629-L675) · **Strategy:** [Part I §3.1](./AN2DL_Playbook_Part1_Strategy.md#31-tabular-data--classification-and-regression)

Periodic PLR embeddings (proper fan-in initialization; tune `sigma` on a log scale) + SwiGLU blocks. Intended to **complement** LightGBM/CatBoost for ensemble diversity, not replace them.

**Symbols and signatures:**
- `PLREmbeddings(n_features, n_frequencies=16, d_embedding=16, sigma=0.1)`
- `SwiGLUBlock(dim, hidden_dim, dropout=0.2)`
- `TabularPLRNet(n_features, out_dim, d_emb=16, hidden_dim=256, n_blocks=2, sigma=0.1, dropout=0.2)`

---

### G. Ensembling and Nested Post-Processing

**Source:** [an2dl_toolkit.py (lines 678–809)](../src/an2dl_toolkit.py#L678-L809) · **Strategy:** [Part I §5.2](./AN2DL_Playbook_Part1_Strategy.md#52-ensembling-calibration-and-nested-post-processing)

Temperature scaling, Caruana selection on log-loss (explicit `labels=` and symmetric tie-breaking), coordinate-ascent class multipliers, and `full_nested_postprocessing` fitting everything inside the fold loop. Multipliers are applied only if nested gain exceeds `min_f1_gain`. Supports 3D `test_logits` `(K, N_test, C)` by applying `mean_temps[m]` to each fold model before averaging probabilities (note in code: a diversity mismatch vs. single-fold OOF still remains because averaging discordant fold probabilities yields softer predictions; check OOF vs. test mean confidence and class balance on real data).

**Symbols and signatures:**
- `softmax_np(logits)`
- `_test_probs(logits_m, T)`
- `fit_temperature(logits, y_true)`
- `caruana_logloss_weights(probs_list, y_true, n_iterations=50)`
- `fit_multipliers_coord_ascent(probs, y_true, n_cycles=2)`
- `full_nested_postprocessing(oof_logits_list, test_logits_list, y_true, folds, min_f1_gain=0.002)`

---

### H. Additional Modules (Outside the Toolkit)

**H1 — Semi-supervised pseudo-labeling loss** (weight `lam` on test pseudo-labels). Generate pseudo-labels per fold ([Part I §4.7](./AN2DL_Playbook_Part1_Strategy.md#47-semi-supervised-learning-and-pseudo-labeling-days-1012)). Unverified.

```python
def compute_semi_supervised_loss(criterion, logits_sup, targets_sup, logits_pseudo, soft_targets_pseudo, lam=0.3):
    loss_sup = criterion(logits_sup, targets_sup)             # Mean over labeled samples
    loss_pseudo = criterion(logits_pseudo, soft_targets_pseudo)  # Mean over pseudo-labeled samples
    return loss_sup + lam * loss_pseudo
```

**H2 — Logit adjustment** (Menon et al.). Use *instead of* class multipliers, never together ([Part I §4.5](./AN2DL_Playbook_Part1_Strategy.md#45-class-imbalance-loss-functions-and-metrics)). At inference, use raw logits without the shift. Unverified; does not optimize raw accuracy.

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class LogitAdjustedSoftCrossEntropy(nn.Module):
    """Fully compatible with Mixup/CutMix (2D targets) and hard targets (1D)."""
    def __init__(self, class_counts, tau=1.0):
        super().__init__()
        priors = torch.tensor(class_counts, dtype=torch.float32)
        priors = priors / priors.sum()
        # Log shift based on training-fold class frequencies
        self.register_buffer('logit_adj', tau * torch.log(priors + 1e-12))

    def forward(self, logits, targets):
        adj_logits = logits + self.logit_adj.unsqueeze(0)
        if targets.ndim == 1:  # Standard hard labels
            return F.cross_entropy(adj_logits, targets)
        else:  # Soft labels from Mixup / CutMix [B, Num_Classes]
            log_probs = F.log_softmax(adj_logits, dim=-1)
            return -(targets * log_probs).sum(dim=-1).mean()
```

**H3 — Safe backbone / BatchNorm freezing** for two-stage transfer learning → fine-tuning ([Part I §4.4](./AN2DL_Playbook_Part1_Strategy.md#44-transfer-learning) and [§6.1 #17](./AN2DL_Playbook_Part1_Strategy.md#61-silent-technical-bugs-already-encountered)). Unverified.

```python
import torch.nn as nn

def freeze_stages(modules):
    """Disable gradients on frozen modules (call once before creating the optimizer)."""
    for m in modules:
        m.requires_grad_(False)
        m.eval()

def keep_frozen_in_eval(model, frozen_modules=(), freeze_bn=False):
    """Call right after `model.train()` (or inside an overridden `train()` method) so frozen
    BatchNorm running stats do not drift and Dropout/StochasticDepth stay off in frozen stages."""
    for m in frozen_modules:
        m.eval()
    if freeze_bn:
        for m in model.modules():
            if isinstance(m, nn.modules.batchnorm._BatchNorm):
                m.eval()
```

**H4 — Grad-CAM diagnostic & weakly supervised masking** ([Part I §2.2](./AN2DL_Playbook_Part1_Strategy.md#22-data-inspection-and-cleaning-pre-split-and-post-baseline-oof), [§3.2](./AN2DL_Playbook_Part1_Strategy.md#32-images--classification), [§3.3](./AN2DL_Playbook_Part1_Strategy.md#33-images--semantic-segmentation)). Works on any convolutional stage, including frozen backbones (`x_in.requires_grad_(True)`). Unverified.

```python
import torch
import torch.nn.functional as F

def grad_cam(model, x, target_layer, classes):
    """Grad-CAM for a batch [B, C, H, W] on any conv layer; returns [B, H, W] maps in [0, 1] on CPU.
    Threshold (e.g., >= 0.5) for rough weakly supervised foreground masks."""
    model.eval()
    stored = {}
    handle = target_layer.register_forward_hook(lambda _m, _inp, out: stored.update(act=out))
    try:
        x_in = x.detach().clone().requires_grad_(True)  # ensures gradients reach frozen stages too
        logits = model(x_in)
        act = stored['act']
        cls = torch.as_tensor(classes, device=logits.device)
        score = logits[torch.arange(len(x_in), device=logits.device), cls].sum()
        grads, = torch.autograd.grad(score, act)
    finally:
        handle.remove()
    alphas = grads.mean(dim=(2, 3), keepdim=True)
    cam = F.relu((alphas * act).sum(dim=1, keepdim=True))
    cam = F.interpolate(cam, size=x.shape[-2:], mode='bilinear', align_corners=False)[:, 0]
    return (cam / (cam.amax(dim=(1, 2), keepdim=True) + 1e-8)).detach().cpu()
```

---

### I. Smoke Test (`smoke_test.py`)

**Source:** [smoke_test.py](../tests/smoke_test.py)

Run on Colab after `!pip install -q timm lightgbm`, with `an2dl_toolkit.py` in the same directory (`python smoke_test.py`). Checks in sequence:

1. **`create_and_save_folds`** on a filtered DataFrame with non-contiguous indices.
2. **Purged Time Blocks & Adversarial Validation** (`purge_gap` enforcement, output shapes, no `NaN`).
3. **Vision (`3A`, `3B`, `3C`)**:
   - `3A`: LLRD active on `resnet18`, `convnext_tiny`, `vit_tiny_patch16_224`;
   - `3B`: full training loop with per-step scheduler (LLRD ratios preserved), MixUp/CutMix, EMA on BN buffers, TTA;
   - `3C`: gradient accumulation (`accum_steps=2`) stepping the scheduler per optimizer step.
4. **Time Series (`4`, `4b`)**:
   - `4`: `ts_augment_batch`, `InceptionTimeClassifier`, `DLinearForecaster`, `PatchTSTForecaster` (including `rem=4`), `forecast_loss`, `FastSlidingWindowDataset`, `aggregate_window_logits`;
   - `4b`: per-timestep label alignment on variable-length series with `stride > 1`.
5. **Tabular (`5`)**: `TabularPLRNet` (PLR init + SwiGLU).
6. **Nested Post-Processing (`6`, `6b`)**:
   - `6`: missing class in a fold, deterministic tie-break on identical models, 2D and 3D `(K, N, C)` test logits;
   - `6b`: 3D `_test_probs` equals mean of per-fold softmaxes (and differs from softmax of mean logits).

---

_Library and model versions referenced (`timm`, `torchvision`, `smp`, `aeon`, `TabPFN`, `Chronos`) must be verified at time of use._
