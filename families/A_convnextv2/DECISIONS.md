# Family A - fine-tuning decisions

Decided before the first real training run. Each line: choice, reason, how it is checked.

## 1. Unfreezing and learning rates
- **LP-FT**: 3 epochs with the backbone frozen (head only, lr 1e-3), then the full network.
  The head starts sensible, so the first gradients do not distort the pretrained features;
  linear-probe-then-fine-tune is known to hold up better under distribution shift (test has
  hue rotation on >= 35% and blur on ~32% of images).
- **Mild layer-wise LR decay: stem at ~40% of the head's LR**, so early layers adapt a bit to the
  astronomical domain. timm splits ConvNeXtV2-tiny into 20 layer groups, so this is
  `layer_decay: 0.95` (0.95^19 = 0.38); the playbook's 0.8 meant per stage (0.8^4 = 0.41) and
  would leave the stem at 1.4% here.
- **Base LR by mini-sweep**: 5e-5, 1e-4, 2e-4 on fold 0, 15 epochs; pick by f1_shifted (shared/val_shift_plan.csv).

## 2. Colour
- **Decided by experiment**: colour + hue rotation (p=0.4, full circle) vs grayscale, same recipe,
  fold 0; pick by f1_shifted (shared/val_shift_plan.csv). Colour carries class information (red ellipticals, blue spirals)
  but >= 35% of the test is hue-rotated.
- No extra tint defence for now (random grayscale, colour+gray ensemble, test de-tinting):
  revisit only if the gap on test-like validation is large.

## 3. Resolution and cropping
- **Native 256**, no resize: bars, tight arms and the core are small details. ~1.3x slower than 224.
  Blur sigma 0.4-1.2 px applies as measured (no rescaling needed).
- **Whole image + mild zoom 0.95-1.05** (+ any-angle rotation, flip): the galaxy stays centred and
  complete, as in the test. No random resized crop (would cut outer arms, train/test mismatch).

## 4. Label noise, imbalance, discarded images
- **Label smoothing 0.1** as the only noise defence in the base recipe. Robust losses (SCE/GCE)
  and a two-pass OOF cleaning (using the 136 EDA candidates) are later experiments.
- **No class weighting / balanced sampling** in training: weighted F1 gives class 4 ~2% weight.
  Per-class logit multipliers fitted on OOF afterwards (`fit_multipliers_coord_ascent`).
- **Train only on the 2,995 clean images**: no grainy with flipped labels (swap confirmed only
  57-58%), no aliens (labels at chance level).

## 5. Augmentation
- **Hue rotation p=0.4 (full circle), blur p=0.35 (sigma 0.4-1.2 px)**, as measured on the test.
- No corruption augmentation (lines/rectangles) in the base recipe: later experiment.
- **No noise**: the test is smoother than train, not noisier.

## 6. Duration and regularisation
- **Epochs from fold-0 curves**: one 40-epoch run on fold 0 with validation every epoch, then a
  single fixed number for all folds (no per-fold best-epoch picking).
- **No MixUp** in the base recipe: blended galaxies look like class 1 (merging).
- **Light regularisation**: stochastic depth 0.1, dropout 0.2 before the head, weight decay 0.05.
- **EMA 0.998** of the weights; the final EMA is the model used for OOF and test.

## 7. Inference
- **D4 TTA** (4 rot90 x flip), identical for OOF and test.
- **Temperature + per-class multipliers** fitted on OOF with nested validation
  (`full_nested_postprocessing`), kept only if the OOF gain passes the threshold.
- **Pseudo-labeling only after the forum confirms** unlabeled test use is allowed.
- **Final test prediction = mean of the 5 fold models** (multi-seed only once the recipe is final).

## Experiment order (fold 0, Colab)
1. Curves run: 40 epochs, validation every epoch -> fix the number of epochs.
2. LR sweep: 5e-5 / 1e-4 / 2e-4.
3. Colour + hue rotation vs grayscale.
4. Full 5-fold run with the winning recipe (Kaggle) -> first real submission.
5. Later: robust loss, corruption augmentation, two-pass cleaning, MixUp 0.2, convnext_small.
