import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import TensorDataset
from an2dl_toolkit import (
    seed_everything, DEVICE, make_loader, create_and_save_folds, create_purged_time_blocks,
    run_adversarial_validation, create_timm_model_and_optimizer, WarmupEMA,
    SoftTargetCrossEntropy, build_mixup_cutmix, build_grad_scaler,
    steps_per_epoch, build_warmup_cosine_scheduler, train_vision_epoch, predict_with_tta,
    ts_augment_batch, InceptionTimeClassifier, DLinearForecaster, PatchTSTForecaster,
    forecast_loss, FastSlidingWindowDataset, aggregate_window_logits,
    TabularPLRNet, full_nested_postprocessing, caruana_logloss_weights, softmax_np, _test_probs
)

def run_all_smoke_tests():
    seed_everything(42)
    print(f"Running Smoke Tests on device: {DEVICE}...")

    # 1. Test Folds on filtered DataFrame
    df = pd.DataFrame({'y': np.random.randint(0, 3, 100), 'grp': np.repeat(np.arange(20), 5)})
    df_folds = create_and_save_folds(df.iloc[10:90], target_col='y', group_col='grp', n_splits=4)
    assert (df_folds['fold'] != -1).all(), "BUG: Some rows were not assigned a fold!"
    print("[OK] 1. create_and_save_folds on non-contiguous index.")

    # 2. Test Purged Time Blocks & Adversarial Validation
    blocks = create_purged_time_blocks(n_timestamps=200, n_splits=4, purge_gap=10)
    for tr_idx, va_idx in blocks:
        assert np.min(np.abs(tr_idx[:, None] - va_idx[None, :])) > 10, "BUG: Purge gap violated!"
    auc, weights, _ = run_adversarial_validation(np.random.randn(80, 5), np.random.randn(40, 5))
    assert len(weights) == 80 and not np.isnan(auc), "BUG: Adversarial validation failed!"
    print("[OK] 2. Purged Time Blocks & Adversarial Validation.")

    # 3A. Test LLRD on ResNet, ConvNeXt, and ViT
    for arch in ['resnet18', 'convnext_tiny', 'vit_tiny_patch16_224']:
        _, opt_arch = create_timm_model_and_optimizer(arch, num_classes=3, lr=1e-3, layer_decay=0.7, pretrained=False)
        lrs = [g['lr'] for g in opt_arch.param_groups]
        assert min(lrs) < max(lrs), f"BUG: LLRD inactive on {arch}!"
    print("[OK] 3A. LLRD verified on ResNet, ConvNeXt, and ViT.")

    # 3B. Vision loop: per-step scheduler (LLRD preserved), MixUp/CutMix, EMA on BN buffers, TTA
    model, optimizer = create_timm_model_and_optimizer(
        'resnet18', num_classes=3, lr=1e-3, layer_decay=0.5, pretrained=False, use_gem_on_resnet=True
    )
    model = model.to(DEVICE, memory_format=torch.channels_last)
    ds = TensorDataset(torch.randn(8, 3, 64, 64), torch.randint(0, 3, (8,)))
    loader = make_loader(ds, batch_size=4, is_train=True, num_workers=0)
    EPOCHS = 2
    total_steps = EPOCHS * steps_per_epoch(loader, accum_steps=1)
    scheduler = build_warmup_cosine_scheduler(optimizer, total_steps)
    lrs_before = [g['lr'] for g in optimizer.param_groups]
    ratio_before = [l / lrs_before[0] for l in lrs_before]
    assert min(lrs_before) < max(lrs_before), "BUG: LLRD did not differentiate learning rates!"

    ema_model = WarmupEMA(model, max_decay=0.9, device=DEVICE)
    bn_buf_before = ema_model.module.bn1.running_mean.clone()
    criterion = SoftTargetCrossEntropy(label_smoothing=0.05)
    mixup_fn = build_mixup_cutmix(num_classes=3)
    scaler = build_grad_scaler()
    for _ in range(EPOCHS):
        train_vision_epoch(model, ema_model, loader, optimizer, scheduler, criterion, scaler, mixup_fn)
    assert scheduler.last_epoch == total_steps, f"BUG: scheduler at {scheduler.last_epoch} steps instead of {total_steps}"
    lrs_after = [g['lr'] for g in optimizer.param_groups]
    ratio_after = [l / lrs_after[0] for l in lrs_after]
    assert np.allclose(ratio_before, ratio_after, rtol=1e-4), "BUG: scheduler altered LLRD ratios!"
    assert not torch.allclose(bn_buf_before, ema_model.module.bn1.running_mean), "BUG: EMA did not update BN buffers!"

    eval_loader = make_loader(ds, batch_size=4, is_train=False, num_workers=0)
    tta_logits = predict_with_tta(ema_model, eval_loader, allow_hflip=True)
    assert tta_logits.shape == (8, 3), f"BUG: Wrong TTA shape {tta_logits.shape}"
    print("[OK] 3B. Vision loop: per-step scheduler + LLRD + MixUp/CutMix + EMA BN + TTA.")

    # 3C. Gradient accumulation: 1 optimizer step every 2 micro-batches
    model2, opt2 = create_timm_model_and_optimizer('resnet18', num_classes=3, lr=1e-3, layer_decay=0.5, pretrained=False)
    model2 = model2.to(DEVICE, memory_format=torch.channels_last)
    sched2 = build_warmup_cosine_scheduler(opt2, total_steps=steps_per_epoch(loader, accum_steps=2))
    train_vision_epoch(model2, None, loader, opt2, sched2, criterion, scaler, mixup_fn, accum_steps=2)
    assert sched2.last_epoch == 1, f"BUG: with accum_steps=2 expected 1 step, got {sched2.last_epoch}"
    print("[OK] 3C. Gradient accumulation: scheduler steps per optimizer step.")

    # 4. Time series: classification and forecasting
    x_ts = ts_augment_batch(torch.randn(4, 6, 128, device=DEVICE))
    inc_model = InceptionTimeClassifier(in_channels=6, num_classes=3).to(DEVICE)
    assert inc_model(x_ts).shape == (4, 3), "BUG: Wrong InceptionTime output shape!"

    x_fc = torch.randn(4, 64, 5, device=DEVICE)
    dlin = DLinearForecaster(seq_len=64, pred_len=16, n_channels=5).to(DEVICE)
    ptst = PatchTSTForecaster(seq_len=64, pred_len=16, n_channels=5).to(DEVICE)
    assert dlin(x_fc).shape == (4, 16, 5) and ptst(x_fc).shape == (4, 16, 5), "BUG: Wrong forecaster shape!"

    # forecast_loss: 2D target accepted only when C == 1; other incompatible shapes must fail
    uni = DLinearForecaster(seq_len=64, pred_len=16, n_channels=1).to(DEVICE)
    out_uni = uni(torch.randn(4, 64, 1, device=DEVICE))
    y2d = torch.randn(4, 16, device=DEVICE)
    assert torch.isclose(forecast_loss(out_uni, y2d), F.mse_loss(out_uni, y2d.unsqueeze(-1))), "BUG: forecast_loss failed!"
    try:
        forecast_loss(dlin(x_fc), torch.randn(4, 16, device=DEVICE))
        raise AssertionError("BUG: incompatible shapes were not caught!")
    except ValueError:
        pass

    # FastSlidingWindowDataset: separate segments, [L, C] layout, y [H, C]
    series = [np.random.randn(120, 5).astype(np.float32), np.random.randn(90, 5).astype(np.float32)]
    swd = FastSlidingWindowDataset(series, window_len=64, horizon=16, stride=8, layout='LC')
    xb, yb = next(iter(make_loader(swd, batch_size=4, is_train=False, num_workers=0)))
    assert xb.shape == (4, 64, 5) and yb.shape == (4, 16, 5), f"BUG: window shapes {xb.shape}, {yb.shape}"
    assert ptst(xb.to(DEVICE)).shape == yb.shape, "BUG: forecaster output incompatible with y!"

    # PatchTST with seq_len not multiple of stride (rem=4): drops first 4 timesteps, not last 4
    ptst_rem = PatchTSTForecaster(seq_len=100, pred_len=16, n_channels=5, patch_len=16, stride=8, use_revin=False).to(DEVICE)
    ptst_rem.eval()
    x_rem = torch.randn(4, 100, 5, device=DEVICE)
    x_head_mod = x_rem.clone(); x_head_mod[:, :4, :] += 10.0
    x_tail_mod = x_rem.clone(); x_tail_mod[:, -4:, :] += 10.0
    with torch.no_grad():
        out_base = ptst_rem(x_rem)
        out_head = ptst_rem(x_head_mod)
        out_tail = ptst_rem(x_tail_mod)
    assert out_base.shape == (4, 16, 5), f"BUG: wrong PatchTST rem shape {out_base.shape}"
    assert torch.allclose(out_base, out_head, atol=1e-5), "BUG: PatchTST used the first 4 droppable timesteps!"
    assert not torch.allclose(out_base, out_tail, atol=1e-5), "BUG: PatchTST truncated the tail!"

    # Windowed classification: one label per series + per-series logit aggregation
    cls_series = [np.random.randn(100, 6).astype(np.float32), np.random.randn(80, 6).astype(np.float32), np.random.randn(90, 6).astype(np.float32)]
    swd_cls = FastSlidingWindowDataset(cls_series, window_len=32, stride=16, labels=[0, 2, 1], layout='CL')
    assert len(swd_cls.series_ids) == len(swd_cls) and swd_cls[0][1] == 0 and swd_cls[-1][1] == 1
    win_logits = np.random.randn(len(swd_cls), 3)
    ser_logits = aggregate_window_logits(win_logits, swd_cls.series_ids, n_series=swd_cls.n_series)
    assert ser_logits.shape == (3, 3), f"BUG: wrong aggregate_window_logits shape {ser_logits.shape}"

    # 4b. Per-timestep label alignment (2 variable-length series, stride > 1)
    s = [np.random.randn(100, 3).astype(np.float32), np.random.randn(70, 3).astype(np.float32)]
    lab = [np.arange(100), 1000 + np.arange(70)]
    ds = FastSlidingWindowDataset(s, window_len=20, stride=7, labels=lab, layout='CL')
    for i in range(len(ds)):
        k = int(ds.series_ids[i]); j = i - int(ds.cum[k])
        assert ds[i][1] == lab[k][j * 7 + 19], "BUG: label not aligned with last timestep!"
    print("[OK] 4. Time series: Augmentation + InceptionTime + DLinear + PatchTST + forecast_loss + windows + aggregation.")

    # 5. Tabular PLR + SwiGLU
    tab_model = TabularPLRNet(n_features=10, out_dim=3).to(DEVICE)
    assert tab_model(torch.randn(8, 10, device=DEVICE)).shape == (8, 3), "BUG: Wrong TabularPLRNet shape!"
    print("[OK] 5. TabularPLRNet (PLR init + SwiGLU).")

    # 6. Nested post-processing with missing class in a fold, identical models, 2D and 3D (K, N, C) test logits
    N, C = 150, 4
    y_true = np.random.randint(0, C, N)
    folds = np.repeat(np.arange(5), N // 5)
    y_true[folds != 0] = np.random.randint(0, C - 1, np.sum(folds != 0))
    y_true[folds == 0] = 3
    p_ident = softmax_np(np.random.randn(N, C))
    p_3d = softmax_np(np.random.randn(5, 10, C))
    assert p_3d.shape == (5, 10, C) and np.allclose(p_3d.sum(axis=-1), 1.0), "BUG: softmax_np failed on 3D!"
    w_ident = caruana_logloss_weights([p_ident, p_ident, p_ident], y_true)
    assert np.allclose(w_ident, [1/3, 1/3, 1/3]), f"BUG: Asymmetric tie-break on identical models: {w_ident}"
    oof_list = [np.random.randn(N, C) + np.eye(C)[y_true] * 1.5 for _ in range(3)]
    test_list_2d = [np.random.randn(40, C) for _ in range(3)]
    test_preds_2d, w_2d, _ = full_nested_postprocessing(oof_list, test_list_2d, y_true, folds)
    assert test_preds_2d.shape == (40,) and np.isclose(w_2d.sum(), 1.0), "BUG: Post-processing failed on 2D!"
    test_list_3d = [np.random.randn(5, 40, C) for _ in range(3)]
    test_preds, w, mults = full_nested_postprocessing(oof_list, test_list_3d, y_true, folds)
    assert test_preds.shape == (40,) and np.isclose(w.sum(), 1.0), "BUG: Post-processing failed on 3D!"

    # 6b. 3D _test_probs = mean of per-fold softmaxes (!= softmax of mean logits)
    L = np.random.randn(5, 30, 4) * 3
    assert np.allclose(_test_probs(L, 1.3), np.mean([softmax_np(L[f] / 1.3) for f in range(5)], axis=0))
    assert not np.allclose(_test_probs(L, 1.3), softmax_np(L.mean(0) / 1.3))
    print("[OK] 6. Nested OOF Post-Processing (missing class + deterministic tie-break + 2D/3D test + _test_probs).")
    print("\nALL TESTS PASSED WITHOUT ERRORS.")

if __name__ == '__main__':
    run_all_smoke_tests()
