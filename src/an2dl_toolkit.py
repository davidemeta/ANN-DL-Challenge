# =====================================================================
# ### A. Setup, Determinism, and Hardware
# =====================================================================
import os
os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'

import copy
import random
import warnings
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision.transforms import v2
from scipy.optimize import minimize
from sklearn.model_selection import StratifiedKFold, StratifiedGroupKFold, TimeSeriesSplit
from sklearn.metrics import roc_auc_score, f1_score, log_loss
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler, QuantileTransformer
from sklearn.linear_model import RidgeClassifierCV
from sklearn.calibration import CalibratedClassifierCV
import timm
from timm.optim import create_optimizer_v2

# =====================================================================
# 1. SETUP, DETERMINISM & HARDWARE
# =====================================================================
def seed_everything(seed=42, deterministic=False):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        torch.use_deterministic_algorithms(True, warn_only=True)
    else:
        torch.backends.cudnn.benchmark = True

def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)

def get_amp_config():
    """T4 (cc 7.5) -> fp16 + GradScaler. L4/A100 (cc >= 8.0) -> bf16 without GradScaler."""
    if not torch.cuda.is_available():
        return torch.float32, False
    major, _ = torch.cuda.get_device_capability()
    if major >= 8:
        return torch.bfloat16, False
    return torch.float16, True

SEED = 42
# cuda (Colab/Kaggle) > mps (Apple Silicon) > cpu. AMP stays CUDA-only: MPS trains in fp32.
DEVICE = torch.device('cuda' if torch.cuda.is_available()
                      else 'mps' if torch.backends.mps.is_available() else 'cpu')
AMP_DTYPE, USE_SCALER = get_amp_config()

def make_loader(dataset, batch_size, is_train, num_workers=2, seed=SEED):
    """DataLoader with persistent workers (no respawn per epoch), pin_memory, and worker seeding.
    - num_workers: on free Colab (2 vCPU) 2 is a good starting point; tune based on os.cpu_count().
    - drop_last=True in training: a trailing 1-sample batch breaks BatchNorm and MixUp.
    - persistent_workers requires num_workers > 0."""
    g = torch.Generator()
    g.manual_seed(seed)
    kwargs = dict(
        batch_size=batch_size, shuffle=is_train, drop_last=is_train,
        num_workers=num_workers, pin_memory=(DEVICE.type == 'cuda'),
        worker_init_fn=seed_worker, generator=g,
    )
    if num_workers > 0:
        kwargs['persistent_workers'] = True
    return DataLoader(dataset, **kwargs)


# =====================================================================
# ### B. Anti-Leakage Validation
# =====================================================================
# =====================================================================
# 2. ANTI-LEAKAGE VALIDATION, BLOCK SPLIT & ADVERSARIAL VALIDATION
# =====================================================================
def create_and_save_folds(df, target_col, group_col=None, n_splits=5, save_path=None):
    """Reset index and use .iloc to avoid misalignments on filtered DataFrames."""
    df = df.reset_index(drop=True).copy()
    df['fold'] = -1
    if group_col is not None:
        splitter = StratifiedGroupKFold(n_splits=n_splits)
        splits = splitter.split(df, df[target_col], df[group_col])
    else:
        splitter = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=SEED)
        splits = splitter.split(df, df[target_col])

    fold_col_idx = df.columns.get_loc('fold')
    for fold, (_, val_idx) in enumerate(splits):
        df.iloc[val_idx, fold_col_idx] = fold
    if save_path:
        df.to_csv(save_path, index=False)
    return df

def create_purged_time_blocks(n_timestamps, n_splits=5, purge_gap=0):
    """
    For time series / sliding windows: split the time series into contiguous blocks
    leaving a purge_gap (>= window_len) between train and val to eliminate window overlap.
    """
    block_size = n_timestamps // n_splits
    folds = []
    all_idx = np.arange(n_timestamps)
    for k in range(n_splits):
        val_start = k * block_size
        val_end = n_timestamps if k == n_splits - 1 else (k + 1) * block_size
        val_idx = all_idx[val_start:val_end]
        # Remove both the val block and the purge_gap on both boundaries from train
        left_end = max(0, val_start - purge_gap)
        right_start = min(n_timestamps, val_end + purge_gap)
        train_idx = np.concatenate([all_idx[:left_end], all_idx[right_start:]])
        folds.append((train_idx, val_idx))
    return folds

def get_forecasting_cv(n_splits=5, pred_len=24):
    """Rolling-origin CV for forecasting: gap >= pred_len prevents train targets from leaking into val."""
    return TimeSeriesSplit(n_splits=n_splits, gap=pred_len)

def run_adversarial_validation(X_train, X_test):
    """
    X_train, X_test: tabular features or backbone embeddings for images/series.
    Computes size-prior-corrected density ratio: (N_train / N_test) * p / (1 - p).
    """
    n_train, n_test = len(X_train), len(X_test)
    X_adv = np.vstack([X_train, X_test])
    y_adv = np.concatenate([np.zeros(n_train), np.ones(n_test)])

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    oof_preds = np.zeros(len(X_adv))
    importances = np.zeros(X_train.shape[1])

    import lightgbm as lgb  # lazy: only needed here, absent from the Mac venv
    for tr_idx, va_idx in skf.split(X_adv, y_adv):
        model = lgb.LGBMClassifier(n_estimators=150, learning_rate=0.05, random_state=SEED, verbose=-1)
        model.fit(X_adv[tr_idx], y_adv[tr_idx])
        oof_preds[va_idx] = model.predict_proba(X_adv[va_idx])[:, 1]
        importances += model.feature_importances_ / 5.0

    auc = roc_auc_score(y_adv, oof_preds)
    p_train = np.clip(oof_preds[:n_train], 1e-4, 1.0 - 1e-4)
    # Correct with (n_train / n_test) ratio to obtain p(x|test) / p(x|train)
    density_ratio = (n_train / float(n_test)) * (p_train / (1.0 - p_train))
    return auc, density_ratio, importances


# =====================================================================
# ### C. Computer Vision
# =====================================================================
# =====================================================================
# 3. COMPUTER VISION: LLRD FIX, OFFLINE WEIGHTS, EMA, TTA
# =====================================================================
class SafeGeM(nn.Module):
    """Use ONLY on post-ReLU CNNs (e.g., ResNet). Executed in fp32 to prevent overflow."""
    def __init__(self, p=3.0, eps=1e-6):
        super().__init__()
        self.p = nn.Parameter(torch.ones(1) * p)
        self.eps = eps

    def forward(self, x):
        with torch.amp.autocast(device_type=x.device.type, enabled=False):
            x_fp32 = F.relu(x.float()).clamp(min=self.eps)
            p_fp32 = self.p.float().clamp(min=1.0, max=10.0)
            return F.adaptive_avg_pool2d(x_fp32.pow(p_fp32), (1, 1)).pow(1.0 / p_fp32)

def create_timm_model_and_optimizer(
    model_name, num_classes, lr=3e-4, weight_decay=1e-2, layer_decay=0.75,
    pretrained=True, local_weights_path=None, use_gem_on_resnet=False, drop_rate=0.2
):
    """
    Supports offline weights via pretrained_cfg_overlay and explicitly applies
    group['lr_scale'] to group['lr'] so LLRD works with PyTorch schedulers.
    """
    overlay = {'file': local_weights_path} if local_weights_path is not None else {}
    model = timm.create_model(
        model_name,
        pretrained=pretrained,
        num_classes=num_classes,
        drop_rate=drop_rate,
        pretrained_cfg_overlay=overlay
    )
    if use_gem_on_resnet and 'resnet' in model_name and hasattr(model, 'global_pool'):
        model.global_pool = nn.Sequential(SafeGeM(), nn.Flatten(1))

    optimizer = create_optimizer_v2(
        model, opt='adamw', lr=lr, weight_decay=weight_decay, layer_decay=layer_decay
    )
    # CRITICAL FIX: timm stores the multiplier in 'lr_scale' but leaves 'lr' unchanged.
    # Apply 'lr_scale' directly to 'lr' so any PyTorch scheduler preserves LLRD.
    for group in optimizer.param_groups:
        if 'lr_scale' in group:
            group['lr'] *= group['lr_scale']

    return model, optimizer

class WarmupEMA(torch.optim.swa_utils.AveragedModel):
    """Updates both parameters and BatchNorm buffers with progressive decay."""
    def __init__(self, model, max_decay=0.995, device=None):
        super().__init__(model, device=device, use_buffers=True)
        self.max_decay = max_decay
        self.num_updates = 0

    def update_parameters(self, model):
        self.num_updates += 1
        decay = min(self.max_decay, (1.0 + self.num_updates) / (10.0 + self.num_updates))
        with torch.no_grad():
            for p_ema, p_mod in zip(self.module.parameters(), model.parameters()):
                p_ema.detach().mul_(decay).add_(p_mod.detach(), alpha=1.0 - decay)
            for b_ema, b_mod in zip(self.module.buffers(), model.buffers()):
                if b_mod.dtype.is_floating_point:
                    b_ema.detach().mul_(decay).add_(b_mod.detach(), alpha=1.0 - decay)
                else:
                    b_ema.detach().copy_(b_mod.detach())

class SoftTargetCrossEntropy(nn.Module):
    """Applies label_smoothing consistently to both 1D (hard) and 2D (MixUp/CutMix) targets."""
    def __init__(self, label_smoothing=0.05):
        super().__init__()
        self.label_smoothing = label_smoothing

    def forward(self, logits, targets):
        if targets.ndim == 1:
            return F.cross_entropy(logits, targets, label_smoothing=self.label_smoothing)
        # 2D branch (MixUp / CutMix): blend soft labels with uniform distribution
        n_classes = logits.size(-1)
        if self.label_smoothing > 0.0:
            targets = targets * (1.0 - self.label_smoothing) + (self.label_smoothing / n_classes)
        log_probs = F.log_softmax(logits, dim=-1)
        return -(targets * log_probs).sum(dim=-1).mean()

def build_mixup_cutmix(num_classes):
    """torchvision.transforms.v2.MixUp / CutMix only accept alpha and num_classes."""
    return v2.RandomChoice([
        v2.MixUp(alpha=0.4, num_classes=num_classes),
        v2.CutMix(alpha=1.0, num_classes=num_classes)
    ])

def build_grad_scaler():
    return torch.amp.GradScaler('cuda', enabled=USE_SCALER)

def steps_per_epoch(loader, accum_steps=1):
    """Number of optimizer steps per epoch (with gradient accumulation)."""
    return (len(loader) + accum_steps - 1) // accum_steps

def build_warmup_cosine_scheduler(optimizer, total_steps, warmup_frac=0.05, min_lr_ratio=0.01):
    """Linear warmup + cosine decay counted in OPTIMIZER STEPS (not epochs).
    total_steps = EPOCHS * steps_per_epoch(loader, accum_steps).
    Uses LambdaLR: scales each group's initial lr, preserving LLRD
    (already applied to group['lr'] in create_timm_model_and_optimizer)."""
    import math
    warmup_steps = max(1, int(warmup_frac * total_steps))
    def lr_lambda(step):
        if step < warmup_steps:
            return (step + 1) / warmup_steps
        progress = min(1.0, (step - warmup_steps) / max(1, total_steps - warmup_steps))
        return min_lr_ratio + (1.0 - min_lr_ratio) * 0.5 * (1.0 + math.cos(math.pi * progress))
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

def train_vision_epoch(model, ema_model, loader, optimizer, scheduler, criterion, scaler,
                       mixup_fn=None, accum_steps=1):
    """Scheduler and EMA advance once per optimizer step, NOT per micro-batch.
    With accum_steps > 1 this simulates a larger batch (gradients averaged over accum_steps).
    Note: MixUp/CutMix still mix within each micro-batch, and BatchNorm stats remain at
    micro-batch scale; accumulation reduces gradient noise without changing those.
    The final step of an epoch may accumulate fewer micro-batches (slightly higher weight)."""
    model.train()
    use_scaler = USE_SCALER and DEVICE.type == 'cuda'
    n_batches = len(loader)
    total_loss, n_seen = 0.0, 0
    optimizer.zero_grad(set_to_none=True)
    for i, (images, targets) in enumerate(loader):
        images = images.to(DEVICE, non_blocking=True, memory_format=torch.channels_last)
        targets = targets.to(DEVICE, non_blocking=True)

        if mixup_fn is not None and images.size(0) > 1 and np.random.rand() < 0.8:
            images, targets = mixup_fn(images, targets)

        with torch.amp.autocast(device_type=DEVICE.type, dtype=AMP_DTYPE, enabled=(DEVICE.type == 'cuda')):
            logits = model(images)
            loss = criterion(logits, targets)
        total_loss += loss.item() * images.size(0)
        n_seen += images.size(0)

        loss = loss / accum_steps
        if use_scaler:
            scaler.scale(loss).backward()
        else:
            loss.backward()

        if (i + 1) % accum_steps == 0 or (i + 1) == n_batches:
            if use_scaler:
                scaler.unscale_(optimizer)  # required before gradient clipping
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                scaler.step(optimizer)
                scaler.update()
            else:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            if scheduler is not None:
                scheduler.step()
            if ema_model is not None:
                ema_model.update_parameters(model)
    return total_loss / max(1, n_seen)

@torch.no_grad()
def predict_with_tta(eval_model, loader, allow_hflip=True, allow_vflip=False):
    """
    Always use ema_model for inference!
    Keeps resolution fixed to avoid breaking ViT pos_embed or stride-32 divisibility.
    Disable allow_hflip if left/right orientation is semantic (e.g., ECG, digits, asymmetric X-rays).
    """
    eval_model.eval()
    all_logits = []
    for batch in loader:
        images = batch[0] if isinstance(batch, (list, tuple)) else batch
        images = images.to(DEVICE, non_blocking=True, memory_format=torch.channels_last)
        with torch.amp.autocast(device_type=DEVICE.type, dtype=AMP_DTYPE, enabled=(DEVICE.type == 'cuda')):
            logits_sum = eval_model(images)
            n_views = 1
            if allow_hflip:
                logits_sum = logits_sum + eval_model(torch.flip(images, dims=[-1]))
                n_views += 1
            if allow_vflip:
                logits_sum = logits_sum + eval_model(torch.flip(images, dims=[-2]))
                n_views += 1
        all_logits.append((logits_sum / n_views).float().cpu().numpy())
    return np.concatenate(all_logits, axis=0)


# =====================================================================
# ### E. Time Series
# =====================================================================
# =====================================================================
# 4. TIME SERIES: INCEPTIONTIME, AUGMENTATION, REVIN, DLINEAR, PATCHTST
# =====================================================================
def ts_augment_batch(x, jitter_sigma=0.02, scale_range=(0.9, 1.1), mask_ratio=0.08):
    """Vectorized augmentation on [B, C, L] tensor during training only."""
    B, C, L = x.shape
    # 1. Random Scaling
    scales = torch.empty(B, C, 1, device=x.device, dtype=x.dtype).uniform_(*scale_range)
    x = x * scales
    # 2. Gaussian Jitter
    if jitter_sigma > 0:
        x = x + torch.randn_like(x) * jitter_sigma
    # 3. Time Masking
    mask_len = int(L * mask_ratio)
    if mask_len > 0:
        t0 = random.randint(0, L - mask_len)
        x[:, :, t0:t0 + mask_len] = 0.0
    return x

class InceptionModule1D(nn.Module):
    def __init__(self, in_channels, n_filters=32, kernel_sizes=(9, 19, 39), bottleneck_channels=32):
        super().__init__()
        self.use_bottleneck = in_channels > 1
        conv_in = bottleneck_channels if self.use_bottleneck else in_channels
        if self.use_bottleneck:
            self.bottleneck = nn.Conv1d(in_channels, bottleneck_channels, kernel_size=1, bias=False)
        self.convs = nn.ModuleList([
            nn.Conv1d(conv_in, n_filters, kernel_size=k, padding=k // 2, bias=False) for k in kernel_sizes
        ])
        self.maxpool = nn.MaxPool1d(kernel_size=3, stride=1, padding=1)
        self.conv_pool = nn.Conv1d(in_channels, n_filters, kernel_size=1, bias=False)
        self.bn = nn.BatchNorm1d(n_filters * (len(kernel_sizes) + 1))
        self.act = nn.GELU()

    def forward(self, x):
        x_in = self.bottleneck(x) if self.use_bottleneck else x
        outs = [conv(x_in) for conv in self.convs] + [self.conv_pool(self.maxpool(x))]
        return self.act(self.bn(torch.cat(outs, dim=1)))

class InceptionTimeClassifier(nn.Module):
    """6 blocks with residual shortcuts every 3 + concatenated AvgPool, MaxPool, AttentionPool."""
    def __init__(self, in_channels, num_classes, n_filters=32, depth=6):
        super().__init__()
        self.depth = depth
        out_ch = n_filters * 4
        self.blocks = nn.ModuleList()
        self.shortcuts = nn.ModuleList()
        for d in range(depth):
            inp = in_channels if d == 0 else out_ch
            self.blocks.append(InceptionModule1D(inp, n_filters=n_filters))
            if d % 3 == 2:
                sc_in = in_channels if d == 2 else out_ch
                self.shortcuts.append(nn.Sequential(
                    nn.Conv1d(sc_in, out_ch, kernel_size=1, bias=False),
                    nn.BatchNorm1d(out_ch)
                ))
        self.att_vector = nn.Linear(out_ch, 1)
        self.head = nn.Sequential(
            nn.LayerNorm(out_ch * 3),
            nn.Dropout(0.3),
            nn.Linear(out_ch * 3, num_classes)
        )

    def forward(self, x):
        # x: [B, C, L]
        res = x
        for d in range(self.depth):
            x = self.blocks[d](x)
            if d % 3 == 2:
                x = F.gelu(x + self.shortcuts[d // 3](res))
                res = x
        avg_pool = x.mean(dim=-1)
        max_pool = x.amax(dim=-1)
        x_perm = x.transpose(1, 2)  # [B, L, C]
        att_w = torch.softmax(self.att_vector(x_perm), dim=1)
        att_pool = torch.sum(x_perm * att_w, dim=1)
        return self.head(torch.cat([avg_pool, max_pool, att_pool], dim=-1))

def build_calibrated_rocket_pipeline(rocket_transformer):
    """
    Fast MiniRocket/MultiRocket wrapper: RidgeClassifierCV + CalibratedClassifierCV
    (much faster than LogisticRegressionCV on 10k-50k features).
    """
    base_ridge = RidgeClassifierCV(alphas=np.logspace(-3, 3, 10))
    return make_pipeline(
        rocket_transformer,
        StandardScaler(with_mean=False),
        CalibratedClassifierCV(estimator=base_ridge, method='sigmoid', cv=3)
    )

class RevIN(nn.Module):
    """Use in forecasting; in classification only if absolute amplitude is non-discriminative."""
    def __init__(self, num_features: int, eps=1e-5, affine=True):
        super().__init__()
        self.num_features = num_features
        self.eps = eps
        self.affine = affine
        if self.affine:
            self.affine_weight = nn.Parameter(torch.ones(num_features))
            self.affine_bias = nn.Parameter(torch.zeros(num_features))

    def forward(self, x, mode: str):
        if mode == 'norm':
            self.mean = torch.mean(x, dim=1, keepdim=True).detach()
            self.stdev = torch.sqrt(torch.var(x, dim=1, keepdim=True, unbiased=False) + self.eps).detach()
            x = (x - self.mean) / self.stdev
            return x * self.affine_weight + self.affine_bias if self.affine else x
        elif mode == 'denorm':
            x = (x - self.affine_bias) / (self.affine_weight + self.eps) if self.affine else x
            return x * self.stdev + self.mean
        raise ValueError(f"Invalid mode: {mode}.")

class MovingAvg1D(nn.Module):
    def __init__(self, kernel_size):
        super().__init__()
        self.kernel_size = kernel_size
        self.avg = nn.AvgPool1d(kernel_size=kernel_size, stride=1, padding=0)

    def forward(self, x):
        front = x[:, 0:1, :].repeat(1, (self.kernel_size - 1) // 2, 1)
        end = x[:, -1:, :].repeat(1, self.kernel_size // 2, 1)
        x_pad = torch.cat([front, x, end], dim=1)
        return self.avg(x_pad.permute(0, 2, 1)).permute(0, 2, 1)

class DLinearForecaster(nn.Module):
    def __init__(self, seq_len, pred_len, n_channels, kernel_size=25, use_revin=True):
        super().__init__()
        self.use_revin = use_revin
        if use_revin:
            self.revin = RevIN(n_channels)
        self.decomp = MovingAvg1D(kernel_size)
        self.linear_seasonal = nn.Linear(seq_len, pred_len)
        self.linear_trend = nn.Linear(seq_len, pred_len)

    def forward(self, x):
        if self.use_revin:
            x = self.revin(x, 'norm')
        trend = self.decomp(x)
        seasonal = x - trend
        out = (self.linear_trend(trend.permute(0, 2, 1)) +
               self.linear_seasonal(seasonal.permute(0, 2, 1))).permute(0, 2, 1)
        return self.revin(out, 'denorm') if self.use_revin else out

class PatchTSTForecaster(nn.Module):
    def __init__(self, seq_len, pred_len, n_channels, patch_len=16, stride=8, d_model=128, nhead=4, num_layers=3, use_revin=True):
        super().__init__()
        assert seq_len >= patch_len, f"seq_len ({seq_len}) must be >= patch_len ({patch_len})"
        self.use_revin = use_revin
        if use_revin:
            self.revin = RevIN(n_channels)
        self.patch_len = patch_len
        self.stride = stride
        self.rem = (seq_len - patch_len) % stride
        self.n_patches = (seq_len - self.rem - patch_len) // stride + 1
        self.patch_proj = nn.Linear(patch_len, d_model)
        self.pos_embed = nn.Parameter(torch.randn(1, self.n_patches, d_model) * 0.02)
        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=d_model * 2,
            dropout=0.2, activation='gelu', batch_first=True, norm_first=True
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=num_layers, norm=nn.LayerNorm(d_model))
        self.head = nn.Linear(self.n_patches * d_model, pred_len)

    def forward(self, x):
        B, L, C = x.shape
        if self.use_revin:
            x = self.revin(x, 'norm')
        x = x.permute(0, 2, 1).reshape(B * C, L)
        if self.rem > 0:
            x = x[:, self.rem:]
        patches = x.unfold(dimension=-1, size=self.patch_len, step=self.stride)
        tokens = self.patch_proj(patches) + self.pos_embed
        enc_out = self.encoder(tokens).flatten(1)
        preds = self.head(enc_out).view(B, C, -1).permute(0, 2, 1)
        return self.revin(preds, 'denorm') if self.use_revin else preds

def forecast_loss(preds, targets, criterion=None):
    """preds: [B, pred_len, C]. A 2D [B, pred_len] target is accepted only when C == 1 (unsqueezed to [B, pred_len, 1]).
    Any other shape mismatch raises ValueError: nn.MSELoss with mismatched shapes silently broadcasts
    (emitting only a warning) and compares every timestep against all others."""
    if targets.ndim == preds.ndim - 1 and preds.size(-1) == 1:
        targets = targets.unsqueeze(-1)
    if preds.shape != targets.shape:
        raise ValueError(f"Incompatible shapes: preds {tuple(preds.shape)} vs targets {tuple(targets.shape)}")
    criterion = criterion if criterion is not None else nn.MSELoss()
    return criterion(preds, targets)

class FastSlidingWindowDataset(torch.utils.data.Dataset):
    """Sliding windows as zero-copy VIEWS (series is not duplicated in RAM; only requested windows are copied).

    series : [T, C] (or [T]) array, 3D [N_series, T, C] array, or list of [T_i, C] arrays.
             A 2D array is always interpreted as a single multivariate series [T, C]; for multiple
             univariate series in one array pass `series[:, :, None]` ([N_series, T, 1]) or a list of arrays.
             Pass segments ALREADY split per fold (purged split): windows never cross segment boundaries.
    window_len, horizon : input window length and forecast horizon (horizon=0 -> input only).
    stride : step between consecutive windows.
    labels : optional; either:
             - one scalar label per series in `series` (e.g., 1D array/list of length len(series));
             - or a list of [T_i] arrays (per-timestep labels; each window uses its last timestep's label).
    layout : 'LC' -> x [L, C] (DLinear, PatchTST); 'CL' -> x [C, L] (InceptionTime).
    Useful attributes: `n_series` (total input series) and `series_ids` (array [N_windows] mapping each
                       window to its parent series index, to pass with `n_series` to `aggregate_window_logits`).
    Returns x, or (x, y) with y [horizon, C], or (x, label)."""
    def __init__(self, series, window_len, horizon=0, stride=1, labels=None, layout='LC'):
        assert layout in ('LC', 'CL')
        assert not (horizon > 0 and labels is not None), "use either horizon (forecasting) or labels (classification)"
        if isinstance(series, np.ndarray) and series.ndim <= 2:
            segs = [series]
            if labels is not None:
                if np.ndim(labels) == 0:
                    raise ValueError(
                        "With a single 1D/2D series, `labels` cannot be a scalar: "
                        "pass `[label]` (series-level label) or a 1D array of length T (per-timestep labels). "
                        "If `series` holds multiple univariate series [N_series, T], pass `series[:, :, None]`."
                    )
                if np.ndim(labels) == 1 and len(labels) != 1:
                    if len(labels) != len(series):
                        raise ValueError(
                            f"Length of `labels` ({len(labels)}) != series length ({len(series)}). "
                            "If `series` is [N_series, T] (multiple univariate series), pass `series[:, :, None]`."
                        )
                    labels = [labels]
        else:
            segs = list(series)
        self.n_series = len(segs)
        self.window_len, self.horizon, self.stride, self.layout = window_len, horizon, stride, layout
        total = window_len + horizon
        self.views, self.label_entries, counts, valid_seg_ids = [], [], [], []
        for k, s in enumerate(segs):
            s = np.asarray(s, dtype=np.float32)
            if s.ndim == 1:
                s = s[:, None]
            if len(s) < total:
                warnings.warn(
                    f"Series {k} dropped: length {len(s)} < window_len + horizon ({total}).",
                    UserWarning
                )
                continue
            v = np.lib.stride_tricks.sliding_window_view(s, total, axis=0)[::stride]  # [N, C, total]
            self.views.append(v)
            counts.append(len(v))
            valid_seg_ids.append(k)
            if labels is not None:
                if np.ndim(labels[k]) == 0:
                    self.label_entries.append(('series', int(labels[k])))
                else:
                    self.label_entries.append(('step', np.asarray(labels[k])))
        self.cum = np.cumsum([0] + counts)
        self.series_ids = (
            np.concatenate([np.full(c, s_id, dtype=np.int64) for s_id, c in zip(valid_seg_ids, counts)])
            if counts else np.empty(0, dtype=np.int64)
        )

    def __len__(self):
        return int(self.cum[-1])

    def __getitem__(self, idx):
        if idx < 0:
            idx += len(self)
        k = int(np.searchsorted(self.cum, idx, side='right') - 1)
        j = int(idx - self.cum[k])
        w = self.views[k][j]                        # [C, total]
        x = w[:, :self.window_len]
        x = x.T if self.layout == 'LC' else x
        x = torch.as_tensor(np.array(x, order="C"))
        if self.horizon > 0:
            y = torch.as_tensor(np.array(w[:, self.window_len:].T, order="C"))  # [horizon, C]
            return x, y
        if self.label_entries:
            kind, lbl = self.label_entries[k]
            if kind == 'series':
                return x, lbl
            last = j * self.stride + self.window_len - 1
            return x, int(lbl[last])
        return x

def aggregate_window_logits(window_logits, series_ids, n_series=None):
    """Aggregates window logits (or probabilities) into one prediction per series (mean over windows)."""
    window_logits = np.asarray(window_logits)
    series_ids = np.asarray(series_ids, dtype=np.int64)
    if n_series is None:
        n_series = int(series_ids.max()) + 1
    out = np.zeros((n_series, window_logits.shape[1]), dtype=np.float64)
    counts = np.bincount(series_ids, minlength=n_series)
    if (counts == 0).any():
        raise ValueError("Some series have no associated windows.")
    np.add.at(out, series_ids, window_logits)
    return (out / counts[:, None]).astype(window_logits.dtype)


# =====================================================================
# ### F. Tabular Data
# =====================================================================
# =====================================================================
# 5. TABULAR: PLR + SWIGLU MLP (COMPLEMENT TO LIGHTGBM / CATBOOST)
# =====================================================================
class PLREmbeddings(nn.Module):
    def __init__(self, n_features, n_frequencies=16, d_embedding=16, sigma=0.1):
        super().__init__()
        self.freqs = nn.Parameter(torch.randn(n_features, n_frequencies) * sigma)
        self.weight = nn.Parameter(torch.empty(n_features, n_frequencies * 2, d_embedding))
        self.bias = nn.Parameter(torch.empty(n_features, d_embedding))
        bound = 1.0 / np.sqrt(n_frequencies * 2)
        nn.init.uniform_(self.weight, -bound, bound)
        nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, x):
        angles = 2.0 * np.pi * x.unsqueeze(-1) * self.freqs.unsqueeze(0)
        periodic = torch.cat([torch.sin(angles), torch.cos(angles)], dim=-1)
        out = F.relu(torch.einsum('bfk,fkd->bfd', periodic, self.weight) + self.bias)
        return out.flatten(1)

class SwiGLUBlock(nn.Module):
    def __init__(self, dim, hidden_dim, dropout=0.2):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fc_gate = nn.Linear(dim, hidden_dim * 2)
        self.drop = nn.Dropout(dropout)
        self.fc_out = nn.Linear(hidden_dim, dim)

    def forward(self, x):
        g, u = self.fc_gate(self.norm(x)).chunk(2, dim=-1)
        return x + self.drop(self.fc_out(F.silu(g) * u))

class TabularPLRNet(nn.Module):
    def __init__(self, n_features, out_dim, d_emb=16, hidden_dim=256, n_blocks=2, sigma=0.1, dropout=0.2):
        super().__init__()
        self.plr = PLREmbeddings(n_features, d_embedding=d_emb, sigma=sigma)
        self.in_proj = nn.Linear(n_features * d_emb, hidden_dim)
        self.blocks = nn.ModuleList([SwiGLUBlock(hidden_dim, hidden_dim, dropout) for _ in range(n_blocks)])
        self.head = nn.Sequential(nn.LayerNorm(hidden_dim), nn.Linear(hidden_dim, out_dim))

    def forward(self, x):
        h = self.in_proj(self.plr(x))
        for b in self.blocks:
            h = b(h)
        return self.head(h)


# =====================================================================
# ### G. Ensembling and Nested Post-Processing
# =====================================================================
# =====================================================================
# 6. ENSEMBLING & FULLY NESTED POST-PROCESSING (ZERO LEAK)
# =====================================================================
def softmax_np(logits):
    shifted = logits - np.max(logits, axis=-1, keepdims=True)
    exp_l = np.exp(shifted)
    return exp_l / np.sum(exp_l, axis=-1, keepdims=True)

def _test_probs(logits_m, T):
    """If logits_m is 3D (K, N, C), apply temperature T to each fold and average probabilities."""
    logits_m = np.asarray(logits_m)
    if logits_m.ndim == 3:
        return softmax_np(logits_m / T).mean(axis=0)
    return softmax_np(logits_m / T)

def fit_temperature(logits, y_true):
    n_classes = logits.shape[1]
    all_labels = np.arange(n_classes)
    def nll(t):
        p = np.clip(softmax_np(logits / t[0]), 1e-15, 1.0 - 1e-15)
        return log_loss(y_true, p, labels=all_labels)
    res = minimize(nll, x0=[1.0], bounds=[(0.05, 10.0)], method='L-BFGS-B')
    return float(res.x[0])

def caruana_logloss_weights(probs_list, y_true, n_iterations=50):
    """
    1. Passes labels=np.arange(n_classes) to log_loss (won't crash if a class is absent in a fold).
    2. On ties (e.g., identical models), splits weight equally 1/k across tied best candidates,
       yielding exact uniform weights [1/M, ..., 1/M] and 100% deterministic behavior.
    """
    n_models = len(probs_list)
    n_classes = probs_list[0].shape[1]
    all_labels = np.arange(n_classes)

    weights = np.zeros(n_models, dtype=np.float64)
    curr_sum = np.zeros_like(probs_list[0], dtype=np.float64)

    for i in range(n_iterations):
        scores = np.array([
            log_loss(y_true, np.clip((curr_sum + p) / (i + 1.0), 1e-15, 1.0 - 1e-15), labels=all_labels)
            for p in probs_list
        ])
        best_cands = np.where(np.isclose(scores, scores.min(), atol=1e-12))[0]
        share = 1.0 / len(best_cands)
        for idx in best_cands:
            weights[idx] += share
            curr_sum += share * probs_list[idx]

    return weights / float(n_iterations)

def fit_multipliers_coord_ascent(probs, y_true, n_cycles=2, average='weighted'):
    n_classes = probs.shape[1]
    mults = np.ones(n_classes, dtype=np.float64)
    best_f1 = f1_score(y_true, np.argmax(probs, axis=1), average=average)
    log_grid = np.geomspace(0.25, 4.0, 61)
    for _ in range(n_cycles):
        for c in range(n_classes):
            best_m = mults[c]
            for m in log_grid:
                mults[c] = m
                score = f1_score(y_true, np.argmax(probs * mults, axis=1), average=average)
                if score > best_f1 + 1e-6:
                    best_f1 = score
                    best_m = m
            mults[c] = best_m
    # Rescaling leaves argmax unchanged (scale-invariant); geometric mean centers values around 1.
    return mults / np.exp(np.mean(np.log(mults)))

def full_nested_postprocessing(oof_logits_list, test_logits_list, y_true, folds, min_f1_gain=0.002, average='weighted'):
    """
    average='weighted' matches the AN2DL 2026-27 metric (weighted F1); pass 'macro' for Macro-F1 challenges.
    Runs Temperature Scaling, Caruana Ensemble Selection, and Coordinate Ascent
    INSIDE the fold loop (fit on K-1 folds, evaluated on the held-out K-th fold) for zero-leak F1 estimation.
    If test_logits_list[m] is 3D (K, N_test, C) — one model per fold — mean_temps[m] is applied
    to each fold model before averaging probabilities (the 2D (N_test, C) fallback applies
    mean_temps[m] to pre-averaged logits).
    Note: a diversity mismatch vs. single-fold OOF still remains, because averaging probabilities
    across discordant fold models produces softer predictions; verify mean confidence and predicted
    class distribution between OOF and test on real data.
    """
    unique_folds = np.unique(folds)
    n_models = len(oof_logits_list)
    n_samples, n_classes = oof_logits_list[0].shape

    nested_blend_probs = np.zeros((n_samples, n_classes))
    nested_thresh_preds = np.zeros(n_samples, dtype=int)

    fold_weights = []
    fold_temps = []
    fold_mults = []

    for f in unique_folds:
        tr, va = (folds != f), (folds == f)
        # 1. Calibrate each model on the K-1 training folds
        temps_f = [fit_temperature(oof_logits_list[m][tr], y_true[tr]) for m in range(n_models)]
        tr_probs_list = [softmax_np(oof_logits_list[m][tr] / temps_f[m]) for m in range(n_models)]
        va_probs_list = [softmax_np(oof_logits_list[m][va] / temps_f[m]) for m in range(n_models)]

        # 2. Fit Caruana weights on the K-1 training folds
        w_f = caruana_logloss_weights(tr_probs_list, y_true[tr])
        tr_blend = sum(w * p for w, p in zip(w_f, tr_probs_list))
        va_blend = sum(w * p for w, p in zip(w_f, va_probs_list))
        nested_blend_probs[va] = va_blend

        # 3. Fit class multipliers on the K-1 training folds and apply to the held-out fold
        m_f = fit_multipliers_coord_ascent(tr_blend, y_true[tr], average=average)
        nested_thresh_preds[va] = np.argmax(va_blend * m_f, axis=1)

        fold_temps.append(temps_f)
        fold_weights.append(w_f)
        fold_mults.append(m_f)

    f1_blend_only = f1_score(y_true, np.argmax(nested_blend_probs, axis=1), average=average)
    f1_with_mults = f1_score(y_true, nested_thresh_preds, average=average)
    print(f"[Nested OOF] {average}-F1 Pure Blend: {f1_blend_only:.5f} | With Multipliers: {f1_with_mults:.5f}")

    # Test set aggregation
    mean_temps = np.mean(fold_temps, axis=0)
    mean_weights = np.mean(fold_weights, axis=0)
    mean_weights = mean_weights / mean_weights.sum()

    test_probs_list = [_test_probs(test_logits_list[m], mean_temps[m]) for m in range(n_models)]
    test_blend_probs = sum(w * p for w, p in zip(mean_weights, test_probs_list))

    # Apply multipliers only if nested CV gain exceeds noise threshold min_f1_gain
    if (f1_with_mults - f1_blend_only) >= min_f1_gain:
        final_mults = np.exp(np.mean(np.log(np.stack(fold_mults)), axis=0))
        return np.argmax(test_blend_probs * final_mults, axis=1), mean_weights, final_mults
    else:
        return np.argmax(test_blend_probs, axis=1), mean_weights, np.ones(n_classes)
