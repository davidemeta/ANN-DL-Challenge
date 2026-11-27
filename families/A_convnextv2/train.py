"""Family A: ConvNeXtV2-tiny fine-tuned on the cleaned train set, 5 shared folds.

Training: phase 1 trains only the head with the backbone frozen (LP-FT, `head_epochs`), phase 2
fine-tunes the whole network with layer-wise LR decay, warmup + cosine and EMA (`epochs`).
Choices and reasons: DECISIONS.md.

Usage (same command on Mac, Colab and Kaggle; paths come from src/env.py):
    python families/A_convnextv2/train.py --smoke                 # 1 epoch on a few batches
    python families/A_convnextv2/train.py --folds 0 --no-test     # quick 1-fold experiment
    python families/A_convnextv2/train.py                         # full 5-fold run + test
    python families/A_convnextv2/train.py --set color_mode=gray --run-name A_convnext_gray

Split: the team's shared files (built by Elia, eda/elia/03_validation_setup.ipynb):
    shared/folds.csv           the 2,995 clean images, 5 stratified folds (id, label, fold, ...)
    shared/val_shift_plan.csv  fixed hue rotation (deg) + blur (sigma) per image, 0 = none
For fold k, train on fold != k and validate on fold k. Reported F1 (official weighted F1):
    f1_clean     validation images as they are (main number)
    f1_shifted   same images with the distortion of val_shift_plan.csv (test-like)

Outputs in <out_dir>/<run>/ (playbook Part I section 1.4):
    oof_logits_<run>.npy   (2995, 10), D4 TTA, ordered like shared/folds.csv (NaN = fold not run)
    test_logits_<run>.npy  (K, 14736, 10), one model per fold, D4 TTA, not averaged
    submission_<run>.csv, fold<k>_* (per-fold logits, history, metrics, EMA checkpoint)
and runs/<run>.json in the repo. Folds already finished are skipped, so a run interrupted on
Colab resumes where it stopped.
"""
import argparse
import copy
import json
import math
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml
from sklearn.metrics import f1_score
from torchvision.transforms import v2

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT / 'src'))
sys.path.insert(0, str(HERE))

import an2dl_toolkit as tk  # noqa: E402
import gpu_aug  # noqa: E402
from data import N_CLASSES, build_cache, load_cache, load_tables, read_image  # noqa: E402
from env import ENV  # noqa: E402
from metric import score  # noqa: E402

DEVICE = tk.DEVICE


# ================================================================ config
def load_config(args):
    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    for item in args.set or []:
        key, value = item.split('=', 1)
        node, parts = cfg, key.split('.')
        for p in parts[:-1]:
            node = node[p]
        if parts[-1] not in node:
            raise KeyError(f'--set {key}: unknown key')
        node[parts[-1]] = yaml.safe_load(value)
    if args.run_name:
        cfg['run_name'] = args.run_name
    if args.epochs:
        cfg['epochs'] = args.epochs
    if args.no_test:
        cfg['predict_test'] = False
    if args.smoke:
        cfg.update(epochs=1, head_epochs=min(1, cfg['head_epochs']), eval_every=1,
                   save_checkpoints=False, batch_size=8,  # fits an 8 GB Mac
                   run_name=cfg['run_name'] + '_smoke')
    cfg['aug']['img_size'] = cfg['img_size']
    return cfg


def git_info():
    try:
        commit = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip()
        dirty = bool(subprocess.check_output(['git', 'status', '--porcelain', '--untracked-files=no'],
                                             cwd=ROOT, text=True).strip())
        return commit, dirty
    except Exception:
        return None, None


# ================================================================ data
def shift_params(plan, rows, n):
    """Hue shift (turns) and blur sigma indexed by train-cache row, from shared/val_shift_plan.csv."""
    hue = np.zeros(n, np.float32)
    sigma = np.zeros(n, np.float32)
    hue[rows] = plan['hue_deg'].values / 360.0
    sigma[rows] = plan['blur_sigma'].values
    return hue, sigma


class Batches:
    """Yields device batches straight from the uint8 array; augmentation runs on the device."""

    def __init__(self, X, idx, labels, batch_size, mean, std, transform,
                 shuffle=False, drop_last=False, gen=None):
        self.X, self.idx, self.labels = X, np.asarray(idx), labels
        self.bs, self.transform = batch_size, transform
        self.shuffle, self.drop_last, self.gen = shuffle, drop_last, gen
        self.mean = torch.tensor(mean, device=DEVICE).view(1, 3, 1, 1)
        self.std = torch.tensor(std, device=DEVICE).view(1, 3, 1, 1)

    def __len__(self):
        n = len(self.idx)
        return n // self.bs if self.drop_last else math.ceil(n / self.bs)

    def __iter__(self):
        order = self.idx
        if self.shuffle:
            order = order[torch.randperm(len(order), generator=self.gen).numpy()]
        n = len(self) * self.bs if self.drop_last else len(order)
        for s in range(0, n, self.bs):
            b = order[s:s + self.bs]
            x = torch.from_numpy(np.ascontiguousarray(self.X[b])).to(DEVICE)
            x = x.permute(0, 3, 1, 2).float().div_(255.0)
            x = (self.transform(x, b) - self.mean) / self.std
            x = x.contiguous(memory_format=torch.channels_last)
            if self.labels is None:
                yield (x,)
            else:
                yield x, torch.from_numpy(self.labels[b]).to(DEVICE)


def load_test(cfg, smoke):
    _, test_df = load_tables(ENV.data_dir)
    ids = test_df['id'].tolist()
    if smoke:  # a few PNGs, no 2.9 GB cache
        ids = ids[:64]
        return ids, np.stack([read_image(os.path.join(ENV.data_dir, 'test', f'{i}.png')) for i in ids])
    build_cache(ENV.data_dir, ENV.cache_dir, 'test')
    return ids, load_cache(ENV.cache_dir, 'test', mmap=True)


# ================================================================ eval
def f1_report(y_true, logits):
    pred = logits.argmax(1)
    per_class = f1_score(y_true, pred, labels=np.arange(N_CLASSES), average=None, zero_division=0)
    return score(y_true, pred), [round(float(v), 4) for v in per_class]


def evaluate(model, mk, va, y, sh_hue, sh_sigma, d4):
    """Returns (logits on the val rows, metrics dict). mk(idx, transform) builds a Batches."""
    gray = mk.gray
    clean = tk.predict_with_tta(model, mk(va, lambda x, b: gpu_aug.eval_transform(x, mk.size, gray)),
                                allow_hflip=False, d4=d4)

    def shifted_transform(x, b):
        x = gpu_aug.test_like(x, torch.from_numpy(sh_hue[b]).to(DEVICE),
                              torch.from_numpy(sh_sigma[b]).to(DEVICE))
        return gpu_aug.eval_transform(x, mk.size, gray)
    shifted = tk.predict_with_tta(model, mk(va, shifted_transform), allow_hflip=False, d4=d4)

    f1_clean, per_class = f1_report(y[va], clean)
    loss = torch.nn.functional.cross_entropy(torch.from_numpy(clean), torch.from_numpy(y[va])).item()
    return clean, {
        'f1_clean': round(f1_clean, 4),
        'f1_shifted': round(score(y[va], shifted.argmax(1)), 4),
        'val_loss': round(loss, 4),
        'f1_per_class': per_class,
    }


# ================================================================ fold
def run_fold(k, cfg, X, y, folds, rows, shift, test, run_dir, smoke):
    """rows[i] = train-cache row of folds.iloc[i]; X, y and shift are indexed by cache row."""
    tag = f'fold{k}'
    if (run_dir / f'{tag}_metrics.json').exists() and \
            (not cfg['predict_test'] or (run_dir / f'{tag}_test_logits.npy').exists()):
        print(f'[{tag}] already done, skipped', flush=True)
        return

    tk.seed_everything(cfg['seed'] + k)
    gen = torch.Generator().manual_seed(cfg['seed'] * 100 + k)
    tr = rows[folds['fold'].values != k]
    va_pos = np.where(folds['fold'].values == k)[0]  # positions in folds.csv (for the OOF file)
    if smoke:
        tr, va_pos = tr[:64], va_pos[:48]
    va = rows[va_pos]
    sh_hue, sh_sigma = shift
    print(f'[{tag}] train {len(tr)} | val {len(va)} ({int((sh_hue[va] != 0).sum())} hue, '
          f'{int((sh_sigma[va] > 0).sum())} blur in the shifted copy)', flush=True)

    model, optimizer = tk.create_timm_model_and_optimizer(
        cfg['model'], N_CLASSES, lr=cfg['lr'], weight_decay=cfg['weight_decay'],
        layer_decay=cfg['layer_decay'], pretrained=cfg['pretrained'],
        drop_rate=cfg['drop_rate'], drop_path_rate=cfg['drop_path_rate'])
    model = model.to(DEVICE, memory_format=torch.channels_last)
    mean, std = model.pretrained_cfg['mean'], model.pretrained_cfg['std']
    gray = cfg['color_mode'] == 'gray'

    def mk(idx, transform, labels=None, train=False):
        return Batches(X, idx, labels, cfg['batch_size'] if train else 2 * cfg['batch_size'],
                       mean, std, transform, shuffle=train, drop_last=train, gen=gen)
    mk.size, mk.gray = cfg['img_size'], gray

    train_batches = mk(tr, lambda x, b: gpu_aug.train_transform(x, cfg['aug'], gen, gray),
                       labels=y, train=True)
    criterion = tk.SoftTargetCrossEntropy(label_smoothing=cfg['label_smoothing'])
    mixup = v2.MixUp(alpha=cfg['mixup_alpha'], num_classes=N_CLASSES) if cfg['mixup_alpha'] > 0 else None
    scaler = tk.build_grad_scaler()
    history, t0 = [], time.time()

    # phase 1 (LP-FT): backbone frozen, only the head (norm + fc) learns
    if cfg['head_epochs'] > 0:
        head_params = list(model.get_submodule('head').parameters())
        head_ids = {id(p) for p in head_params}
        for p in model.parameters():
            p.requires_grad_(id(p) in head_ids)
        head_opt = torch.optim.AdamW(head_params, lr=cfg['head_lr'], weight_decay=cfg['weight_decay'])
        head_sched = tk.build_warmup_cosine_scheduler(
            head_opt, cfg['head_epochs'] * tk.steps_per_epoch(train_batches), warmup_frac=0.0)
        for epoch in range(1, cfg['head_epochs'] + 1):
            te = time.time()
            loss = tk.train_vision_epoch(model, None, train_batches, head_opt, head_sched,
                                         criterion, scaler, mixup_fn=mixup)
            row = {'phase': 'head', 'epoch': epoch, 'train_loss': round(loss, 4),
                   'lr': head_opt.param_groups[0]['lr'], 'sec': round(time.time() - te, 1)}
            if epoch == cfg['head_epochs']:
                _, m = evaluate(model, mk, va, y, sh_hue, sh_sigma, d4=False)
                row.update({kk: v for kk, v in m.items() if kk != 'f1_per_class'})
            history.append(row)
            print(f'[{tag}] ' + ' '.join(f'{kk}={v:.4g}' if isinstance(v, float) else f'{kk}={v}'
                                         for kk, v in row.items()), flush=True)
        for p in model.parameters():
            p.requires_grad_(True)
        del head_opt, head_sched

    # phase 2: full fine-tuning with layer-wise LR decay; EMA starts from the trained head
    total_steps = cfg['epochs'] * tk.steps_per_epoch(train_batches)
    scheduler = tk.build_warmup_cosine_scheduler(optimizer, total_steps, warmup_frac=cfg['warmup_frac'])
    ema = tk.WarmupEMA(model, max_decay=cfg['ema_decay'], device=DEVICE)
    for epoch in range(1, cfg['epochs'] + 1):
        te = time.time()
        loss = tk.train_vision_epoch(model, ema, train_batches, optimizer, scheduler,
                                     criterion, scaler, mixup_fn=mixup)
        row = {'phase': 'full', 'epoch': epoch, 'train_loss': round(loss, 4),
               'lr': optimizer.param_groups[-1]['lr'], 'sec': round(time.time() - te, 1)}
        if epoch % cfg['eval_every'] == 0 or epoch == cfg['epochs']:
            _, m = evaluate(ema.module, mk, va, y, sh_hue, sh_sigma, d4=False)
            row.update({kk: v for kk, v in m.items() if kk != 'f1_per_class'})
        history.append(row)
        print(f'[{tag}] ' + ' '.join(f'{kk}={v:.4g}' if isinstance(v, float) else f'{kk}={v}'
                                     for kk, v in row.items()), flush=True)
    pd.DataFrame(history).to_csv(run_dir / f'{tag}_history.csv', index=False)

    # final: EMA weights of the last epoch, D4 TTA (same TTA for OOF and test)
    val_logits, metrics = evaluate(ema.module, mk, va, y, sh_hue, sh_sigma, d4=True)
    np.save(run_dir / f'{tag}_val_idx.npy', va_pos)
    np.save(run_dir / f'{tag}_val_logits.npy', val_logits.astype(np.float32))
    if cfg['predict_test']:
        test_ids, X_test = test
        tb = Batches(X_test, np.arange(len(test_ids)), None, 2 * cfg['batch_size'], mean, std,
                     lambda x, b: gpu_aug.eval_transform(x, cfg['img_size'], gray))
        np.save(run_dir / f'{tag}_test_logits.npy',
                tk.predict_with_tta(ema.module, tb, allow_hflip=False, d4=True).astype(np.float32))
    if cfg['save_checkpoints']:
        torch.save(ema.module.state_dict(), run_dir / f'{tag}_ema.pt')
    metrics.update(fold=k, minutes=round((time.time() - t0) / 60, 1), n_train=len(tr))
    with open(run_dir / f'{tag}_metrics.json', 'w') as f:
        json.dump(metrics, f, indent=2)
    print(f'[{tag}] D4 TTA: f1_clean={metrics["f1_clean"]} f1_shifted={metrics["f1_shifted"]}',
          flush=True)
    del model, ema, optimizer
    if DEVICE.type == 'cuda':
        torch.cuda.empty_cache()


# ================================================================ assemble
def assemble(cfg, folds, run_dir, test_ids, args):
    run = cfg['run_name']
    oof = np.full((len(folds), N_CLASSES), np.nan, np.float32)
    done, fold_metrics, test_logits = [], [], []
    for k in sorted(folds['fold'].unique()):
        mpath = run_dir / f'fold{k}_metrics.json'
        if not mpath.exists():
            continue
        done.append(int(k))
        oof[np.load(run_dir / f'fold{k}_val_idx.npy')] = np.load(run_dir / f'fold{k}_val_logits.npy')
        fold_metrics.append(json.loads(mpath.read_text()))
        tpath = run_dir / f'fold{k}_test_logits.npy'
        if tpath.exists():
            test_logits.append(np.load(tpath))
    if not done:
        return
    np.save(run_dir / f'oof_logits_{run}.npy', oof)
    have = ~np.isnan(oof[:, 0])
    y = folds['label'].values.astype(np.int64)
    summary = {
        'run': run, 'family': 'A', 'owner': cfg.get('owner', ''), 'change': args.change or '',
        'model': cfg['model'], 'seed': cfg['seed'], 'folds_done': done,
        'cv_folds': [m['f1_clean'] for m in fold_metrics],
        'cv_oof': round(score(y[have], oof[have].argmax(1)), 4),
        'cv_shifted_folds': [m['f1_shifted'] for m in fold_metrics],
        'f1_per_class_oof': f1_report(y[have], oof[have])[1],
        'leaderboard': None,
        'runtime_min': round(sum(m['minutes'] for m in fold_metrics), 1),
        'platform': ENV.platform,
        'device': torch.cuda.get_device_name(0) if DEVICE.type == 'cuda' else DEVICE.type,
    }
    summary['commit'], summary['dirty'] = git_info()
    summary['config'] = cfg
    if len(test_logits) == len(done) and test_logits:
        tl = np.stack(test_logits)
        np.save(run_dir / f'test_logits_{run}.npy', tl)
        sub = pd.DataFrame({'id': test_ids, 'label': tl.mean(0).argmax(1)})
        sub.to_csv(run_dir / f'submission_{run}.csv', index=False)
        summary['submission'] = str(run_dir / f'submission_{run}.csv')
    text = json.dumps(summary, indent=2)
    (run_dir / f'{run}.json').write_text(text)
    if not args.smoke:
        (ROOT / 'runs').mkdir(exist_ok=True)
        (ROOT / 'runs' / f'{run}.json').write_text(text)
    print(f'\n{run}: folds {done} | OOF f1_clean={summary["cv_oof"]} '
          f'| f1_shifted per fold {summary["cv_shifted_folds"]}', flush=True)


# ================================================================ main
def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('--config', default=str(HERE / 'config.yaml'))
    ap.add_argument('--folds', type=int, nargs='*', help='default: all 5')
    ap.add_argument('--epochs', type=int)
    ap.add_argument('--run-name')
    ap.add_argument('--no-test', action='store_true', help='skip test prediction')
    ap.add_argument('--smoke', action='store_true', help='1 epoch on a few batches')
    ap.add_argument('--change', help='one line for runs/<run>.json: what changed vs the previous run')
    ap.add_argument('--set', action='append', metavar='KEY=VALUE', help='override a config key')
    args = ap.parse_args()
    cfg = load_config(args)

    print(ENV.describe(), flush=True)
    print(f'device={DEVICE} amp={tk.AMP_DTYPE} run={cfg["run_name"]}', flush=True)
    run_dir = Path(ENV.out_dir) / cfg['run_name']
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / 'config.yaml').write_text(yaml.safe_dump(cfg, sort_keys=False))

    folds = pd.read_csv(ROOT / 'shared' / 'folds.csv')
    plan = pd.read_csv(ROOT / 'shared' / 'val_shift_plan.csv')
    assert plan['id'].tolist() == folds['id'].tolist(), 'val_shift_plan.csv not aligned with folds.csv'
    train_df, _ = load_tables(ENV.data_dir)
    row_of = pd.Series(np.arange(len(train_df)), index=train_df['id'])
    assert folds['id'].isin(row_of.index).all(), 'folds.csv has ids missing from train.csv'
    rows = row_of[folds['id']].values  # cache row of every image in folds.csv
    y = train_df['label'].values.astype(np.int64)
    assert (y[rows] == folds['label'].values).all(), 'labels in folds.csv differ from train.csv'
    shift = shift_params(plan, rows, len(train_df))

    build_cache(ENV.data_dir, ENV.cache_dir, 'train')
    # in RAM on Colab/Kaggle (850 MB); memory-mapped on the Mac, whose GPU shares the RAM
    X = load_cache(ENV.cache_dir, 'train', mmap=ENV.platform == 'local')
    test = load_test(cfg, args.smoke) if cfg['predict_test'] else (None, None)

    for k in (args.folds if args.folds else sorted(folds['fold'].unique())):
        run_fold(int(k), copy.deepcopy(cfg), X, y, folds, rows, shift, test, run_dir, args.smoke)
    assemble(cfg, folds, run_dir, test[0], args)


if __name__ == '__main__':
    main()
