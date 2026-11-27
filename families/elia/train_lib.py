"""Training library for the image-classification experiments (families/elia).

One function, `run_experiment(cfg, data)`, trains a model on the shared folds, evaluates it on the
clean and shifted validation, predicts the test set and saves everything in outputs/<run_name>/.
The baseline (10_baseline.ipynb) is `run_experiment(baseline_config())`; every experiment changes
only a few entries of the configuration ("one thing at a time").

Augmentations are applied on the GPU, on whole batches, with an implementation (PyTorch) and
parameter ranges different from the shifted validation (PIL, fixed plan), so that the model learns a
general robustness and not the exact alteration used to evaluate it.
"""
import sys, os, glob, json, time, math, random
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor

# --- find common.py: repo (eda/elia), Mac working copy (elia_pc) or a Kaggle input -----------------
_HERE = Path(__file__).resolve().parent
_cands = [_HERE.parent.parent / "eda" / "elia", _HERE.parent.parent / "elia_pc"]
_cands += [Path(p).parent for p in glob.glob("/kaggle/input/**/common.py", recursive=True)]
sys.path.insert(0, str(next(p for p in _cands if (p / "common.py").exists())))

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
import timm.optim

import common as C

DEVICE = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
USE_AMP = DEVICE == "cuda"          # bf16 mixed precision on the NVIDIA GPU

# where to write: next to this file if possible, otherwise the Kaggle working directory
BASE_DIR = _HERE if os.access(_HERE, os.W_OK) and not Path("/kaggle/working").exists() else Path("/kaggle/working")
OUT_DIR = BASE_DIR / "outputs"
LOG_FILE = BASE_DIR / "experiments_log.csv"


# ============================================================================ #
# Configuration                                                                #
# ============================================================================ #
def baseline_config(**overrides):
    """The baseline of 10_baseline.ipynb; every key can be overridden."""
    cfg = dict(
        run_name="baseline_convnext_tiny_256",
        note="baseline: dihedral aug only, label smoothing 0.1",
        model="convnext_tiny.fb_in22k_ft_in1k",
        img_size=256,
        epochs=20, batch_size=64, lr=2e-4, weight_decay=0.05, warmup_epochs=1,
        label_smoothing=0.1, drop_path=0.1,
        folds=[0, 1, 2, 3, 4], seed=42, tta=True,
        # --- augmentations (training only); probability 0 = off ---
        hue_p=0.0, hue_max_deg=180.0,                 # hue rotation by U(-max, +max) degrees
        blur_p=0.0, blur_sigma=(0.1, 1.2),            # Gaussian blur, sigma in pixels
        bc_p=0.0, brightness=(0.8, 1.2), contrast=(0.8, 1.2),
        noise_p=0.0, noise_std=(0.0, 0.03),           # additive Gaussian noise, std on the [0, 1] scale
        save_models=False,                            # model weights are ~110 MB per fold
    )
    cfg.update(overrides)
    return cfg


def seed_everything(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)


# ============================================================================ #
# Data                                                                         #
# ============================================================================ #
def _load_many(ids, loader):
    with ThreadPoolExecutor(8) as ex:
        return np.stack(list(ex.map(lambda i: np.asarray(loader(i)), ids)))


_DATA = {}

def load_data(smoke_test=False):
    """Load (once per kernel) the clean training images, their shifted versions and the test ids."""
    key = "smoke" if smoke_test else "full"
    if key in _DATA:
        return _DATA[key]
    folds = pd.read_csv(C.SPLITS / "folds.csv")
    plan = pd.read_csv(C.SPLITS / "val_shift_plan.csv").set_index("id")
    test_ids = pd.read_csv(C.DATA / "sample_submission.csv").id.tolist()
    if smoke_test:
        folds = folds.groupby("fold").head(40).reset_index(drop=True)
        test_ids = test_ids[:32]
    t0 = time.time()
    data = dict(folds=folds, y=folds.label.to_numpy(), test_ids=test_ids,
                X_clean=_load_many(folds.id, C.load_image),
                X_shift=_load_many(folds.id, lambda i: C.load_shifted(i, plan)),
                X_test=None, smoke_test=smoke_test)
    print(f"loaded {len(folds)} training images (+ shifted versions) in {time.time() - t0:.0f}s")
    _DATA[key] = data
    return data


def load_test(data):
    if data["X_test"] is None:
        t0 = time.time()
        data["X_test"] = _load_many(data["test_ids"], C.load_image)
        print(f"loaded {len(data['X_test'])} test images in {time.time() - t0:.0f}s")
    return data["X_test"]


# ============================================================================ #
# Tensors, symmetries, augmentations                                           #
# ============================================================================ #
MEAN = torch.tensor([0.485, 0.456, 0.406], device=DEVICE).view(1, 3, 1, 1)
STD = torch.tensor([0.229, 0.224, 0.225], device=DEVICE).view(1, 3, 1, 1)


def to_float(x_uint8, img_size):
    """(B, H, W, 3) uint8 -> (B, 3, H, W) float in [0, 1] on the device, resized if needed."""
    x = torch.as_tensor(x_uint8, device=DEVICE).permute(0, 3, 1, 2).float() / 255.0
    if x.shape[-1] != img_size:
        x = F.interpolate(x, size=img_size, mode="bilinear", antialias=True, align_corners=False)
    return x


def normalize(x):
    return (x - MEAN) / STD


def dihedral(x, k):
    """Symmetry k in 0..7 of a (B, C, H, W) batch: k % 4 quarter turns, flipped if k >= 4."""
    x = torch.rot90(x, k % 4, dims=(2, 3))
    return torch.flip(x, dims=(3,)) if k >= 4 else x


def random_dihedral(x):
    ks = torch.randint(0, 8, (x.shape[0],), device=x.device)
    out = torch.empty_like(x)
    for k in range(8):
        m = ks == k
        if m.any():
            out[m] = dihedral(x[m], k)
    return out


def rgb_to_hsv(x):
    """(B, 3, H, W) RGB in [0, 1] -> h, s, v each (B, H, W), h in [0, 1)."""
    r, g, b = x[:, 0], x[:, 1], x[:, 2]
    maxc, _ = x.max(1)
    minc, _ = x.min(1)
    delta = maxc - minc
    safe = torch.where(delta > 0, delta, torch.ones_like(delta))
    h = torch.where(maxc == r, ((g - b) / safe) % 6,
        torch.where(maxc == g, (b - r) / safe + 2, (r - g) / safe + 4)) / 6
    h = torch.where(delta > 0, h, torch.zeros_like(h)) % 1.0
    s = torch.where(maxc > 0, delta / torch.where(maxc > 0, maxc, torch.ones_like(maxc)), torch.zeros_like(maxc))
    return h, s, maxc


def hsv_to_rgb(h, s, v):
    """Inverse of rgb_to_hsv: h, s, v (B, H, W) -> (B, 3, H, W)."""
    h6 = (h % 1.0) * 6
    i = torch.floor(h6)
    f = h6 - i
    p, q, t = v * (1 - s), v * (1 - s * f), v * (1 - s * (1 - f))
    i = i.long() % 6
    r = torch.stack([v, q, p, p, t, v], 1).gather(1, i.unsqueeze(1)).squeeze(1)
    g = torch.stack([t, v, v, q, p, p], 1).gather(1, i.unsqueeze(1)).squeeze(1)
    b = torch.stack([p, p, t, v, v, q], 1).gather(1, i.unsqueeze(1)).squeeze(1)
    return torch.stack([r, g, b], 1)


def rotate_hue(x, deg):
    """Rotate the hue of each image of the batch by deg[i] degrees (saturation and value unchanged)."""
    h, s, v = rgb_to_hsv(x)
    h = (h + (deg / 360.0).view(-1, 1, 1)) % 1.0
    return hsv_to_rgb(h, s, v)


def gaussian_blur(x, sigma, ks=9):
    """Separable Gaussian blur with a different sigma (pixels) for each image of the batch."""
    B, Ch, H, W = x.shape
    r = ks // 2
    coords = torch.arange(-r, r + 1, device=x.device, dtype=x.dtype)
    k = torch.exp(-coords.view(1, -1) ** 2 / (2 * sigma.view(-1, 1) ** 2))
    k = k / k.sum(1, keepdim=True)                                   # (B, ks)
    k = k.repeat_interleave(Ch, 0)                                   # (B*Ch, ks)
    y = x.reshape(1, B * Ch, H, W)
    y = F.conv2d(F.pad(y, (r, r, 0, 0), mode="reflect"), k.view(B * Ch, 1, 1, ks), groups=B * Ch)
    y = F.conv2d(F.pad(y, (0, 0, r, r), mode="reflect"), k.view(B * Ch, 1, ks, 1), groups=B * Ch)
    return y.view(B, Ch, H, W)


def _pick(n, p, device):
    return torch.rand(n, device=device) < p


def _uniform(n, lo, hi, device):
    return lo + (hi - lo) * torch.rand(n, device=device)


def augment(x, cfg):
    """Training augmentations on a (B, 3, H, W) batch in [0, 1]; order: hue -> brightness/contrast -> blur -> noise."""
    n, dev = x.shape[0], x.device
    if cfg["hue_p"] > 0:
        m = _pick(n, cfg["hue_p"], dev)
        if m.any():
            x[m] = rotate_hue(x[m], _uniform(int(m.sum()), -cfg["hue_max_deg"], cfg["hue_max_deg"], dev))
    if cfg["bc_p"] > 0:
        m = _pick(n, cfg["bc_p"], dev)
        if m.any():
            k = int(m.sum())
            xb = x[m] * _uniform(k, *cfg["brightness"], dev).view(-1, 1, 1, 1)
            gray = (0.299 * xb[:, 0] + 0.587 * xb[:, 1] + 0.114 * xb[:, 2]).mean((1, 2)).view(-1, 1, 1, 1)
            x[m] = (xb - gray) * _uniform(k, *cfg["contrast"], dev).view(-1, 1, 1, 1) + gray
    if cfg["blur_p"] > 0:
        m = _pick(n, cfg["blur_p"], dev)
        if m.any():
            x[m] = gaussian_blur(x[m], _uniform(int(m.sum()), *cfg["blur_sigma"], dev))
    if cfg["noise_p"] > 0:
        m = _pick(n, cfg["noise_p"], dev)
        if m.any():
            std = _uniform(int(m.sum()), *cfg["noise_std"], dev).view(-1, 1, 1, 1)
            x[m] = x[m] + torch.randn_like(x[m]) * std
    return x.clamp(0, 1)


# ============================================================================ #
# Model, prediction, schedule                                                  #
# ============================================================================ #
def make_model(cfg):
    return timm.create_model(cfg["model"], pretrained=True, num_classes=C.N_CLASSES,
                             drop_path_rate=cfg["drop_path"]).to(DEVICE)


@torch.no_grad()
def predict_proba(model, X, cfg, tta=True, batch_size=128):
    """Class probabilities (N, 10) for uint8 images X; with TTA: mean over the 8 symmetries."""
    model.eval()
    out = []
    for i in range(0, len(X), batch_size):
        x = normalize(to_float(X[i:i + batch_size], cfg["img_size"]))
        views = range(8) if tta else [0]
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=USE_AMP):
            p = sum(F.softmax(model(dihedral(x, k)).float(), dim=1) for k in views) / len(views)
        out.append(p.cpu().numpy())
    return np.concatenate(out)


def make_scheduler(optimizer, cfg, steps_per_epoch):
    """Linear warmup, then cosine decay to ~0, computed per optimisation step."""
    total = cfg["epochs"] * steps_per_epoch
    warm = cfg["warmup_epochs"] * steps_per_epoch
    def factor(step):
        if step < warm:
            return (step + 1) / warm
        return 0.5 * (1 + math.cos(math.pi * (step - warm) / max(1, total - warm)))
    return torch.optim.lr_scheduler.LambdaLR(optimizer, factor)


def _nll(p, y):
    return F.nll_loss(torch.log(torch.tensor(p) + 1e-8), torch.tensor(y)).item()


# ============================================================================ #
# One fold                                                                     #
# ============================================================================ #
def train_fold(cfg, data, k, verbose=True):
    seed_everything(cfg["seed"] + k)
    fold_ids = data["folds"].fold.to_numpy()
    tr_idx, va_idx = np.where(fold_ids != k)[0], np.where(fold_ids == k)[0]
    X, y = data["X_clean"], data["y"]
    model = make_model(cfg)
    optimizer = timm.optim.create_optimizer_v2(model, "adamw", lr=cfg["lr"], weight_decay=cfg["weight_decay"])
    scheduler = make_scheduler(optimizer, cfg, math.ceil(len(tr_idx) / cfg["batch_size"]))
    criterion = nn.CrossEntropyLoss(label_smoothing=cfg["label_smoothing"])
    y_va = y[va_idx]

    history = []
    for epoch in range(cfg["epochs"]):
        model.train()
        t0, losses = time.time(), []
        perm = np.random.permutation(tr_idx)
        for i in range(0, len(perm), cfg["batch_size"]):
            b = perm[i:i + cfg["batch_size"]]
            x = augment(random_dihedral(to_float(X[b], cfg["img_size"])), cfg)
            target = torch.as_tensor(y[b], device=DEVICE)
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=USE_AMP):
                loss = criterion(model(normalize(x)), target)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            scheduler.step()
            losses.append(loss.item())
        p_c = predict_proba(model, data["X_clean"][va_idx], cfg, tta=False)
        p_s = predict_proba(model, data["X_shift"][va_idx], cfg, tta=False)
        history.append({"fold": k, "epoch": epoch + 1, "train_loss": float(np.mean(losses)),
                        "val_loss_clean": _nll(p_c, y_va), "val_loss_shifted": _nll(p_s, y_va),
                        "f1_clean": C.evaluate(y_va, p_c.argmax(1))["f1_weighted"],
                        "f1_shifted": C.evaluate(y_va, p_s.argmax(1))["f1_weighted"],
                        "seconds": time.time() - t0})
        if verbose:
            h = history[-1]
            print(f"{cfg['run_name']} | fold {k} | epoch {h['epoch']:2d} | train loss {h['train_loss']:.3f} | "
                  f"F1 clean {h['f1_clean']:.3f} | F1 shifted {h['f1_shifted']:.3f} | {h['seconds']:.0f}s")

    p_clean = predict_proba(model, data["X_clean"][va_idx], cfg, tta=cfg["tta"])
    p_shift = predict_proba(model, data["X_shift"][va_idx], cfg, tta=cfg["tta"])
    return model, pd.DataFrame(history), va_idx, p_clean, p_shift


# ============================================================================ #
# One experiment (all folds)                                                   #
# ============================================================================ #
def run_dir(cfg):
    return OUT_DIR / cfg["run_name"]


def run_experiment(cfg, data, force=False):
    """Train all folds of cfg, save OOF + test predictions + submission, append to the log.

    If the experiment was already completed (results.json exists) it is not re-run, unless force=True.
    Returns the summary dict (also saved as results.json).
    """
    out = run_dir(cfg)
    if (out / "results.json").exists() and not force:
        print(f"{cfg['run_name']}: already done, loading results")
        return json.loads((out / "results.json").read_text())
    out.mkdir(parents=True, exist_ok=True)
    (out / "config.json").write_text(json.dumps(cfg, indent=2))

    folds, y = data["folds"], data["y"]
    oof_c = np.zeros((len(folds), C.N_CLASSES), np.float32)
    oof_s = np.zeros((len(folds), C.N_CLASSES), np.float32)
    X_test = load_test(data)
    test_probs, histories, fold_rows = [], [], []
    t_start = time.time()

    for k in cfg["folds"]:
        t0 = time.time()
        model, hist, va_idx, p_c, p_s = train_fold(cfg, data, k)
        oof_c[va_idx], oof_s[va_idx] = p_c, p_s
        r_c, r_s = C.evaluate(y[va_idx], p_c.argmax(1)), C.evaluate(y[va_idx], p_s.argmax(1))
        fold_rows.append({"fold": k, **C.summary_row(cfg["run_name"], r_c, r_s), "minutes": (time.time() - t0) / 60})
        histories.append(hist)
        test_probs.append(predict_proba(model, X_test, cfg, tta=cfg["tta"]))
        if cfg["save_models"]:
            torch.save(model.state_dict(), out / f"model_fold{k}.pt")
        print(f"\n{cfg['run_name']} | fold {k} with TTA: F1 clean {r_c['f1_weighted']:.4f} | "
              f"F1 shifted {r_s['f1_weighted']:.4f} | {(time.time() - t0) / 60:.1f} min\n")
        del model
        if DEVICE == "cuda":
            torch.cuda.empty_cache()

    # --- out-of-fold predictions (same format as the baseline) ---
    oof = pd.DataFrame({"id": folds.id, "label": y, "fold": folds.fold})
    for name, P in [("clean", oof_c), ("shifted", oof_s)]:
        for c in range(C.N_CLASSES):
            oof[f"p{c}_{name}"] = P[:, c]
    oof.to_csv(out / "oof.csv", index=False)
    pd.concat(histories).to_csv(out / "history.csv", index=False)

    # --- test: mean over the fold models ---
    test_probs = np.stack(test_probs)
    np.save(out / "test_probs.npy", test_probs)
    sub = pd.DataFrame({"id": data["test_ids"], "label": test_probs.mean(0).argmax(1)})
    sub.to_csv(out / ("submission_SMOKE_TEST.csv" if data["smoke_test"] else "submission.csv"), index=False)

    # --- summary + log ---
    done = folds.fold.isin(cfg["folds"]).to_numpy()
    res_c, res_s = C.evaluate(y[done], oof_c[done].argmax(1)), C.evaluate(y[done], oof_s[done].argmax(1))
    fr = pd.DataFrame(fold_rows)
    results = {"run_name": cfg["run_name"], "note": cfg["note"],
               "oof_f1_clean": res_c["f1_weighted"], "oof_f1_shifted": res_s["f1_weighted"],
               "cv_f1_clean_mean": fr.f1_clean.mean(), "cv_f1_clean_std": fr.f1_clean.std(),
               "cv_f1_shifted_mean": fr.f1_shifted.mean(), "cv_f1_shifted_std": fr.f1_shifted.std(),
               "f1_per_class_clean": res_c["f1_per_class"].round(4).to_dict(),
               "f1_per_class_shifted": res_s["f1_per_class"].round(4).to_dict(),
               "minutes": (time.time() - t_start) / 60}
    (out / "results.json").write_text(json.dumps(results, indent=2))

    if not data["smoke_test"]:
        now = pd.Timestamp.now().strftime("%Y-%m-%d %H:%M")
        rows = [{"date": now, "train_set": "clean (keep=True)", "kaggle": None, "note": cfg["note"], **r} for r in fold_rows]
        rows.append({"date": now, "fold": "OOF", "train_set": "clean (keep=True)",
                     **C.summary_row(cfg["run_name"], res_c, res_s), "kaggle": None, "note": cfg["note"]})
        log = pd.concat([pd.read_csv(LOG_FILE), pd.DataFrame(rows)]) if LOG_FILE.exists() else pd.DataFrame(rows)
        log.to_csv(LOG_FILE, index=False)

    print(f"{cfg['run_name']} done in {results['minutes']:.1f} min: OOF F1 clean {res_c['f1_weighted']:.4f} | "
          f"shifted {res_s['f1_weighted']:.4f}")
    return results


# ============================================================================ #
# Comparison of experiments                                                    #
# ============================================================================ #
def oof_scores(run_name):
    """Weighted and per-class F1 recomputed from outputs/<run_name>/oof.csv (works for the baseline too)."""
    oof = pd.read_csv(OUT_DIR / run_name / "oof.csv")
    y = oof.label.to_numpy()
    out = {}
    for name in ["clean", "shifted"]:
        P = oof[[f"p{c}_{name}" for c in range(C.N_CLASSES)]].to_numpy()
        out[name] = C.evaluate(y, P.argmax(1))
    return out


def compare(run_names, reference="baseline_convnext_tiny_256"):
    """Table of OOF F1 (clean / shifted / class 4) for several runs, with the difference from the reference."""
    ref = oof_scores(reference)
    rows = []
    for name in [reference] + [r for r in run_names if r != reference]:
        if not (OUT_DIR / name / "oof.csv").exists():
            continue
        s = oof_scores(name)
        rows.append({"run": name,
                     "F1 clean": s["clean"]["f1_weighted"], "F1 shifted": s["shifted"]["f1_weighted"],
                     "Δ clean": s["clean"]["f1_weighted"] - ref["clean"]["f1_weighted"],
                     "Δ shifted": s["shifted"]["f1_weighted"] - ref["shifted"]["f1_weighted"],
                     "gap (clean − shifted)": s["clean"]["f1_weighted"] - s["shifted"]["f1_weighted"],
                     "cigar clean": s["clean"]["f1_per_class"].iloc[4],
                     "cigar shifted": s["shifted"]["f1_per_class"].iloc[4]})
    return pd.DataFrame(rows).set_index("run")


def per_class(run_names, which="shifted"):
    """Per-class OOF F1 of several runs side by side (validation 'clean' or 'shifted')."""
    return pd.DataFrame({name: oof_scores(name)[which]["f1_per_class"] for name in run_names
                         if (OUT_DIR / name / "oof.csv").exists()})
