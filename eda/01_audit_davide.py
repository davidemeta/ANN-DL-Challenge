"""EDA + corruption audit (Davide) — written before looking at the teammate's EDA, compared with it afterwards.

Follows the Day 1-2 checklist of docs/AN2DL_Playbook_Part0_Challenge_2627.md (§0.4). This script needs no torch;
the CNN-embedding items are in eda/02_embeddings_davide.py, the test-distortion measurements in eda/03_test_distortions_davide.py.

Run (from the repo root, venv active):
    python eda/01_audit_davide.py --limit 200     # dry run on 200 images per split (separate folder)
    python eda/01_audit_davide.py                 # full run (~19k images, ~2.5 min)

Every run REBUILDS eda/outputs/davide/ from scratch (gitignored). Layout: see OUTPUT_README below,
also written to eda/outputs/davide/README.md. How to read each file: eda/EDA_GUIDE_davide.md.
Findings, decisions and conclusions: eda/00_eda_davide.ipynb (reads these outputs).

Categories (same detectors and thresholds on train and test, all in CONFIG):
  - alien   : "Area 51" intruder, an alien face pasted on the image (large pale-skin blob)
  - grainy  : brightened sky + strong colour noise; in train these are degraded copies of clean
              train images with labels swapped 2<->3 and 5<->6 (see 03_corruptions/grainy)
  - broken  : (train) missing-data blocks (dark flat 16x16 blocks > 5%); blank = whole image flat grey
  - tinted  : (test) galaxy core colour outside the 1-99% range of clean train (purple/pink or green)
  - clean   : none of the above
"""
# %% Imports and CONFIG
import argparse
import ctypes
import hashlib
import os
import shutil
import sys
import time
from multiprocessing import Pool

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFilter, ImageFont
from scipy import ndimage

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, 'src'))
from data import CLASS_NAMES, IMG_SIZE, N_CLASSES  # noqa: E402  (constants only; read_image would hide raw modes)

CONFIG = {
    'data_dir': os.path.join(ROOT, 'data'),
    'out_dir': os.path.join(ROOT, 'eda', 'outputs', 'davide'),
    'seed': 42,
    'workers': max(1, (os.cpu_count() or 2) - 1),
    'chunk_size': 200,
    'border_px': 16,             # outer ring = "sky" background (basic stats)
    'center_px': 64,             # central square = galaxy core (basic stats)
    'noise_ring_px': 24,         # ring used for the grain measure
    # category detectors
    'grain_chroma_noise': 24.0,  # grainy if std of high-passed (R-G) in the sky ring > this (grainy median ~34) ...
    'grain_sky_mean': 52.0,      # ... OR sky (border) mean luminance > this (train is bimodal, gap ~46-55) ...
    'grain_lum_noise': 10.0,     # ... AND luminance noise > this (keeps naturally bright-sky galaxies out)
    'alien_skin_blob': 0.06,     # alien if largest pale-skin blob covers > this fraction (train aliens start ~0.065, galaxies <= 0.05; test galaxies reach ~0.08 -> test hits are CANDIDATES)
    'tint_quantiles': (0.01, 0.99),  # test core green-chromaticity outside these quantiles of clean train -> tinted
    'blank_std': 4.0,            # blank = flat grey noise (3 images at std 2.01; next real image is 8.1)
    'flat_block_std': 3.5,       # broken (train only): a 16x16 block is "missing data" if dark (mean < 100) and flat (std < this) ...
    'flat_block_mean': 100.0,
    'broken_dark_flat': 0.05,    # ... and > 5% of the blocks are like that (rule from the teammate's EDA; real sky block std ~8).
                                 #     Not applied to test: blurred test skies are flat too (639 test hits, 579 of them smooth)
    'line_flag': 25.0,           # straight row/column artifact (trail, colour line) if line_score > this (informative only)
    'smooth_quantile': 0.01,     # smooth if luminance noise < this quantile of clean train (test looks blurred/denoised)
    'sat_frac': 0.05,
    # duplicates (dHash, exact)
    'dhash_size': 16,
    'dhash_max_dist': 24,
    'dup_confirm_mad': 6.0,
    # grainy -> source matching (blurred 32x32 standardized thumbnails, cosine, max over D4)
    'match_topk': 5,             # thumbnail candidates per grainy image (look-alikes reach cos 0.99, so not proof)
    'ident_confirm': 0.70,       # star-field correlation (galaxy masked): same object if >= this
    'ident_none': 0.50,          # no source if best < this (different galaxies: max ~0.64 on calibration)
    # outputs
    'adv_folds': 5,
    'sheet_per_page': 100,       # thumbnails per contact-sheet page (10 x 10)
    'thumb_px': 128,
    'overview_per_class': 12,
    'test_random_folder': 300,
    'copy_images': True,         # APFS clones (no extra disk) of the PNGs into images/
}

# Decisions taken BY EYE (Davide, 2026-11-26). Applied on every run; logged in eda/00_eda_davide.ipynb §7.
MANUAL = {
    # grainy detector false positive: bright spiral halo, not noise
    'not_grainy': ['train_01849'],
    # grainy copy missed by the detector (galaxy fills the sky ring); same galaxy as grainy 01368 / 01861
    'force_grainy': ['train_00851'],
    # near-duplicate pairs with different labels: key = copy with the WRONG label, value = copy kept
    'wrong_label_duplicates': {'train_00854': 'train_00713', 'train_00284': 'train_01821', 'train_00944': 'train_02152'},
    # 'uncertain' grainy -> source pairs checked by eye: same galaxy (grainy id -> clean source id)
    'confirmed_pairs_by_eye': {'train_01992': 'train_00683', 'train_00075': 'train_04081',
                               'train_01000': 'train_02570', 'train_01894': 'train_02166'},
    # test 'smooth' images look blurred (no sharp galaxy details) -> use blur augmentation
    'test_smooth_is_blur': True,
    # alien detector false positive: ring spiral whose yellow halo looks like skin (found by the teammate's EDA)
    'not_alien': ['train_01971'],
}

OUTPUT_README = """# eda/outputs/davide (generated by eda/01_audit_davide.py, rebuilt on every run)

Explanation of every file: eda/EDA_GUIDE_davide.md. Narrative + conclusions: eda/00_eda_davide.ipynb.

| folder | content |
|---|---|
| `01_overview/` | label counts, 12 clean images per class, mean image + FFT, shape/colour per class |
| `02_class_sheets/` | every train image by class; frame orange=grainy, red=alien, blue=wrong-label duplicate, black=blank, grey=broken |
| `03_corruptions/` | `aliens/`, `grainy/` (copies of clean images with swapped labels), `duplicates/`, `lines/` (straight-line artifacts) |
| `04_test/` | what the test looks like: random, tinted purple/green, blurred, blank |
| `05_shift/` | train vs test statistics, clean train images least like test |
| `06_embeddings/` | CNN embeddings (written by eda/02_embeddings_davide.py): UMAP maps, k-NN label check, linear probe, outliers |
| `07_test_distortions/` | test hue rotation and blur measured for augmentation (written by eda/03_test_distortions_davide.py) |
| `csv/` | `cleaning_proposal_davide.csv` (keep/drop + dup_group per train image), `per_image.csv` (category + stats of all 19,064 images), `grainy_matches.csv`, `near_duplicates.csv` |
| `images/` | single PNGs: `train_by_class/<c>_<name>/<category>/`, `test/<group>/`, `review/` (pairs a/b) |
"""

CAT_COLORS = {'clean': None, 'grainy': (255, 140, 0), 'alien': (220, 30, 30), 'tinted': (160, 60, 220), 'blank': (0, 0, 0),
              'wrong_label_dup': (30, 120, 220), 'broken': (120, 120, 120)}
CAT_LETTER = {'clean': '', 'grainy': 'g', 'alien': 'a', 'blank': 'b', 'wrong_label_dup': 'w', 'broken': 'x'}

# %% Per-image worker (top-level for macOS spawn)


def _d4_views(a):
    out = []
    for k in range(4):
        r = np.rot90(a, k)
        out.append(r)
        out.append(np.fliplr(r))
    return out


def _dhash_bits(gray_u8, size):
    g = np.asarray(Image.fromarray(gray_u8).resize((size + 1, size), Image.BILINEAR), dtype=np.int16)
    return np.packbits((g[:, 1:] > g[:, :-1]).ravel()).view(np.uint64)


def _corr(a, c):
    sa, sc = a.std(), c.std()
    return float(((a - a.mean()) * (c - c.mean())).mean() / (sa * sc)) if sa > 0 and sc > 0 else 1.0


def analyze_chunk(task):
    """task = (split, [(img_id, label), ...], data_dir, cfg) -> rows, dhashes, thumbs, partial sums."""
    split, items, data_dir, cfg = task
    bp, cp, hs, nr = cfg['border_px'], cfg['center_px'], cfg['dhash_size'], cfg['noise_ring_px']
    rows, hashes, thumbs, sums = [], [], [], {}
    yy, xx = np.mgrid[-IMG_SIZE // 2:IMG_SIZE // 2, -IMG_SIZE // 2:IMG_SIZE // 2]
    radius = np.sqrt(yy ** 2 + xx ** 2)
    ring = np.ones((IMG_SIZE, IMG_SIZE), bool); ring[bp:-bp, bp:-bp] = False
    nring = np.ones((IMG_SIZE, IMG_SIZE), bool); nring[nr:-nr, nr:-nr] = False
    c0 = (IMG_SIZE - cp) // 2
    for img_id, label in items:
        path = os.path.join(data_dir, split, f'{img_id}.png')
        row = {'id': img_id, 'split': split, 'label': label, 'file_bytes': os.path.getsize(path)}
        try:
            with Image.open(path) as im:
                row['raw_mode'], (row['raw_w'], row['raw_h']) = im.mode, im.size
                pil = im.convert('RGB')
        except Exception as e:
            row.update(raw_mode='ERR', raw_w=-1, raw_h=-1, readable=False, error=repr(e))
            rows.append(row); hashes.append(np.zeros((8, hs * hs // 64), np.uint64)); thumbs.append(np.zeros(1024, np.float32))
            continue
        row['readable'] = True
        if pil.size != (IMG_SIZE, IMG_SIZE):
            pil = pil.resize((IMG_SIZE, IMG_SIZE), Image.BILINEAR)
        rgb = np.asarray(pil, dtype=np.uint8)
        row['pixel_md5'] = hashlib.md5(rgb.tobytes()).hexdigest()
        f = rgb.astype(np.float32)
        r, g, b = f[..., 0], f[..., 1], f[..., 2]
        gray = 0.299 * r + 0.587 * g + 0.114 * b
        center = f[c0:c0 + cp, c0:c0 + cp]
        for i, ch in enumerate('rgb'):
            row[f'mean_{ch}'] = float(f[..., i].mean())
            row[f'std_{ch}'] = float(f[..., i].std())
            row[f'border_{ch}'] = float(f[..., i][ring].mean())
            row[f'center_{ch}'] = float(center[..., i].mean())
        row['gray_std'] = float(gray.std())
        # missing data: share of 16x16 blocks that are dark and flat (teammate's rule)
        blk = gray.reshape(16, 16, 16, 16)
        row['dark_flat'] = float(((blk.std(axis=(1, 3)) < cfg['flat_block_std']) & (blk.mean(axis=(1, 3)) < cfg['flat_block_mean'])).mean())
        # straight-line artifact: largest deviation of a row/column channel mean from its +-3 neighbours (median)
        best = 0.0
        for ax in (1, 0):
            prof = f.mean(axis=ax)
            pad = np.pad(prof, ((3, 3), (0, 0)), mode='edge')
            nb = np.median(np.stack([pad[k:k + IMG_SIZE] for k in (0, 1, 2, 4, 5, 6)]), axis=0)
            best = max(best, float(np.abs(prof - nb).max()))
        row['line_score'] = best
        lap = -4 * gray[1:-1, 1:-1] + gray[:-2, 1:-1] + gray[2:, 1:-1] + gray[1:-1, :-2] + gray[1:-1, 2:]
        row['lap_var'] = float(lap.var())
        row['sat_hi_frac'] = float((rgb == 255).any(-1).mean())
        rg, yb = r - g, 0.5 * (r + g) - b
        row['colorfulness'] = float(np.sqrt(rg.std() ** 2 + yb.std() ** 2) + 0.3 * np.sqrt(rg.mean() ** 2 + yb.mean() ** 2))
        row['corr_rg'], row['corr_gb'] = _corr(r, g), _corr(g, b)
        mag = np.abs(np.fft.fftshift(np.fft.fft2(gray - gray.mean())))
        row['hf_energy'] = float((mag[radius > IMG_SIZE // 4] ** 2).sum() / max((mag ** 2).sum(), 1e-9))
        # grain: high-pass noise in the sky ring (chroma = R-G, luminance)
        hp = f - ndimage.uniform_filter(f, size=(5, 5, 1))
        row['chroma_noise'] = float(np.std((hp[..., 0] - hp[..., 1])[nring]))
        row['lum_noise'] = float(np.std(hp.mean(-1)[nring]))
        # alien: largest pale-skin blob (HSV range measured on known aliens)
        hsv = np.asarray(pil.convert('HSV')).astype(np.int16)
        skin = (hsv[..., 0] >= 20) & (hsv[..., 0] <= 60) & (hsv[..., 1] >= 25) & (hsv[..., 1] <= 110) & (hsv[..., 2] >= 140) & (hsv[..., 2] <= 235)
        row['skin_frac'] = float(skin.mean())
        lab, n = ndimage.label(ndimage.binary_opening(skin, iterations=2))
        row['skin_blob'] = float(np.bincount(lab.ravel())[1:].max() / skin.size) if n else 0.0
        # galaxy core colour: chromaticity of the brightest 20% pixels of the central 64x64
        c = center.reshape(-1, 3); w = c.sum(1); top = c[w >= np.percentile(w, 80)].mean(0)
        chrom = top / max(top.sum(), 1e-6)
        row['core_r'], row['core_g'], row['core_b'] = (float(x) for x in chrom)
        # galaxy shape: connected component under the centre above sky + 3 sigma
        sky = gray[ring]; thr = np.median(sky) + 3 * 1.4826 * np.median(np.abs(sky - np.median(sky))) + 1
        lab, n = ndimage.label(ndimage.binary_opening(gray > thr, iterations=1))
        cl = lab[IMG_SIZE // 2 - 4:IMG_SIZE // 2 + 4, IMG_SIZE // 2 - 4:IMG_SIZE // 2 + 4].ravel()
        cl = cl[cl > 0]
        if len(cl):
            m = lab == np.bincount(cl).argmax()
            ys, xs = np.nonzero(m); wts = gray[m]
            cy, cx = np.average(ys, weights=wts), np.average(xs, weights=wts)
            cov = np.cov(np.stack([ys - cy, xs - cx]), aweights=wts)
            ev = np.sort(np.linalg.eigvalsh(cov))
            row['gal_area'] = float(m.mean())
            row['gal_axis_ratio'] = float(np.sqrt(max(ev[0], 1e-6) / max(ev[1], 1e-6)))
            row['gal_offset'] = float(np.hypot(cy - IMG_SIZE / 2, cx - IMG_SIZE / 2))
        else:
            row['gal_area'], row['gal_axis_ratio'], row['gal_offset'] = 0.0, np.nan, np.nan
        rows.append(row)
        g8 = np.clip(gray, 0, 255).astype(np.uint8)
        hashes.append(np.stack([_dhash_bits(np.ascontiguousarray(v), hs) for v in _d4_views(g8)]))
        t = np.asarray(pil.convert('L').filter(ImageFilter.GaussianBlur(3)).resize((32, 32), Image.BILINEAR), np.float32)
        thumbs.append(((t - t.mean()) / (t.std() + 1e-6)).ravel())
        key = f'{split}_{label}'
        if key not in sums:
            sums[key] = [np.zeros((IMG_SIZE, IMG_SIZE, 3), np.float64), np.zeros((IMG_SIZE, IMG_SIZE), np.float64), 0]
        sums[key][0] += f; sums[key][1] += np.log1p(mag); sums[key][2] += 1
    return rows, np.stack(hashes), np.stack(thumbs).astype(np.float32), sums


EXTRA_COLS = ['dark_flat', 'line_score']  # in per_image.csv, kept out of the adversarial features

STAT_COLS = [
    'mean_r', 'mean_g', 'mean_b', 'std_r', 'std_g', 'std_b', 'border_r', 'border_g', 'border_b',
    'center_r', 'center_g', 'center_b', 'gray_std', 'lap_var', 'sat_hi_frac', 'colorfulness',
    'corr_rg', 'corr_gb', 'hf_energy', 'chroma_noise', 'lum_noise', 'skin_blob',
    'core_r', 'core_g', 'core_b', 'gal_area', 'gal_axis_ratio', 'gal_offset',
]

# unit shown on chart axes (pixel values are 8-bit levels 0-255)
UNITS = {
    **{f'{k}_{c}': 'level 0-255' for k in ('mean', 'std', 'border', 'center') for c in 'rgb'},
    'gray_std': 'level 0-255', 'lap_var': 'variance, level^2', 'sat_hi_frac': 'fraction 0-1',
    'colorfulness': 'Hasler-Suesstrunk, level', 'corr_rg': 'Pearson r', 'corr_gb': 'Pearson r',
    'hf_energy': 'fraction of FFT energy', 'chroma_noise': 'std, level 0-255', 'lum_noise': 'std, level 0-255',
    'skin_blob': 'fraction of image', 'core_r': 'chromaticity 0-1', 'core_g': 'chromaticity 0-1',
    'core_b': 'chromaticity 0-1', 'gal_area': 'fraction of image', 'gal_axis_ratio': 'minor/major 0-1',
    'gal_offset': 'pixels',
}

# %% Image / file helpers
_FONT = None


def _font():
    global _FONT
    if _FONT is None:
        try:
            _FONT = ImageFont.load_default(size=11)
        except TypeError:
            _FONT = ImageFont.load_default()
    return _FONT


def img_path(data_dir, split, img_id):
    return os.path.join(data_dir, split, f'{img_id}.png')


def make_sheet(data_dir, entries, path, title='', ncols=10, thumb=128):
    """Fast PIL contact sheet. entries: list of (split, id, caption, frame_rgb_or_None)."""
    if not entries:
        return
    cap_h, pad, head = 14, 4, (24 if title else 0)
    cell_w, cell_h = thumb + pad, thumb + cap_h + pad
    nrows = int(np.ceil(len(entries) / ncols))
    sheet = Image.new('RGB', (ncols * cell_w + pad, head + nrows * cell_h + pad), (255, 255, 255))
    draw = ImageDraw.Draw(sheet)
    if title:
        draw.text((pad, 5), title.replace('—', '-'), fill=(0, 0, 0), font=_font())  # default font has no em dash
    for k, (split, img_id, cap, frame) in enumerate(entries):
        x = pad + (k % ncols) * cell_w
        y = head + (k // ncols) * cell_h
        with Image.open(img_path(data_dir, split, img_id)) as im:
            th = im.convert('RGB').resize((thumb, thumb), Image.BILINEAR)
        sheet.paste(th, (x, y + cap_h))
        if frame:
            draw.rectangle([x, y + cap_h, x + thumb - 1, y + cap_h + thumb - 1], outline=frame, width=4)
        draw.text((x, y), cap, fill=(0, 0, 0), font=_font())
    sheet.save(path, optimize=True)


_LIBC = None


def clone_file(src, dst):
    """APFS clone (copy-on-write, no extra disk; edits never touch the dataset); plain copy fallback."""
    global _LIBC
    try:
        if _LIBC is None:
            _LIBC = ctypes.CDLL('libc.dylib', use_errno=True)
        if _LIBC.clonefile(src.encode(), dst.encode(), 0) == 0:
            return
    except OSError:
        pass
    shutil.copy2(src, dst)


def place(data_dir, split, img_id, folder, name=None):
    os.makedirs(folder, exist_ok=True)
    clone_file(img_path(data_dir, split, img_id), os.path.join(folder, name or f'{img_id}.png'))


def hamming_min_d4(q, refs_all_views):
    best = None
    for v in range(refs_all_views.shape[1]):
        d = np.bitwise_count(q[:, None, :] ^ refs_all_views[None, :, v, :]).sum(-1, dtype=np.int32)
        best = d if best is None else np.minimum(best, d)
    return best


def small_gray(data_dir, split, img_id):
    with Image.open(img_path(data_dir, split, img_id)) as im:
        return np.asarray(im.convert('L').resize((64, 64), Image.BILINEAR), dtype=np.float32)


def thumb_views(T):
    """T: (N, 1024) standardized thumbnails -> list of 8 (N, 1024) L2-normalised D4 views."""
    T = T / (np.linalg.norm(T, axis=1, keepdims=True) + 1e-9)
    t = T.reshape(-1, 32, 32)
    out = []
    for k in range(4):
        r = np.rot90(t, k, axes=(1, 2))
        out.append(np.ascontiguousarray(r.reshape(len(T), -1)))
        out.append(np.ascontiguousarray(r[:, :, ::-1].reshape(len(T), -1)))
    return out


def best_match(views, q_idx, ref_idx, exclude_self=False):
    """Max cosine over D4 between view-0 of queries and all views of refs -> (best_cos, best_ref_index)."""
    best = np.full(len(q_idx), -2.0); arg = np.zeros(len(q_idx), int)
    for s0 in range(0, len(q_idx), 1024):
        q = q_idx[s0:s0 + 1024]
        S = np.full((len(q), len(ref_idx)), -2.0, np.float32)
        for v in views:
            np.maximum(S, views[0][q] @ v[ref_idx].T, out=S)
        if exclude_self:
            S[q[:, None] == ref_idx[None, :]] = -2.0
        a = S.argmax(1)
        best[s0:s0 + len(q)] = S[np.arange(len(q)), a]; arg[s0:s0 + len(q)] = ref_idx[a]
    return best, arg


def topk_match(views, q_idx, ref_idx, k):
    """Top-k refs by max cosine over D4 -> (cos (n,k), ref indices (n,k))."""
    cos_out, idx_out = [], []
    for s0 in range(0, len(q_idx), 512):
        q = q_idx[s0:s0 + 512]
        S = np.full((len(q), len(ref_idx)), -2.0, np.float32)
        for v in views:
            np.maximum(S, views[0][q] @ v[ref_idx].T, out=S)
        top = np.argsort(-S, axis=1)[:, :k]
        cos_out.append(np.take_along_axis(S, top, 1)); idx_out.append(ref_idx[top])
    return np.concatenate(cos_out), np.concatenate(idx_out)


_YY, _XX = np.mgrid[:128, :128]
_STAR_MASK = np.hypot(_YY - 64, _XX - 64) > 20  # ignore the galaxy, keep the surrounding star field


def _star_map(data_dir, split, img_id):
    """Point sources at 128 px (median-denoised minus blurred background): an identity fingerprint
    that survives the brightening + noise of the grainy copies."""
    with Image.open(img_path(data_dir, split, img_id)) as im:
        g = im.convert('L').resize((128, 128), Image.BILINEAR).filter(ImageFilter.MedianFilter(3))
    a = np.asarray(g, np.float32) - np.asarray(g.filter(ImageFilter.GaussianBlur(4)), np.float32)
    return np.clip(a, 0, None)


def ident_task(task):
    """task = (data_dir, (split, id), [(split, id), ...]) -> best star-field correlation over D4 for each ref."""
    data_dir, (sa, ia), refs = task
    x = _star_map(data_dir, sa, ia)[_STAR_MASK]
    x = x - x.mean()
    out = []
    for sb, ib in refs:
        best = -1.0
        for v in _d4_views(_star_map(data_dir, sb, ib)):
            y = v[_STAR_MASK]; y = y - y.mean()
            best = max(best, float((x * y).sum() / np.sqrt((x * x).sum() * (y * y).sum() + 1e-9)))
        out.append(best)
    return out


def savefig(fig, path, dpi=100):
    fig.tight_layout(); fig.savefig(path, dpi=dpi); plt.close(fig)


def short(c):
    return CLASS_NAMES[c].split('_')[0]


# %% Main


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--limit', type=int, default=0, help='images per split (0 = all); ~200 for a dry run')
    ap.add_argument('--no-copy', action='store_true', help='skip the images/ folders')
    args = ap.parse_args()
    cfg = dict(CONFIG)
    if args.no_copy:
        cfg['copy_images'] = False
    rng = np.random.default_rng(cfg['seed'])
    data_dir = cfg['data_dir']
    out = cfg['out_dir'] + (f'_dryrun_{args.limit}' if args.limit else '')
    # rebuild from scratch: the folder only holds generated files; 06_/07_ belong to scripts 02_/03_
    assert out.startswith(os.path.join(ROOT, 'eda', 'outputs')), out
    if os.path.isdir(out):
        for e in os.listdir(out):
            if e not in ('06_embeddings', '07_test_distortions'):
                p_ = os.path.join(out, e)
                shutil.rmtree(p_) if os.path.isdir(p_) else os.remove(p_)
    D = {k: os.path.join(out, v) for k, v in {
        'ov': '01_overview', 'sheets': '02_class_sheets', 'alien': '03_corruptions/aliens',
        'grainy': '03_corruptions/grainy', 'dups': '03_corruptions/duplicates', 'test': '04_test',
        'shift': '05_shift', 'csv': 'csv', 'img': 'images'}.items()}
    for p in D.values():
        os.makedirs(p, exist_ok=True)
    with open(os.path.join(out, 'README.md'), 'w') as fh:
        fh.write(OUTPUT_README)
    t0 = time.time()
    report = {}

    # ---- 1. Tables and file integrity ----
    train_df = pd.read_csv(os.path.join(data_dir, 'train.csv'))
    test_df = pd.read_csv(os.path.join(data_dir, 'sample_submission.csv'))[['id']]
    test_df['label'] = -1
    integ = {
        'n_train_csv': len(train_df), 'n_test_csv': len(test_df),
        'train_dup_ids': int(train_df['id'].duplicated().sum()), 'test_dup_ids': int(test_df['id'].duplicated().sum()),
        'labels_out_of_range': int((~train_df['label'].between(0, N_CLASSES - 1)).sum()),
        'labels_nan': int(train_df['label'].isna().sum()),
    }
    for split, df in [('train', train_df), ('test', test_df)]:
        on_disk = {os.path.splitext(f)[0] for f in os.listdir(os.path.join(data_dir, split)) if not f.startswith('.')}
        integ[f'{split}_files_on_disk'] = len(on_disk)
        integ[f'{split}_csv_missing_file'] = sorted(set(df['id']) - on_disk)[:20]
        integ[f'{split}_file_not_in_csv'] = sorted(on_disk - set(df['id']))[:20]
    report['integrity'] = integ
    if args.limit:
        train_df = train_df.sample(min(args.limit, len(train_df)), random_state=cfg['seed']).sort_index()
        test_df = test_df.sample(min(args.limit, len(test_df)), random_state=cfg['seed']).sort_index()

    # ---- 2. Per-image pass (parallel) ----
    tasks = []
    for split, df in [('train', train_df), ('test', test_df)]:
        items = list(zip(df['id'], df['label'].astype(int)))
        tasks += [(split, items[i:i + cfg['chunk_size']], data_dir, cfg) for i in range(0, len(items), cfg['chunk_size'])]
    rows, hash_list, thumb_list, sums = [], [], [], {}
    try:
        from tqdm import tqdm
    except ImportError:
        tqdm = lambda x, **k: x  # noqa: E731
    with Pool(cfg['workers']) as pool:
        for r, h, th, s in tqdm(pool.imap(analyze_chunk, tasks), total=len(tasks), desc='images'):
            rows += r; hash_list.append(h); thumb_list.append(th)
            for k, (a, b, n) in s.items():
                if k not in sums:
                    sums[k] = [a, b, n]
                else:
                    sums[k][0] += a; sums[k][1] += b; sums[k][2] += n
    stats = pd.DataFrame(rows)
    hashes, thumbs = np.concatenate(hash_list), np.concatenate(thumb_list)
    print(f'[pass] {len(stats)} images in {time.time() - t0:.0f}s')
    report['raw_formats'] = stats.groupby(['split', 'raw_mode', 'raw_w', 'raw_h']).size().reset_index(name='n').to_dict('records')
    report['unreadable'] = stats.loc[~stats['readable'], 'id'].tolist()
    keep = stats['readable'].to_numpy()
    ok, hashes, thumbs = stats[keep].reset_index(drop=True), hashes[keep], thumbs[keep]
    is_tr, is_te = (ok['split'] == 'train').to_numpy(), (ok['split'] == 'test').to_numpy()

    # ---- 3. Categories ----
    ok['alien'] = ok['skin_blob'] > cfg['alien_skin_blob']
    ok.loc[ok['id'].isin(MANUAL['not_alien']), 'alien'] = False
    sky = ok[['border_r', 'border_g', 'border_b']].mean(axis=1)
    ok['grainy'] = (ok['chroma_noise'] > cfg['grain_chroma_noise']) | ((sky > cfg['grain_sky_mean']) & (ok['lum_noise'] > cfg['grain_lum_noise']))
    ok.loc[ok['id'].isin(MANUAL['not_grainy']), 'grainy'] = False
    ok.loc[ok['id'].isin(MANUAL['force_grainy']), 'grainy'] = True
    ok['wrong_label_dup'] = ok['id'].isin(list(MANUAL['wrong_label_duplicates']))
    ok['broken'] = is_tr & (ok['dark_flat'] > cfg['broken_dark_flat']).to_numpy() & (ok['gray_std'] >= cfg['blank_std']).to_numpy()
    ok['line'] = ok['line_score'] > cfg['line_flag']
    clean_tr = is_tr & ~ok['alien'].to_numpy() & ~ok['grainy'].to_numpy() & ~ok['wrong_label_dup'].to_numpy() & ~ok['broken'].to_numpy()
    lo, hi = np.quantile(ok.loc[clean_tr, 'core_g'], cfg['tint_quantiles'])
    ok['tint'] = np.where(ok['core_g'] < lo, 'purple', np.where(ok['core_g'] > hi, 'green', ''))
    ok['blank'] = ok['gray_std'] < cfg['blank_std']
    ok['saturated'] = ok['sat_hi_frac'] > cfg['sat_frac']
    ok['smooth'] = ok['lum_noise'] < np.quantile(ok.loc[clean_tr, 'lum_noise'], cfg['smooth_quantile'])
    clean_tr = clean_tr & ~ok['blank'].to_numpy()

    def category(r):
        # train: detectors verified by eye; test: grainy/alien hits are look-alikes, not the train corruption
        if r.blank:
            return 'blank'
        if r.split == 'train':
            if r.broken:
                return 'broken'
            return 'alien' if r.alien else 'wrong_label_dup' if r.wrong_label_dup else 'grainy' if r.grainy else 'clean'
        if r.tint:
            return 'tinted'
        if r.smooth:
            return 'smooth'
        if r.grainy:
            return 'grainy_like'
        if r.alien:
            return 'alien_candidate'
        return 'clean'
    ok['category'] = [category(r) for r in ok.itertuples()]
    ok['sky_mean'] = sky
    tr, te = ok[is_tr], ok[is_te]
    clean = ok[clean_tr]
    report['test_flags_overlapping'] = {f: int(te[f].astype(bool).sum()) for f in ['tint', 'smooth', 'grainy', 'alien', 'blank']}
    cat_split = pd.crosstab(ok['category'], ok['split'])
    report['category_by_split'] = cat_split.to_dict()
    report['tint_bounds_core_g'] = [round(float(lo), 4), round(float(hi), 4)]
    report['tint_test'] = te['tint'].value_counts().to_dict()
    report['tint_clean_train'] = clean['tint'].value_counts().to_dict()
    report['rule_flags'] = {f: {s: int(ok.loc[ok['split'] == s, f].sum()) for s in ['train', 'test']} for f in ['blank', 'broken', 'saturated', 'line']}

    # ---- 4. Overview: labels three ways, class overview, mean image/FFT, per-class stats ----
    counts = pd.DataFrame({
        'raw': tr['label'].value_counts(),
        'no_alien': tr[~tr['alien']]['label'].value_counts(),
        'clean': clean['label'].value_counts(),
    }).reindex(range(N_CLASSES)).fillna(0).astype(int)
    counts['clean_pct'] = (counts['clean'] / counts['clean'].sum() * 100).round(1)
    counts['raw_pct'] = (counts['raw'] / counts['raw'].sum() * 100).round(1)
    counts.index.name = 'label'
    report['label_counts'] = counts.reset_index().to_dict('records')
    fig, ax = plt.subplots(figsize=(10, 3.8))
    w = 0.27
    for k, (col, color) in enumerate([('raw', '#9aa5b1'), ('no_alien', '#5b8fd1'), ('clean', '#1f4e8c')]):
        ax.bar(np.arange(N_CLASSES) + (k - 1) * w, counts[col], w, label=col, color=color)
    ax.set_xticks(range(N_CLASSES)); ax.set_xticklabels([f'{c}\n{short(c)}' for c in range(N_CLASSES)], fontsize=8)
    ax.set_ylabel('number of train images'); ax.set_xlabel('class'); ax.legend(); ax.set_title('Train labels: raw / without aliens / clean (no aliens, no grainy)')
    savefig(fig, os.path.join(D['ov'], 'label_distribution.png'))

    ent = []
    for c in range(N_CLASSES):
        ids = clean.loc[clean['label'] == c, 'id'].to_numpy()
        pick = rng.choice(ids, min(cfg['overview_per_class'], len(ids)), replace=False) if len(ids) else []
        ent += [('train', i, (f'{c} {short(c)}' if j == 0 else i[-5:]), None) for j, i in enumerate(pick)]
    make_sheet(data_dir, ent, os.path.join(D['ov'], 'class_overview_clean.png'),
               'One row per class (0..9), 12 random CLEAN images each', ncols=cfg['overview_per_class'], thumb=112)

    keys = [k for k in [f'train_{c}' for c in range(N_CLASSES)] + ['test_-1'] if k in sums]
    fig, axes = plt.subplots(2, len(keys), figsize=(len(keys) * 1.9, 4.2), squeeze=False)
    for j, k in enumerate(keys):
        img_sum, fft_sum, n = sums[k]
        m = img_sum / n
        axes[0, j].imshow(np.clip(m / max(m.max(), 1) * 255, 0, 255).astype(np.uint8))
        axes[1, j].imshow(fft_sum / n, cmap='magma')
        axes[0, j].set_title(('test' if k == 'test_-1' else 'c' + k.split('_')[1]) + f' n={n}', fontsize=8)
        axes[0, j].axis('off'); axes[1, j].axis('off')
    fig.suptitle('Mean image (auto-scaled) and mean log|FFT| — all images incl. corrupted')
    savefig(fig, os.path.join(D['ov'], 'mean_image_fft.png'))

    shape_cols = ['gal_area', 'gal_axis_ratio', 'gal_offset', 'core_r', 'core_g', 'core_b', 'colorfulness', 'lap_var']
    fig, axes = plt.subplots(2, 4, figsize=(16, 6.5))
    for ax, col in zip(axes.ravel(), shape_cols):
        data = [te[col].dropna().to_numpy()] + [clean.loc[clean['label'] == c, col].dropna().to_numpy() for c in range(N_CLASSES)]
        ax.boxplot(data, showfliers=False)
        ax.set_xticks(range(1, N_CLASSES + 2)); ax.set_xticklabels(['test'] + [str(c) for c in range(N_CLASSES)], fontsize=7)
        ax.set_title(f'{col} [{UNITS[col]}]', fontsize=9); ax.set_xlabel('class (test = all test images)', fontsize=7)
    fig.suptitle('Galaxy shape / colour per class (clean train) vs test — how the classes differ')
    savefig(fig, os.path.join(D['ov'], 'stats_by_class_clean.png'), dpi=90)

    # ---- 5. Full class sheets (every train image) ----
    order = {'clean': 0, 'wrong_label_dup': 1, 'grainy': 2, 'alien': 3, 'broken': 4, 'blank': 5}
    pp = cfg['sheet_per_page']
    for c in range(N_CLASSES):
        sub = tr[tr['label'] == c].assign(o=lambda d: d['category'].map(order)).sort_values(['o', 'id'])
        n_cat = sub['category'].value_counts().to_dict()
        ent = [('train', r.id, f'{r.id[-5:]} {CAT_LETTER[r.category]}', CAT_COLORS[r.category]) for r in sub.itertuples()]
        pages = int(np.ceil(len(ent) / pp))
        for p in range(pages):
            make_sheet(data_dir, ent[p * pp:(p + 1) * pp],
                       os.path.join(D['sheets'], f'class{c}_{CLASS_NAMES[c]}_p{p + 1:02d}of{pages:02d}.png'),
                       f'Class {c} {CLASS_NAMES[c]} — page {p + 1}/{pages} — {n_cat} — frame: orange=grainy, red=alien, blue=wrong-label dup, black=blank, grey=broken', thumb=cfg['thumb_px'])
    print(f'[sheets] done at {time.time() - t0:.0f}s')

    # ---- 6. Aliens ----
    al = tr[tr['alien']].sort_values('skin_blob')
    make_sheet(data_dir, [('train', r.id, f'{r.id[-5:]} y={r.label}', None) for r in al.itertuples()],
               os.path.join(D['alien'], 'aliens_all_train.png'), f'All {len(al)} train aliens (skin_blob > {cfg["alien_skin_blob"]})', ncols=20, thumb=80)
    make_sheet(data_dir, [('train', r.id, f'{r.id[-5:]} {r.skin_blob:.3f}', None) for r in al.head(30).itertuples()],
               os.path.join(D['alien'], 'aliens_lowest_score.png'), 'Aliens with the LOWEST score (check: all aliens?)', thumb=110)
    na = tr[~tr['alien']].sort_values('skin_blob', ascending=False).head(30)
    make_sheet(data_dir, [('train', r.id, f'{r.id[-5:]} {r.skin_blob:.3f}', None) for r in na.itertuples()],
               os.path.join(D['alien'], 'non_aliens_highest_score.png'), 'Train NON-aliens with the HIGHEST score (check: any missed alien?)', thumb=110)
    tc = te.sort_values('skin_blob', ascending=False).head(30)
    make_sheet(data_dir, [('test', r.id, f'{r.id[-5:]} {r.skin_blob:.3f}', None) for r in tc.itertuples()],
               os.path.join(D['alien'], 'test_highest_score.png'), 'TEST images with the highest alien score (candidates; eyeball)', thumb=110)
    fig, ax = plt.subplots(figsize=(7, 3.2))
    bins = np.linspace(0, max(ok['skin_blob'].max(), 0.25), 60)
    for s, color in [('train', '#4C72B0'), ('test', '#DD8452')]:
        ax.hist(ok.loc[ok['split'] == s, 'skin_blob'], bins=bins, alpha=0.6, label=s, color=color, log=True)
    ax.axvline(cfg['alien_skin_blob'], color='k', ls='--'); ax.set_title('Alien score (largest pale-skin blob)'); ax.legend()
    ax.set_xlabel('skin_blob = fraction of the image covered by the largest pale-skin blob'); ax.set_ylabel('number of images (log scale)')
    savefig(fig, os.path.join(D['alien'], 'alien_score_hist.png'))

    # ---- 7. Grainy subset and its source ----
    fig, ax = plt.subplots(figsize=(7, 3.2))
    bins = np.linspace(0, ok['chroma_noise'].max(), 70)
    for name, d, color in [('clean+alien train', tr[~tr['grainy']], '#4C72B0'), ('grainy train', tr[tr['grainy']], '#e07b00'), ('test', te, '#999999')]:
        ax.hist(d['chroma_noise'], bins=bins, alpha=0.6, density=True, label=name, color=color)
    ax.axvline(cfg['grain_chroma_noise'], color='k', ls='--'); ax.set_title('Grain: chroma noise in the sky ring'); ax.legend(fontsize=8)
    ax.set_xlabel('chroma_noise = std of high-passed (R-G) in the outer ring [level 0-255]'); ax.set_ylabel('density (each histogram has area 1)')
    savefig(fig, os.path.join(D['grainy'], 'grain_hist.png'))
    views = thumb_views(thumbs)
    idx_gr = np.where(is_tr & ok['grainy'].to_numpy())[0]
    idx_cl = np.where(clean_tr)[0]
    idx_te = np.where(is_te)[0]
    gm = pd.DataFrame()
    if len(idx_gr) and len(idx_cl):
        # 1) thumbnail top-k candidates in clean train and in test; 2) star-field identity test
        K = cfg['match_topk']
        cos_c, cand_c = topk_match(views, idx_gr, idx_cl, K)
        has_te = len(idx_te) > 0
        cos_t, cand_t = topk_match(views, idx_gr, idx_te, K) if has_te else (None, None)
        base_q = idx_cl[rng.choice(len(idx_cl), min(300, len(idx_cl)), replace=False)]
        _, base_nn = best_match(views, base_q, idx_cl, exclude_self=True)

        def ref(i):
            return (ok.at[i, 'split'], ok.at[i, 'id'])
        tasks = [(data_dir, ref(q), [ref(j) for j in cand_c[n]] + ([ref(j) for j in cand_t[n]] if has_te else [])) for n, q in enumerate(idx_gr)]
        tasks += [(data_dir, ref(q), [ref(j)]) for q, j in zip(base_q, base_nn)]
        with Pool(cfg['workers']) as pool:
            res = list(tqdm(pool.imap(ident_task, tasks, chunksize=8), total=len(tasks), desc='identity'))
        ident_gr, base = res[:len(idx_gr)], np.array([r[0] for r in res[len(idx_gr):]])
        rows = []
        for n, q in enumerate(idx_gr):
            sc = np.array(ident_gr[n])
            bc = int(np.argmax(sc[:K])); src = cand_c[n, bc]
            row = {'id': ok.at[q, 'id'], 'label': int(ok.at[q, 'label']), 'src_clean_id': ok.at[src, 'id'],
                   'src_clean_label': int(ok.at[src, 'label']), 'ident_clean': round(float(sc[bc]), 4), 'thumb_cos_clean': round(float(cos_c[n, bc]), 4)}
            if has_te:
                bt = int(np.argmax(sc[K:])); row.update(best_test_id=ok.at[cand_t[n, bt], 'id'], ident_test=round(float(sc[K + bt]), 4))
            rows.append(row)
        gm = pd.DataFrame(rows).sort_values('ident_clean', ascending=False)
        gm['source_status'] = np.where(gm['ident_clean'] >= cfg['ident_confirm'], 'confirmed',
                                       np.where(gm['ident_clean'] < cfg['ident_none'], 'none_found', 'uncertain'))
        by_eye = gm['id'].map(MANUAL['confirmed_pairs_by_eye'])
        ok_eye = by_eye.notna() & (by_eye == gm['src_clean_id'])
        gm.loc[ok_eye, 'source_status'] = 'confirmed'
        gm['confirmed_by'] = np.where(ok_eye, 'eye', np.where(gm['source_status'] == 'confirmed', 'identity_score', ''))
        report['manual_pairs_not_matching'] = sorted(set(by_eye[by_eye.notna() & ~ok_eye].index.map(lambda i: gm.at[i, 'id'])))
        gm.drop(columns=['thumb_cos_clean']).to_csv(os.path.join(D['csv'], 'grainy_matches.csv'), index=False)
        cand = gm[gm['source_status'] == 'confirmed']
        fig, ax = plt.subplots(figsize=(7, 3.2))
        bins = np.linspace(-0.2, 1, 50)
        ax.hist(base, bins=bins, alpha=0.6, density=True, label='clean vs nearest OTHER clean (different galaxies)', color='#4C72B0')
        ax.hist(gm['ident_clean'], bins=bins, alpha=0.6, density=True, label='grainy vs best clean candidate', color='#e07b00')
        ax.axvline(cfg['ident_confirm'], color='k', ls='--'); ax.axvline(cfg['ident_none'], color='k', ls=':')
        ax.set_title('Star-field identity score (calibration)'); ax.legend(fontsize=7)
        ax.set_xlabel('identity = correlation of the star fields, galaxy masked, best of 8 rotations/flips [-1..1]'); ax.set_ylabel('density (area 1)')
        savefig(fig, os.path.join(D['grainy'], 'identity_calibration.png'))
        swap = pd.crosstab(cand['src_clean_label'], cand['label'], rownames=['clean source label'], colnames=['grainy label'])
        fig, ax = plt.subplots(figsize=(4.5, 3.8))
        ax.imshow(swap.values, cmap='Blues')
        for (i, j), v in np.ndenumerate(swap.values):
            ax.text(j, i, v, ha='center', va='center', fontsize=9)
        ax.set_xticks(range(swap.shape[1])); ax.set_xticklabels(swap.columns); ax.set_yticks(range(swap.shape[0])); ax.set_yticklabels(swap.index)
        ax.set_xlabel('grainy copy label'); ax.set_ylabel('clean source label'); ax.set_title(f'Grainy vs CONFIRMED source labels (n={len(cand)})', fontsize=9)
        savefig(fig, os.path.join(D['grainy'], 'label_swap_matrix.png'))

        def pairs(df, n, title, path):
            ent = []
            for r in df.head(n).itertuples():
                ent += [('train', r.id, f'G {r.id[-5:]} y={r.label}', CAT_COLORS['grainy']),
                        ('train', r.src_clean_id, f'S {r.src_clean_id[-5:]} y={r.src_clean_label} id={r.ident_clean:.2f}', None)]
            make_sheet(data_dir, ent, path, title, thumb=110)
        pairs(cand, 30, 'Grainy copy (orange) next to its confirmed clean source - strongest', os.path.join(D['grainy'], 'grainy_vs_source_strongest30.png'))
        pairs(cand.sort_values('ident_clean'), 30, 'Confirmed pairs with the WEAKEST identity score (check: same galaxy?)', os.path.join(D['grainy'], 'grainy_vs_source_weakest30.png'))
        unc = gm[gm['source_status'] == 'uncertain']
        pairs(unc, 30, 'UNCERTAIN pairs (between thresholds)', os.path.join(D['grainy'], 'grainy_vs_source_uncertain.png'))
        nf = gm[gm['source_status'] == 'none_found'].sort_values(['label', 'id'])
        make_sheet(data_dir, [('train', r.id, f'{r.id[-5:]} y={r.label}', None) for r in nf.itertuples()],
                   os.path.join(D['grainy'], 'grainy_no_source_found.png'), f'Grainy images with NO clean source found ({len(nf)})', ncols=15, thumb=90)
        if has_te:
            ent = []
            for r in gm.sort_values('ident_test', ascending=False).head(20).itertuples():
                ent += [('train', r.id, f'G {r.id[-5:]} y={r.label}', CAT_COLORS['grainy']), ('test', r.best_test_id, f'T {r.best_test_id[-5:]} id={r.ident_test:.2f}', None)]
            make_sheet(data_dir, ent, os.path.join(D['grainy'], 'grainy_vs_test_top20.png'), 'Grainy train next to best TEST candidate (identity score)', thumb=110)
        report['grainy_match'] = {
            'n_grainy': int(len(idx_gr)),
            'identity_calibration_different_galaxies_quantiles(.5,.9,.99,max)': np.round(np.quantile(base, [.5, .9, .99, 1]), 3).tolist(),
            'status_counts': gm['source_status'].value_counts().to_dict(),
            'confirmed_same_label_share': round(float((cand['label'] == cand['src_clean_label']).mean()), 4) if len(cand) else None,
            'confirmed_unique_sources': int(cand['src_clean_id'].nunique()),
            'copies_per_source': {int(k): int(v) for k, v in cand['src_clean_id'].value_counts().value_counts().sort_index().items()},
            'none_found_by_label': {int(k): int(v) for k, v in nf['label'].value_counts().sort_index().items()},
            f'with_test_match_ident>={cfg["ident_confirm"]}': int((gm['ident_test'] >= cfg['ident_confirm']).sum()) if has_te else None,
        }
        report['grainy_swap_matrix'] = swap.to_dict()
    gr_sheet = tr[tr['grainy']].sort_values(['label', 'id'])
    make_sheet(data_dir, [('train', r.id, f'{r.id[-5:]} y={r.label}', None) for r in gr_sheet.head(200).itertuples()],
               os.path.join(D['grainy'], 'grainy_train_first200.png'), 'Grainy train images (first 200 by label) — any alien hidden by the noise?', ncols=20, thumb=80)

    # ---- 7b. Cleaning proposal (Davide's, NOT the team's shared/cleaning_ids.csv) ----
    src = dict(zip(gm['id'], gm['src_clean_id'])) if len(gm) else {}
    status = dict(zip(gm['id'], gm['source_status'])) if len(gm) else {}
    rows = []
    for r in tr.itertuples():
        action, reason = 'keep', 'clean'
        if r.category == 'blank':
            action, reason = 'drop', 'blank image (flat grey noise)'
        elif r.category == 'broken':
            action, reason = 'drop', 'broken image (missing-data block + artifact)'
        elif r.category == 'alien':
            action, reason = 'drop', 'alien face pasted on the image'
        elif r.category == 'wrong_label_dup':
            action, reason = 'drop', f'duplicate of {MANUAL["wrong_label_duplicates"][r.id]} with a wrong label (checked by eye)'
        elif r.category == 'grainy' and status.get(r.id) == 'confirmed':
            action, reason = 'drop', f'grainy degraded copy of {src[r.id]} with swapped label'
        elif r.category == 'grainy':
            action, reason = 'drop', 'grainy, no confirmed clean source; label probably swapped (2<->3, 5<->6), not checked'
        rows.append({'id': r.id, 'label': r.label, 'category': r.category, 'action': action, 'reason': reason})
    cp = pd.DataFrame(rows)  # written after section 8, with the duplicate groups
    kept = cp[cp['action'] == 'keep']
    report['cleaning_proposal'] = {'actions': cp['action'].value_counts().to_dict(),
                                   'kept_by_label_after_proposal': {int(k): int(v) for k, v in kept['label'].value_counts().sort_index().items()}}

    # ---- 8. Duplicates (exact pixels + dHash over D4 confirmed by pixel MAD) ----
    exact = []
    for h, g in ok.groupby('pixel_md5'):
        if len(g) > 1:
            exact.append({'pixel_md5': h, 'n': len(g), 'ids': ';'.join(g['id']), 'splits': ';'.join(g['split']),
                          'labels': ';'.join(map(str, g['label'])), 'label_conflict': g.loc[g['split'] == 'train', 'label'].nunique() > 1})
    exact_df = pd.DataFrame(exact, columns=['pixel_md5', 'n', 'ids', 'splits', 'labels', 'label_conflict'])  # console summary only (0 groups)
    view0 = np.ascontiguousarray(hashes[:, 0, :])
    ci, cj, cd = [], [], []
    for s0 in tqdm(range(0, len(ok), 128), desc='dhash pairs'):
        d = hamming_min_d4(view0[s0:s0 + 128], hashes)
        ii, jj = np.nonzero(d <= cfg['dhash_max_dist'])
        gi = ii + s0; m = jj > gi
        ci.append(gi[m]); cj.append(jj[m]); cd.append(d[ii[m], jj[m]])
    ci, cj, cd = (np.concatenate(x) for x in (ci, cj, cd))
    cache = {}

    def sg(k):
        if k not in cache:
            cache[k] = small_gray(data_dir, ok.at[k, 'split'], ok.at[k, 'id'])
        return cache[k]
    near = []
    for i, j, dd in zip(ci, cj, cd):
        mad = min(float(np.abs(sg(i) - v).mean()) for v in _d4_views(sg(j)))
        if mad <= cfg['dup_confirm_mad']:
            near.append({'id_a': ok.at[i, 'id'], 'split_a': ok.at[i, 'split'], 'label_a': int(ok.at[i, 'label']), 'cat_a': ok.at[i, 'category'],
                         'id_b': ok.at[j, 'id'], 'split_b': ok.at[j, 'split'], 'label_b': int(ok.at[j, 'label']), 'cat_b': ok.at[j, 'category'],
                         'dhash_min_d4': int(dd), 'mad_64_min_d4': round(mad, 3)})
    near_df = pd.DataFrame(near, columns=['id_a', 'split_a', 'label_a', 'cat_a', 'id_b', 'split_b', 'label_b', 'cat_b', 'dhash_min_d4', 'mad_64_min_d4'])
    if len(near_df):
        near_df['pair_type'] = (near_df['split_a'] + '-' + near_df['split_b']).replace({'test-train': 'train-test'})
        near_df['label_conflict'] = (near_df['pair_type'] == 'train-train') & (near_df['label_a'] != near_df['label_b'])
        near_df = near_df.sort_values('mad_64_min_d4').reset_index(drop=True)
    near_df.to_csv(os.path.join(D['csv'], 'near_duplicates.csv'), index=False)

    # duplicate groups for StratifiedGroupKFold: union-find over train near duplicates + confirmed grainy->source pairs
    parent = {i: i for i in cp['id']}

    def root(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]; i = parent[i]
        return i
    links = []
    if len(near_df):
        tt = near_df[near_df['pair_type'] == 'train-train']
        links += list(zip(tt['id_a'], tt['id_b']))
    if len(gm):
        cf = gm[gm['source_status'] == 'confirmed']
        links += list(zip(cf['id'], cf['src_clean_id']))
    for a, b in links:
        ra, rb = root(a), root(b)
        if ra != rb:
            parent[max(ra, rb)] = min(ra, rb)   # representative = smallest id of the group
    cp['dup_group'] = [root(i) for i in cp['id']]
    cp.to_csv(os.path.join(D['csv'], 'cleaning_proposal_davide.csv'), index=False)
    kg = cp[cp['action'] == 'keep']
    report['dup_groups'] = {'train_groups_gt1': int((cp['dup_group'].value_counts() > 1).sum()),
                            'kept_groups_gt1': int((kg['dup_group'].value_counts() > 1).sum())}

    def pair_sheet(df, path, title):
        ent = []
        for r in df.itertuples():
            ent += [(r.split_a, r.id_a, f'{r.id_a[-5:]} y={r.label_a}', CAT_COLORS.get(r.cat_a)), (r.split_b, r.id_b, f'{r.id_b[-5:]} y={r.label_b}', CAT_COLORS.get(r.cat_b))]
        make_sheet(data_dir, ent, path, title, thumb=110)
    if len(near_df):
        pair_sheet(near_df[near_df['pair_type'] != 'train-train'], os.path.join(D['dups'], 'near_dups_involving_test.png'), 'Near duplicates involving TEST (side by side)')
        pair_sheet(near_df[near_df['label_conflict']], os.path.join(D['dups'], 'near_dups_label_conflict.png'), 'Train near duplicates with DIFFERENT labels')
        pair_sheet(near_df[near_df['pair_type'] == 'train-train'].head(40), os.path.join(D['dups'], 'near_dups_train_top40.png'), 'Closest train-train near duplicates (frame: orange=grainy)')
    report['duplicates'] = {
        'exact_groups': len(exact_df), 'exact_images': int(exact_df['n'].sum()) if len(exact_df) else 0,
        'dhash_candidates': int(len(ci)), 'near_confirmed_pairs': len(near_df),
        'near_by_type': near_df['pair_type'].value_counts().to_dict() if len(near_df) else {},
        'near_by_categories': (near_df['cat_a'] + '-' + near_df['cat_b']).value_counts().to_dict() if len(near_df) else {},
        'near_train_label_conflicts': int(near_df['label_conflict'].sum()) if len(near_df) else 0,
    }

    # ---- 9. Test: random, tints, grainy-like, colour scatter ----
    te_pick = rng.choice(te['id'].to_numpy(), min(64, len(te)), replace=False)
    make_sheet(data_dir, [('test', i, i[-5:], None) for i in te_pick], os.path.join(D['test'], 'test_random64.png'), 'Test — 64 random images', ncols=8, thumb=128)
    for t in ['purple', 'green']:
        sub = te[te['tint'] == t]
        sub = sub.sort_values('core_g', ascending=(t == 'purple'))
        make_sheet(data_dir, [('test', r.id, f'{r.id[-5:]} g={r.core_g:.3f}', None) for r in sub.head(60).itertuples()],
                   os.path.join(D['test'], f'test_tinted_{t}_most60.png'), f'Test tinted {t}: most extreme 60 of {len(sub)}', thumb=110)
        mid = sub.iloc[len(sub) // 2: len(sub) // 2 + 30] if len(sub) > 60 else sub.iloc[:0]
        make_sheet(data_dir, [('test', r.id, f'{r.id[-5:]} g={r.core_g:.3f}', None) for r in mid.itertuples()],
                   os.path.join(D['test'], f'test_tinted_{t}_median30.png'), f'Test tinted {t}: 30 typical (median) ones', thumb=110)
    ctr = clean[clean['tint'] != ''].sort_values('core_g')
    make_sheet(data_dir, [('train', r.id, f'{r.id[-5:]} y={r.label} g={r.core_g:.3f}', None) for r in ctr.head(40).itertuples()],
               os.path.join(D['test'], 'clean_train_outside_tint_bounds.png'), 'Reference: CLEAN TRAIN images outside the same bounds (2% by construction)', thumb=110)
    gl = te[te['grainy']].sort_values('chroma_noise', ascending=False)
    make_sheet(data_dir, [('test', r.id, f'{r.id[-5:]} n={r.chroma_noise:.0f}', None) for r in gl.head(60).itertuples()],
               os.path.join(D['test'], 'test_grainy_like_top60.png'), f'Test above the grain threshold: {len(gl)} (eyeball: same corruption as train?)', thumb=110)
    sm = te[te['smooth']].sort_values('lum_noise')
    make_sheet(data_dir, [('test', r.id, f'{r.id[-5:]} n={r.lum_noise:.1f}', None) for r in sm.head(60).itertuples()],
               os.path.join(D['test'], 'test_smooth_most60.png'), f'Test smoother than 99% of clean train: {len(sm)} (blurred / denoised?)', thumb=110)
    ref_s = clean.sort_values('lum_noise').head(30)
    make_sheet(data_dir, [('train', r.id, f'{r.id[-5:]} y={r.label} n={r.lum_noise:.1f}', None) for r in ref_s.itertuples()],
               os.path.join(D['test'], 'clean_train_smoothest30.png'), 'Reference: smoothest CLEAN train images', thumb=110)
    ls = os.path.join(out, '03_corruptions', 'lines'); os.makedirs(ls, exist_ok=True)
    ent = []
    for s_ in ['train', 'test']:
        top = ok[ok['split'] == s_].sort_values('line_score', ascending=False).head(15)
        ent += [(s_, r.id, f'{r.id[-5:]} {s_[:2]} {r.line_score:.0f}', CAT_COLORS.get(r.category)) for r in top.itertuples()]
    make_sheet(data_dir, ent, os.path.join(ls, 'lines_top15_train_test.png'),
               f'Straight-line artifacts: top 15 train, then top 15 test (caption = line_score; flag > {cfg["line_flag"]:.0f})', ncols=10, thumb=128)
    bl = ok[ok['blank']]
    make_sheet(data_dir, [(r.split, r.id, f'{r.id} y={r.label}', None) for r in bl.itertuples()],
               os.path.join(D['test'], 'blank_images_train_and_test.png'), f'Blank images (flat grey noise): {len(bl)}', ncols=5, thumb=128)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8))
    for ax, (xc, yc) in zip(axes, [('core_g', 'core_b'), ('core_r', 'core_g')]):
        ax.scatter(te[xc], te[yc], s=2, alpha=0.25, color='#DD8452', label='test')
        ax.scatter(clean[xc], clean[yc], s=2, alpha=0.4, color='#1f4e8c', label='clean train')
        ax.set_xlabel(f'{xc} = share of {xc[-1].upper()} in the core colour [0-1]'); ax.set_ylabel(f'{yc} = share of {yc[-1].upper()} [0-1]'); ax.legend(markerscale=5)
    axes[0].axvline(lo, color='k', ls=':'); axes[0].axvline(hi, color='k', ls=':')
    fig.suptitle('Galaxy core chromaticity: clean train vs test (dotted = tint bounds)')
    savefig(fig, os.path.join(D['test'], 'core_colour_scatter.png'))

    # ---- 10. Shift: stats train vs test, adversarial validation ----
    plot_cols = [c for c in STAT_COLS if c not in ('gal_offset',)]
    ncol = 4; nrow = int(np.ceil(len(plot_cols) / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(ncol * 3.6, nrow * 2.5))
    for ax, col in zip(axes.ravel(), plot_cols):
        v = ok[col].dropna()
        a, b = np.percentile(v, [0.1, 99.9]); bins = np.linspace(a, b if b > a else a + 1, 60)
        for name, d, color in [('clean train', clean, '#1f4e8c'), ('grainy train', tr[tr['grainy']], '#e07b00'), ('test', te, '#999999')]:
            if len(d):
                ax.hist(np.clip(d[col].dropna(), a, b), bins=bins, density=True, alpha=0.5, color=color, label=name)
        ax.set_title(f'{col} [{UNITS[col]}]', fontsize=9); ax.tick_params(labelsize=7)
        ax.set_ylabel('density', fontsize=7)
    for ax in axes.ravel()[len(plot_cols):]:
        ax.axis('off')
    axes.ravel()[0].legend(fontsize=7)
    savefig(fig, os.path.join(D['shift'], 'stats_clean_grainy_test.png'), dpi=85)

    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.metrics import roc_auc_score
    from sklearn.model_selection import StratifiedKFold
    feat_cols = [c for c in STAT_COLS if c not in ('skin_blob',)]
    adv = {}
    clean_te = is_te & (ok['category'] == 'clean').to_numpy()
    untinted_te = is_te & (ok['tint'] == '').to_numpy() & ~ok['blank'].to_numpy()
    for name, mask in [('all_train_vs_test', is_tr | is_te), ('clean_train_vs_test', clean_tr | is_te),
                       ('clean_train_vs_untinted_test', clean_tr | untinted_te), ('clean_train_vs_clean_test', clean_tr | clean_te)]:
        d = ok[mask]
        X = d[feat_cols].to_numpy(np.float32); y = (d['split'] == 'test').astype(int).to_numpy()
        if y.min() == y.max() or min(np.bincount(y)) < cfg['adv_folds']:
            continue
        oof = np.zeros(len(y))
        for tri, vai in StratifiedKFold(cfg['adv_folds'], shuffle=True, random_state=cfg['seed']).split(X, y):
            clf = HistGradientBoostingClassifier(max_iter=300, learning_rate=0.05, random_state=cfg['seed']).fit(X[tri], y[tri])
            oof[vai] = clf.predict_proba(X[vai])[:, 1]
        uni = pd.Series({c: roc_auc_score(y, d[c].fillna(d[c].median())) for c in feat_cols})
        uni = uni.where(uni >= 0.5, 1 - uni).sort_values(ascending=False)
        adv[name] = {'auc': round(float(roc_auc_score(y, oof)), 4), 'top_univariate': uni.head(6).round(3).to_dict()}
        if name == 'all_train_vs_test':
            ok['p_test'] = oof  # mask covers every image, same row order
        if name == 'clean_train_vs_test':
            u = d[['id', 'split', 'label']].assign(p_test=oof)
            u = u[u['split'] == 'train'].sort_values('p_test').head(40)
            make_sheet(data_dir, [('train', r.id, f'{r.id[-5:]} y={r.label} p={r.p_test:.2f}', None) for r in u.itertuples()],
                       os.path.join(D['shift'], 'clean_train_least_testlike.png'), 'Clean train images least like test (adversarial)', thumb=110)
    report['adversarial'] = adv
    # one row per image: category, flags, detector stats, adversarial p_test (all train vs test)
    flag_cols = ['alien', 'grainy', 'wrong_label_dup', 'broken', 'tint', 'smooth', 'blank', 'saturated', 'line']
    per_cols = ['id', 'split', 'label', 'category'] + flag_cols + ['sky_mean'] + STAT_COLS + EXTRA_COLS + (['p_test'] if 'p_test' in ok else [])
    ok[per_cols].round(4).to_csv(os.path.join(D['csv'], 'per_image.csv'), index=False)

    # ---- 11. Single PNG folders (APFS clones) ----
    if cfg['copy_images']:
        I = D['img']
        for r in tr.itertuples():
            place(data_dir, 'train', r.id, os.path.join(I, 'train_by_class', f'{r.label}_{CLASS_NAMES[r.label]}', r.category))
        tdir = os.path.join(I, 'test')
        for i in rng.choice(te['id'].to_numpy(), min(cfg['test_random_folder'], len(te)), replace=False):
            place(data_dir, 'test', i, os.path.join(tdir, 'random'))
        for t in ['purple', 'green']:
            for i in te.loc[te['tint'] == t].sort_values('core_g', ascending=(t == 'purple'))['id'].head(300):
                place(data_dir, 'test', i, os.path.join(tdir, f'tinted_{t}'))
        for i in te.loc[te['grainy'], 'id']:
            place(data_dir, 'test', i, os.path.join(tdir, 'grainy_like'))
        for i in te.sort_values('skin_blob', ascending=False)['id'].head(30):
            place(data_dir, 'test', i, os.path.join(tdir, 'alien_candidates'))
        for i in te[te['smooth']].sort_values('lum_noise')['id'].head(300):
            place(data_dir, 'test', i, os.path.join(tdir, 'smooth'))
        for r in ok[ok['blank']].itertuples():
            place(data_dir, r.split, r.id, os.path.join(tdir, 'blank'), f'{r.id}.png')
        R = os.path.join(I, 'review')
        if len(gm):
            for k, r in enumerate(gm[gm['source_status'] == 'confirmed'].itertuples(), 1):
                fd = os.path.join(R, 'grainy_vs_source')
                place(data_dir, 'train', r.id, fd, f'{k:03d}_a_grainy_{r.id}_y{r.label}.png')
                place(data_dir, 'train', r.src_clean_id, fd, f'{k:03d}_b_source_{r.src_clean_id}_y{r.src_clean_label}.png')
        for sub, df in [('near_duplicates', near_df), ('label_conflicts', near_df[near_df['label_conflict']] if len(near_df) else near_df)]:
            for k, r in enumerate(df.itertuples(), 1):
                fd = os.path.join(R, sub)
                place(data_dir, r.split_a, r.id_a, fd, f'{k:03d}_a_{r.id_a}_y{r.label_a}.png')
                place(data_dir, r.split_b, r.id_b, fd, f'{k:03d}_b_{r.id_b}_y{r.label_b}.png')
        print(f'[images] done at {time.time() - t0:.0f}s')

    # ---- 12. Summary ----
    elapsed = time.time() - t0
    print_summary(report, cat_split)
    print(f'[done] {elapsed:.0f}s -> {out}')


def print_summary(rep, cat_split):
    # short console recap; the full story (tables, charts, decisions) is in eda/00_eda_davide.ipynb
    G, D, A = rep.get('grainy_match', {}), rep['duplicates'], rep['adversarial']
    print('\n' + cat_split.to_string())
    print('cleaning proposal:', rep['cleaning_proposal']['actions'], '| kept per label:', rep['cleaning_proposal']['kept_by_label_after_proposal'])
    print('grainy sources:', G.get('status_counts'), '| unique sources:', G.get('confirmed_unique_sources'),
          '| manual pairs not matching:', rep.get('manual_pairs_not_matching', []))
    print('near duplicates:', D['near_confirmed_pairs'], D['near_by_type'], '| train label conflicts:', D['near_train_label_conflicts'])
    print('adversarial AUC:', {k: v['auc'] for k, v in A.items()})
    print('rule flags:', rep['rule_flags'], '| duplicate groups (train / kept):', rep.get('dup_groups'))


if __name__ == '__main__':
    main()
