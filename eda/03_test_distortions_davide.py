"""Measure the two test distortions (colour cast, blur) so that training augmentation reproduces them (Davide).

Run AFTER eda/01_audit_davide.py (reads csv/per_image.csv):
    python eda/03_test_distortions_davide.py        # ~1 min

Findings it reproduces
  - Colour cast = HSV HUE ROTATION: brightness (V) and saturation (S) are unchanged, grey sky stays grey, the hue of the
    galaxy is shifted by a per-image angle. Measured on the galaxy core hue vs clean train. The core_g detector of
    01_audit only sees purple/green casts; a ~180 deg rotation turns yellow cores blue and stays inside its bounds.
  - Blur = Gaussian-like: calibrated by blurring clean train images with known sigma and matching the luminance noise
    of the sky ring (lum_noise), cross-checked on the average power spectrum.
Writes eda/outputs/davide/07_test_distortions/ (figures, test_distortions.csv, augmentation_params.json).
"""
import colorsys
import importlib
import json
import os
import sys
from multiprocessing import Pool

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image, ImageDraw
from scipy import ndimage

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
audit = importlib.import_module('01_audit_davide')
DATA = audit.CONFIG['data_dir']
N = 256

CONFIG = {
    'out_dir': os.path.join(audit.CONFIG['out_dir'], '07_test_distortions'),
    'seed': 42,
    'n_calib': 600,                       # clean train images blurred for the calibration / spectra
    'sigmas': [0, 0.5, 0.75, 1.0, 1.25, 1.5, 2.0, 2.5, 3.0],
    'blur_min_sigma': 0.55,               # "blurred" if estimated sigma >= this (clean train: 2.9% above)
    'hue_bin_deg': 10,
}

_yy, _xx = np.meshgrid(np.fft.fftfreq(N), np.fft.fftfreq(N), indexing='ij')
_RR = np.minimum((np.sqrt(_xx ** 2 + _yy ** 2) * N).astype(int), N // 2)
_RING = np.ones((N, N), bool); _RING[24:-24, 24:-24] = False   # same 24 px ring as lum_noise in 01_audit


def _load(split, img_id):
    return np.asarray(Image.open(os.path.join(DATA, split, f'{img_id}.png')).convert('RGB'), dtype=np.float32)


def _lum_noise(f):
    hp = f - ndimage.uniform_filter(f, size=(5, 5, 1))
    return float(np.std(hp.mean(-1)[_RING]))


def _radial(gray):
    p = np.abs(np.fft.fft2(gray - gray.mean())) ** 2
    return np.bincount(_RR.ravel(), p.ravel(), minlength=N // 2 + 1)[:N // 2 + 1] / np.bincount(_RR.ravel(), minlength=N // 2 + 1)[:N // 2 + 1]


def calib_task(args):  # top-level for macOS spawn
    split, img_id, sigmas = args
    f = _load(split, img_id)
    out = {'spec': np.log(_radial(f.mean(2)) + 1e-9)}
    if sigmas:
        out['ln'] = [_lum_noise(ndimage.gaussian_filter(f, (s, s, 0)) if s > 0 else f) for s in sigmas]
    return out


def hue_rotate(rgb_u8, deg):
    """HSV hue rotation (what torchvision adjust_hue does), deg in degrees."""
    h = np.asarray(Image.fromarray(rgb_u8).convert('HSV')).copy()
    h[..., 0] = ((h[..., 0].astype(int) + int(round(deg / 360 * 256))) % 256).astype(np.uint8)
    return np.asarray(Image.fromarray(h, 'HSV').convert('RGB'))


def tile_grid(rows, captions, path, title, px=128):
    """rows = list of lists of uint8 RGB arrays; captions same shape."""
    nr, nc = len(rows), max(len(r) for r in rows)
    sheet = Image.new('RGB', (nc * px, nr * (px + 14) + 22), 'white')
    d = ImageDraw.Draw(sheet)
    d.text((4, 4), title, fill='black', font=audit._font())
    for i, (row, caps) in enumerate(zip(rows, captions)):
        for j, (a, c) in enumerate(zip(row, caps)):
            x, y = j * px, 22 + i * (px + 14)
            sheet.paste(Image.fromarray(a).resize((px, px)), (x, y + 14))
            d.text((x + 2, y + 1), c, fill='black', font=audit._font())
    sheet.save(path)


def main():
    cfg = CONFIG
    out = cfg['out_dir']
    if os.path.isdir(out):
        for f in os.listdir(out):
            os.remove(os.path.join(out, f))
    os.makedirs(out, exist_ok=True)
    per = pd.read_csv(os.path.join(audit.CONFIG['out_dir'], 'csv', 'per_image.csv'))
    per['tint'] = per['tint'].fillna('')
    clean = (per['split'] == 'train') & (per['category'] == 'clean')
    test = (per['split'] == 'test') & (per['category'] != 'blank')

    # ---- 1. hue: core hue shift w.r.t. clean train ----
    hsv = np.array([colorsys.rgb_to_hsv(*c) for c in per[['core_r', 'core_g', 'core_b']].to_numpy()])
    hue, sat = hsv[:, 0] * 360, hsv[:, 1]
    c = clean.to_numpy()
    h0 = float(np.degrees(np.angle(np.sum(sat[c] * np.exp(1j * np.radians(hue[c]))))) % 360)
    per['core_hue_shift'] = ((hue - h0 + 180) % 360) - 180
    edges = np.arange(-180, 181, cfg['hue_bin_deg'])
    hc = np.histogram(per.loc[clean, 'core_hue_shift'], edges)[0] / clean.sum()
    ht = np.histogram(per.loc[test, 'core_hue_shift'], edges)[0] / test.sum()
    # smallest share p of rotated images such that test - (1-p) * clean has no negative bin (tolerance 0.3%)
    p_rot = float(min(p for p in np.arange(0, 1, 0.005) if (ht - (1 - p) * hc).min() >= -0.003))
    resid = ht - (1 - p_rot) * hc
    bands = {'purple/magenta [-150,-60)': (-150, -60), 'green/cyan [15,150)': (15, 150), '~180 (yellow->blue) [150,180)+[-180,-150)': None}
    band_share = {}
    for k, b in bands.items():
        m = ((edges[:-1] >= 150) | (edges[:-1] < -150)) if b is None else ((edges[:-1] >= b[0]) & (edges[:-1] < b[1]))
        band_share[k] = round(float(resid[m].sum()) * 100, 1)

    fig, ax = plt.subplots(figsize=(10, 3.8))
    ctr = edges[:-1] + cfg['hue_bin_deg'] / 2
    ax.bar(ctr, hc * 100, width=cfg['hue_bin_deg'] * 0.9, color='#1f4e8c', alpha=0.6, label='clean train')
    ax.bar(ctr, ht * 100, width=cfg['hue_bin_deg'] * 0.45, color='#DD8452', alpha=0.9, label='test')
    ax.step(ctr, resid * 100, where='mid', color='k', lw=1, label=f'test minus {1 - p_rot:.2f} x clean = rotated part (>= {p_rot:.0%})')
    ax.set_xlabel(f'hue of the galaxy core minus the clean-train mean ({h0:.0f} deg) [degrees]')
    ax.set_ylabel(f'% of images per {cfg["hue_bin_deg"]}-deg bin'); ax.legend(fontsize=8)
    ax.set_title('Core hue shift: train cores sit near 0; the test has rotated cores almost all around the circle', fontsize=10)
    audit.savefig(fig, os.path.join(out, 'hue_shift_hist.png'))

    # V/S invariance (HSV hue rotation keeps brightness and saturation)
    inv = []
    for name, m in [('clean train', clean), ('test untinted', test & (per['tint'] == '')), ('test purple', test & (per['tint'] == 'purple')), ('test green', test & (per['tint'] == 'green'))]:
        for r in per[m].sample(150, random_state=cfg['seed']).itertuples():
            h = np.asarray(Image.open(os.path.join(DATA, r.split, f'{r.id}.png')).convert('RGB').convert('HSV'), dtype=np.float32) / 255
            b = h[..., 2] > 0.25
            inv.append({'group': name, 'V median (bright px)': float(np.median(h[..., 2][b])) if b.any() else np.nan,
                        'S 90th pct (bright px)': float(np.quantile(h[..., 1][b], .9)) if b.any() else np.nan,
                        'sky R-G (levels)': float(_load(r.split, r.id)[:16, :, 0].mean() - _load(r.split, r.id)[:16, :, 1].mean())})
    inv = pd.DataFrame(inv).groupby('group').median().round(3)

    # undo examples: typical purple and green test images, rotated back by -shift
    rows, caps = [], []
    for t in ['purple', 'green']:
        d = per[test & (per['tint'] == t)].copy()
        d = d.iloc[(d['core_hue_shift'] - d['core_hue_shift'].median()).abs().argsort()[:8]]
        imgs = [_load('test', i).astype(np.uint8) for i in d['id']]
        rows += [imgs, [hue_rotate(a, -s) for a, s in zip(imgs, d['core_hue_shift'])]]
        caps += [[f'{i[-5:]} {t}' for i in d['id']], [f'undo {-s:+.0f} deg' for s in d['core_hue_shift']]]
    tile_grid(rows, caps, os.path.join(out, 'tint_undo_examples.png'),
              'Tinted test images (rows 1, 3) and the same rotated back in HSV hue by the measured angle (rows 2, 4)')

    # ---- 2. blur: calibration on clean train + spectra ----
    cal = per[clean].sample(cfg['n_calib'], random_state=cfg['seed'])
    smo = per[per['category'] == 'smooth'].sample(cfg['n_calib'], random_state=cfg['seed'])
    cte = per[test & (per['category'] == 'clean')].sample(cfg['n_calib'], random_state=cfg['seed'])
    with Pool(audit.CONFIG['workers']) as pool:
        rc = pool.map(calib_task, [('train', i, cfg['sigmas']) for i in cal['id']])
        rs = pool.map(calib_task, [('test', i, None) for i in smo['id']])
        rt = pool.map(calib_task, [('test', i, None) for i in cte['id']])
    L = np.array([r['ln'] for r in rc])
    med = np.median(L, 0)
    sig = cfg['sigmas']
    per['blur_sigma_est'] = np.interp(-per['lum_noise'], -med, sig)
    f = np.arange(N // 2 + 1) / N
    spec = {k: np.mean([r['spec'] for r in R], 0) for k, R in [('train', rc), ('smooth', rs), ('clean_test', rt)]}
    fits = {}
    m = (f > 0.03) & (f < 0.25)
    for k in ['smooth', 'clean_test']:
        dlt = spec[k] - spec['train']
        A = np.vstack([np.ones(m.sum()), -4 * np.pi ** 2 * f[m] ** 2]).T
        c0, s2 = np.linalg.lstsq(A, dlt[m], rcond=None)[0]
        fits[k] = (float(c0), float(np.sqrt(max(s2, 0))))

    fig, axes = plt.subplots(1, 3, figsize=(16, 4.2))
    ax = axes[0]
    ax.plot(sig, med, 'o-', color='#1f4e8c', label='clean train blurred (median)')
    ax.fill_between(sig, np.quantile(L, .25, 0), np.quantile(L, .75, 0), color='#1f4e8c', alpha=0.2, label='25-75%')
    for name, d, col in [('test smooth', per[per['category'] == 'smooth'], '#7fb3d5'), ('test clean', per[test & (per['category'] == 'clean')], '#999999')]:
        ax.axhline(d['lum_noise'].median(), color=col, ls='--', label=f'{name} median')
    ax.set_xlabel('Gaussian blur sigma applied [pixels at 256 px]'); ax.set_ylabel('lum_noise [std, level 0-255]')
    ax.set_title('Calibration: noise left after blurring', fontsize=10); ax.legend(fontsize=7)
    ax = axes[1]
    for k, col in [('smooth', '#7fb3d5'), ('clean_test', '#999999')]:
        ax.plot(f[1:], (spec[k] - spec['train'])[1:], color=col, label=f'{k.replace("_", " ")} / clean train')
        c0, s = fits[k]
        ax.plot(f[m], c0 - 4 * np.pi ** 2 * s ** 2 * f[m] ** 2, color=col, ls=':', label=f'Gaussian fit sigma={s:.2f}')
    ax.set_xlabel('spatial frequency [cycles/pixel]'); ax.set_ylabel('log power ratio (0 = same as train)')
    ax.set_title('Average spectrum relative to clean train', fontsize=10); ax.legend(fontsize=7)
    ax = axes[2]
    be = [0, 0.1, 0.25, 0.4, 0.55, 0.7, 0.85, 1.0, 1.2, 1.5, 3.0]
    hb_c = np.histogram(per.loc[clean, 'blur_sigma_est'], be)[0] / clean.sum() * 100
    hb_t = np.histogram(per.loc[test, 'blur_sigma_est'], be)[0] / test.sum() * 100
    xx = np.arange(len(be) - 1)
    ax.bar(xx - 0.2, hb_c, 0.4, color='#1f4e8c', label='clean train'); ax.bar(xx + 0.2, hb_t, 0.4, color='#DD8452', label='test')
    ax.set_xticks(xx); ax.set_xticklabels([f'{a}-{b}' for a, b in zip(be[:-1], be[1:])], fontsize=7, rotation=45)
    ax.set_xlabel('estimated blur sigma [pixels]'); ax.set_ylabel('% of images'); ax.legend(fontsize=8)
    ax.set_title('Estimated blur per image', fontsize=10)
    audit.savefig(fig, os.path.join(out, 'blur_calibration.png'))
    blurred = per.loc[test, 'blur_sigma_est'] >= cfg['blur_min_sigma']
    share_blur = float(blurred.mean()); share_blur_train = float((per.loc[clean, 'blur_sigma_est'] >= cfg['blur_min_sigma']).mean())
    excess = (hb_t - hb_c)[3:].clip(0).sum() / 100  # bins from 0.4 px up

    # ---- 3. simulated vs real ----
    ex = per[clean & (per['label'].isin([2, 5, 7, 9]))].sample(4, random_state=3)['id']
    rows, caps = [], []
    for i in ex:
        a = _load('train', i).astype(np.uint8)
        bl = np.clip(ndimage.gaussian_filter(a.astype(np.float32), (0.9, 0.9, 0)), 0, 255).astype(np.uint8)
        rows.append([a, hue_rotate(a, -100), hue_rotate(a, 70), hue_rotate(a, 180), bl, hue_rotate(bl, -100)])
        caps.append([f'{i[-5:]} original', 'hue -100', 'hue +70', 'hue 180', 'blur s=0.9', 'blur + hue -100'])
    real = per[test & (per['tint'] == 'purple')].sample(2, random_state=1)['id'].tolist() + per[test & (per['tint'] == 'green')].sample(2, random_state=1)['id'].tolist() \
        + per[per['category'] == 'smooth'].sample(2, random_state=1)['id'].tolist()
    rows.append([_load('test', i).astype(np.uint8) for i in real])
    caps.append([f'real test {i[-5:]}' for i in real])
    tile_grid(rows, caps, os.path.join(out, 'simulated_vs_real.png'),
              'Clean train images with simulated distortions (rows 1-4) next to real test images (last row: 2 purple, 2 green, 2 blurred)')

    # ---- outputs ----
    per.loc[per['split'] == 'test', ['id', 'category', 'tint', 'core_hue_shift', 'blur_sigma_est']].round(3).to_csv(
        os.path.join(out, 'test_distortions.csv'), index=False)
    params = {
        'hue_rotation': {
            'what': 'HSV hue rotation (torchvision.transforms.functional.adjust_hue / ColorJitter(hue=...)); brightness and saturation unchanged',
            'measured_min_share_of_test': round(p_rot, 3),
            'measured_bands_pct_of_test': band_share,
            'recommended': {'p': 0.4, 'angle_deg': [-180, 180], 'torchvision_hue_factor': [-0.5, 0.5]},
        },
        'gaussian_blur': {
            'measured_share_of_test_sigma_ge_%.2f' % cfg['blur_min_sigma']: round(share_blur, 3),
            'same_on_clean_train': round(share_blur_train, 3),
            'measured_excess_share_sigma_ge_0.4': round(float(excess), 3),
            'spectrum_fit_sigma_px': {'smooth_test': round(fits['smooth'][1], 2), 'clean_test': round(fits['clean_test'][1], 2)},
            'recommended': {'p': 0.35, 'sigma_px_at_256': [0.4, 1.2], 'note': 'scale sigma by new_size/256 if images are resized first'},
        },
        'independent': 'blur appears in both tinted (~16%) and untinted (~25%) test images: apply the two augmentations independently',
    }
    with open(os.path.join(out, 'augmentation_params.json'), 'w') as fh:
        json.dump(params, fh, indent=2)
    print('[hue] reference core hue', round(h0, 1), '| rotated share >=', round(p_rot, 3), band_share)
    print('[hue] V/S invariance check:\n' + inv.to_string())
    print('[blur] spectrum fit sigma:', {k: round(v[1], 2) for k, v in fits.items()}, f'| test sigma>={cfg["blur_min_sigma"]}: {share_blur:.1%} (clean train {share_blur_train:.1%}) | excess >=0.4: {excess:.1%}')
    print(f'[done] -> {out}')


if __name__ == '__main__':
    main()
