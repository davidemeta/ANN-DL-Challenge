"""CNN-embedding block of the Day 1-2 checklist (Davide). Needs torch + timm + umap-learn.

Run AFTER eda/01_audit_davide.py (it reads csv/per_image.csv and the grainy matches):
    python eda/02_embeddings_davide.py              # ~10 min the first time, ~1 min once embeddings are cached
    python eda/02_embeddings_davide.py --recompute  # re-extract the embeddings

What it does
  1. Frozen pretrained CNN (CONFIG['backbone'], global-pooled features), each image under the 8 D4 views
     (4 rotations x flip) averaged -> one rotation-invariant vector per image, L2-normalised.
     Cached in eda/outputs/embeddings_cache/ (the same files family D needs).
  2. UMAP map of all 19k images (by class, by category).
  3. k-NN among clean train images: how often the neighbours share the label (class overlap, possible label errors).
  4. Linear probe (logistic regression, 5-fold OOF on clean train): a free first baseline. Then applied to the
     grainy images (does it predict the given or the swapped label?), to the aliens, and to the test (rough class prior).
  5. Outlier score = mean cosine distance to the k nearest clean train images.
Writes eda/outputs/davide/06_embeddings/ (figures + embedding_analysis.csv).
"""
import argparse
import importlib
import os
import sys
import time

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from PIL import Image

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
audit = importlib.import_module('01_audit_davide')  # reuse paths, sheets, colours
ROOT, CLASS_NAMES, N_CLASSES = audit.ROOT, audit.CLASS_NAMES, audit.N_CLASSES

CONFIG = {
    'backbone': 'convnextv2_tiny.fcmae_ft_in22k_in1k',  # playbook family D backbone (768-d)
    'img_size': 224,
    'batch': 64,
    'workers': 4,
    'cache_dir': os.path.join(ROOT, 'eda', 'outputs', 'embeddings_cache'),
    'out_dir': os.path.join(audit.CONFIG['out_dir'], '06_embeddings'),
    'seed': 42,
    'pca_dim': 50,             # PCA before UMAP / adversarial validation
    'umap_neighbors': 30,
    'umap_min_dist': 0.15,
    'knn_k': 10,
    'suspect_agree_max': 0.2,  # possible label error: <= 20% of the 10 neighbours share the label ...
    'suspect_major_min': 0.7,  # ... and >= 70% agree on ONE other class
    'probe_C': 2.0,
}
SWAP = {2: 3, 3: 2, 5: 6, 6: 5}


class Images(torch.utils.data.Dataset):  # top-level for macOS spawn workers
    def __init__(self, paths, size, mean, std):
        self.paths, self.size = paths, size
        self.mean, self.std = torch.tensor(mean).view(3, 1, 1), torch.tensor(std).view(3, 1, 1)

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, i):
        im = Image.open(self.paths[i]).convert('RGB').resize((self.size, self.size), Image.BICUBIC)
        x = torch.from_numpy(np.asarray(im, dtype=np.float32) / 255.).permute(2, 0, 1)
        return (x - self.mean) / self.std


def extract(per, cfg):
    import timm
    dev = 'mps' if torch.backends.mps.is_available() else 'cuda' if torch.cuda.is_available() else 'cpu'
    model = timm.create_model(cfg['backbone'], pretrained=True, num_classes=0).eval().to(dev)
    dc = timm.data.resolve_model_data_config(model)
    paths = [os.path.join(audit.CONFIG['data_dir'], s, f'{i}.png') for s, i in zip(per['split'], per['id'])]
    dl = torch.utils.data.DataLoader(Images(paths, cfg['img_size'], dc['mean'], dc['std']), batch_size=cfg['batch'],
                                     num_workers=cfg['workers'], persistent_workers=cfg['workers'] > 0)
    feats, t0 = [], time.time()
    with torch.no_grad():
        for b, x in enumerate(dl):
            x = x.to(dev)
            acc = 0
            for k in range(4):
                r = torch.rot90(x, k, dims=(2, 3))
                acc = acc + model(r) + model(torch.flip(r, dims=(3,)))
            feats.append((acc / 8).float().cpu().numpy())
            if b % 50 == 0:
                print(f'[embed] {b * cfg["batch"]}/{len(paths)}  {time.time() - t0:.0f}s', flush=True)
    return np.concatenate(feats)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--recompute', action='store_true')
    args = ap.parse_args()
    cfg = CONFIG
    csv_dir = os.path.join(audit.CONFIG['out_dir'], 'csv')
    per = pd.read_csv(os.path.join(csv_dir, 'per_image.csv'))
    gm = pd.read_csv(os.path.join(csv_dir, 'grainy_matches.csv'))
    os.makedirs(cfg['cache_dir'], exist_ok=True)
    tag = cfg['backbone'].split('.')[0]
    cache = {s: os.path.join(cfg['cache_dir'], f'emb_{tag}_{s}.npy') for s in ('train', 'test')}
    ids_path = os.path.join(cfg['cache_dir'], f'emb_{tag}_ids.csv')

    if args.recompute or not all(os.path.exists(p) for p in cache.values()):
        E = extract(per, cfg)
        for s in ('train', 'test'):
            np.save(cache[s], E[(per['split'] == s).to_numpy()].astype(np.float32))
        per[['id', 'split']].to_csv(ids_path, index=False)
    ids = pd.read_csv(ids_path)
    assert (ids['id'].to_numpy() == per['id'].to_numpy()).all(), 'cache order differs from per_image.csv: use --recompute'
    E = np.zeros((len(per), np.load(cache['train'], mmap_mode='r').shape[1]), np.float32)
    for s in ('train', 'test'):
        E[(per['split'] == s).to_numpy()] = np.load(cache[s])
    E /= np.linalg.norm(E, axis=1, keepdims=True)
    print(f'[embed] {E.shape} from {cfg["backbone"]}')

    out = cfg['out_dir']
    if os.path.isdir(out):
        for f in os.listdir(out):
            os.remove(os.path.join(out, f))
    os.makedirs(out, exist_ok=True)
    data_dir = audit.CONFIG['data_dir']
    is_tr, is_te = (per['split'] == 'train').to_numpy(), (per['split'] == 'test').to_numpy()
    clean = is_tr & (per['category'] == 'clean').to_numpy()
    y = per['label'].to_numpy()
    ci = np.where(clean)[0]

    # ---- k-NN among clean train (leave-one-out) + outlier score for every image ----
    k = cfg['knn_k']
    S = E @ E[ci].T                                         # cosine similarity to every clean train image
    S[ci, np.arange(len(ci))] = -2                          # exclude self
    nn = np.argpartition(-S, k, axis=1)[:, :k]
    nn_sim = np.take_along_axis(S, nn, axis=1)
    outlier = 1 - nn_sim.mean(1)
    nn_lab = y[ci][nn]
    counts = np.stack([(nn_lab == c).sum(1) for c in range(N_CLASSES)], 1)
    knn_pred = counts.argmax(1)
    knn_agree = np.where(is_tr, counts[np.arange(len(per)), np.clip(y, 0, None)] / k, np.nan)
    del S

    # ---- linear probe: 5-fold OOF on clean train, then fit on all clean ----
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import f1_score
    from sklearn.model_selection import StratifiedKFold
    probe_p = np.zeros((len(per), N_CLASSES))
    for a, b in StratifiedKFold(5, shuffle=True, random_state=cfg['seed']).split(E[ci], y[ci]):
        m = LogisticRegression(C=cfg['probe_C'], max_iter=3000).fit(E[ci[a]], y[ci[a]])
        probe_p[ci[b]] = m.predict_proba(E[ci[b]])
    m = LogisticRegression(C=cfg['probe_C'], max_iter=3000).fit(E[ci], y[ci])
    rest = np.setdiff1d(np.arange(len(per)), ci)
    probe_p[rest] = m.predict_proba(E[rest])
    probe_pred, probe_conf = probe_p.argmax(1), probe_p.max(1)
    f1_oof = f1_score(y[ci], probe_pred[ci], average='weighted')
    f1_knn = f1_score(y[ci], knn_pred[ci], average='weighted')
    print(f'[probe] clean train OOF weighted F1 {f1_oof:.3f} | k-NN leave-one-out {f1_knn:.3f}')

    # ---- UMAP ----
    import umap
    from sklearn.decomposition import PCA
    Z = PCA(cfg['pca_dim'], random_state=cfg['seed']).fit_transform(E)
    U = umap.UMAP(n_neighbors=cfg['umap_neighbors'], min_dist=cfg['umap_min_dist'], metric='cosine',
                  random_state=cfg['seed']).fit_transform(Z)

    res = per[['id', 'split', 'label', 'category']].assign(
        umap_x=U[:, 0], umap_y=U[:, 1], knn_pred=knn_pred, knn_agree=knn_agree, outlier=outlier,
        probe_pred=probe_pred, probe_conf=probe_conf)
    res.round(4).to_csv(os.path.join(out, 'embedding_analysis.csv'), index=False)

    # ---- figures ----
    cmap = plt.get_cmap('tab10')
    fig, ax = plt.subplots(figsize=(9, 7.5))
    ax.scatter(U[is_te, 0], U[is_te, 1], s=1, c='#dddddd', label='test', rasterized=True)
    for c in range(N_CLASSES):
        mk = clean & (y == c)
        ax.scatter(U[mk, 0], U[mk, 1], s=4, color=cmap(c), label=f'{c} {audit.short(c)}', rasterized=True)
    ax.set_title(f'UMAP of CNN embeddings ({tag}, 8-view average): clean train by class, test in grey', fontsize=10)
    ax.set_xlabel('UMAP 1 (no unit)'); ax.set_ylabel('UMAP 2 (no unit)'); ax.legend(markerscale=4, fontsize=8, loc='best')
    audit.savefig(fig, os.path.join(out, 'umap_by_class.png'))

    groups = [('test clean', is_te & (per['category'] == 'clean').to_numpy(), '#cccccc', 1),
              ('test smooth (blurred)', (per['category'] == 'smooth').to_numpy(), '#7fb3d5', 2),
              ('test tinted', (per['category'] == 'tinted').to_numpy(), '#b07cc6', 2),
              ('train clean', clean, '#1f4e8c', 3),
              ('train grainy', (per['category'] == 'grainy').to_numpy(), '#e07b00', 3),
              ('train alien', (per['category'] == 'alien').to_numpy(), '#d62728', 6)]
    fig, ax = plt.subplots(figsize=(9, 7.5))
    for name, mk, col, sz in groups:
        ax.scatter(U[mk, 0], U[mk, 1], s=sz, c=col, label=f'{name} ({mk.sum()})', rasterized=True)
    ax.set_title('UMAP of CNN embeddings by category', fontsize=10)
    ax.set_xlabel('UMAP 1 (no unit)'); ax.set_ylabel('UMAP 2 (no unit)'); ax.legend(markerscale=4, fontsize=8)
    audit.savefig(fig, os.path.join(out, 'umap_by_category.png'))

    from sklearn.metrics import confusion_matrix
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))
    for ax, (name, pred, f1) in zip(axes, [('k-NN (k=10, leave-one-out)', knn_pred, f1_knn), ('linear probe (5-fold OOF)', probe_pred, f1_oof)]):
        cm = confusion_matrix(y[ci], pred[ci], labels=range(N_CLASSES), normalize='true')
        ax.imshow(cm, cmap='Blues', vmin=0, vmax=1)
        for (i, j), v in np.ndenumerate(cm):
            if v >= 0.05:
                ax.text(j, i, f'{v:.2f}', ha='center', va='center', fontsize=7, color='white' if v > 0.5 else 'black')
        ax.set_xticks(range(N_CLASSES)); ax.set_yticks(range(N_CLASSES))
        ax.set_xlabel('predicted class'); ax.set_ylabel('true (given) class')
        ax.set_title(f'{name}: weighted F1 {f1:.3f}\n(row-normalised: share of each true class)', fontsize=9)
    fig.suptitle('Clean train: which classes the embeddings confuse')
    audit.savefig(fig, os.path.join(out, 'confusion_knn_probe.png'))

    order = ['clean', 'wrong_label_dup', 'grainy', 'alien', 'broken', 'blank', 'tinted', 'smooth', 'grainy_like', 'alien_candidate']
    cats = [c for c in order if (res['category'] == c).any()]
    fig, ax = plt.subplots(figsize=(11, 4))
    ax.boxplot([res.loc[(res['category'] == c) & (res['split'] == s), 'outlier'] for c in cats
                for s in ['train', 'test'] if ((res['category'] == c) & (res['split'] == s)).any()], showfliers=False)
    labels = [f'{s}\n{c}' for c in cats for s in ['train', 'test'] if ((res['category'] == c) & (res['split'] == s)).any()]
    ax.set_xticks(range(1, len(labels) + 1)); ax.set_xticklabels(labels, fontsize=7)
    ax.set_ylabel('outlier score = 1 - mean cosine similarity\nto the 10 nearest clean train images', fontsize=8)
    ax.set_title('How far each category is from the clean training set (embedding space)', fontsize=10)
    audit.savefig(fig, os.path.join(out, 'outlier_score_by_category.png'))

    nn_share = counts.max(1) / k
    sus = res[clean & (knn_agree <= cfg['suspect_agree_max']) & (nn_share >= cfg['suspect_major_min'])].copy()
    sus = sus.assign(share=nn_share[sus.index]).sort_values('share', ascending=False)
    audit.make_sheet(data_dir, [('train', r.id, f'{r.id[-5:]} y={r.label} nn={r.knn_pred} {r.share:.0%}', None) for r in sus.head(60).itertuples()],
                     os.path.join(out, 'possible_label_errors.png'),
                     f'Clean train whose 10 nearest neighbours mostly have ONE other label: {len(sus)} (caption: given y, neighbour class, share)', thumb=110)
    top = res[clean].sort_values('outlier', ascending=False).head(40)
    audit.make_sheet(data_dir, [('train', r.id, f'{r.id[-5:]} y={r.label} {r.outlier:.2f}', None) for r in top.itertuples()],
                     os.path.join(out, 'clean_train_outliers.png'), 'Clean train images farthest from the other clean images (outlier score)', thumb=110)
    top = res[is_te].sort_values('outlier', ascending=False).head(40)
    audit.make_sheet(data_dir, [('test', r.id, f'{r.id[-5:]} {r.category[:6]} {r.outlier:.2f}', audit.CAT_COLORS.get(r.category)) for r in top.itertuples()],
                     os.path.join(out, 'test_far_from_train.png'), 'Test images farthest from clean train (outlier score; frame purple = tinted)', thumb=110)

    # ---- console recap ----
    gr = res[res['category'] == 'grainy'].merge(gm[['id', 'source_status']], on='id')
    for st, d in gr.groupby(gr['source_status'] == 'confirmed'):
        print(f'[grainy {"confirmed" if st else "no source"}] n={len(d)}: probe predicts given label {np.mean(d.probe_pred == d.label):.0%}, '
              f'swapped label {np.mean(d.probe_pred == d.label.map(SWAP)):.0%}')
    al = res[res['category'] == 'alien']
    print(f'[aliens] probe agrees with the given label {np.mean(al.probe_pred == al.label):.0%}')
    print(f'[label check] possible label errors in clean train: {len(sus)}')
    print('[test prior] probe predictions on test (%):', (res[is_te]['probe_pred'].value_counts(normalize=True).sort_index() * 100).round(1).to_dict())
    print(f'[done] -> {out}')


if __name__ == '__main__':
    main()
