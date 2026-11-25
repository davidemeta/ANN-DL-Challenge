"""Shared data loading for AN2DL 2026-27 (SPAGHETTI-1 galaxy images).

Layout expected under DATA_DIR (Kaggle download, kept on the shared drive, never in git):
    train/train_00000.png ... train_04327.png   (256x256 RGB PNG)
    test/test_00000.png  ... test_14735.png
    train.csv              (id, label)
    sample_submission.csv  (id, label)

PNG decoding is the bottleneck: call `build_cache` once, then load the uint8 arrays
(test with mmap_mode='r', ~2.9 GB).
"""
import os
import numpy as np
import pandas as pd
from PIL import Image

N_CLASSES = 10
IMG_SIZE = 256
CLASS_NAMES = [
    'spaghetti_disturbed', 'gemelli_merging', 'gnocchi_round_smooth', 'ravioli_in_between_smooth',
    'penne_cigar', 'fusilli_barred_spiral', 'fusilli_tight_spiral', 'riccioli_loose_spiral',
    'linguine_edge_on_no_bulge', 'lasagna_edge_on_bulge',
]


def read_image(path):
    """Read a PNG as (256, 256, 3) uint8, forcing RGB (guards against grayscale/RGBA corrupted files)."""
    with Image.open(path) as im:
        if im.mode != 'RGB':
            im = im.convert('RGB')
        arr = np.asarray(im, dtype=np.uint8)
    if arr.shape != (IMG_SIZE, IMG_SIZE, 3):
        raise ValueError(f'{path}: unexpected shape {arr.shape}')
    return arr


def load_tables(data_dir):
    train_df = pd.read_csv(os.path.join(data_dir, 'train.csv'))
    test_df = pd.read_csv(os.path.join(data_dir, 'sample_submission.csv'))[['id']]
    return train_df, test_df


def build_cache(data_dir, cache_dir, split='train'):
    """Decode all PNGs of a split into one uint8 .npy of shape (N, 256, 256, 3), ordered like the CSV."""
    train_df, test_df = load_tables(data_dir)
    df = train_df if split == 'train' else test_df
    out_path = os.path.join(cache_dir, f'{split}_uint8.npy')
    if os.path.exists(out_path):
        return out_path
    os.makedirs(cache_dir, exist_ok=True)
    arr = np.lib.format.open_memmap(out_path, mode='w+', dtype=np.uint8,
                                    shape=(len(df), IMG_SIZE, IMG_SIZE, 3))
    bad = []
    for i, img_id in enumerate(df['id']):
        try:
            arr[i] = read_image(os.path.join(data_dir, split, f'{img_id}.png'))
        except Exception as e:  # keep going, report at the end
            bad.append((img_id, repr(e)))
    arr.flush()
    if bad:
        print(f'[build_cache] {len(bad)} unreadable/odd files in {split}, e.g. {bad[:5]}')
    return out_path


def load_cache(cache_dir, split='train', mmap=True):
    return np.load(os.path.join(cache_dir, f'{split}_uint8.npy'), mmap_mode='r' if mmap else None)


try:
    import torch
    from torch.utils.data import Dataset

    class GalaxyDataset(Dataset):
        """images: (N, H, W, 3) uint8 array (or memmap); labels: (N,) ints or None.
        transform: callable on a uint8 CHW tensor (e.g. torchvision.transforms.v2 pipeline
        ending with ToDtype(float32, scale=True) + Normalize with the backbone's mean/std)."""

        def __init__(self, images, labels=None, indices=None, transform=None):
            self.images = images
            self.labels = None if labels is None else np.asarray(labels, dtype=np.int64)
            self.indices = np.arange(len(images)) if indices is None else np.asarray(indices)
            self.transform = transform

        def __len__(self):
            return len(self.indices)

        def __getitem__(self, i):
            j = self.indices[i]
            x = torch.from_numpy(np.ascontiguousarray(self.images[j])).permute(2, 0, 1)  # CHW uint8
            if self.transform is not None:
                x = self.transform(x)
            if self.labels is None:
                return x
            return x, int(self.labels[j])
except ImportError:  # torch not installed (e.g. EDA-only environment)
    pass
