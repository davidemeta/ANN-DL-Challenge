"""Batched augmentation on the GPU (or MPS/CPU), one random draw per image.

Doing it on the device avoids DataLoader workers: Kaggle has 4 vCPUs, and macOS spawns
workers that would each copy the image array. All functions take float images in [0, 1],
shape (B, 3, H, W).

Order used in training (see `train_transform`), matching what the EDA measured on the test set:
    hue rotation (HSV, p=0.4, full circle) -> Gaussian blur at 256 px (p=0.35, sigma 0.4-1.2)
    -> brightness/contrast -> resize to img_size -> random rotation + flip (+ mild zoom)
    -> normalize
"""
import math

import torch
import torch.nn.functional as F

EPS = 1e-8
BLUR_KERNEL = 9  # covers +-3 sigma for sigma up to 1.33 px


# ---------------------------------------------------------------- colour
def rgb_to_hsv(x):
    r, g, b = x.unbind(1)
    maxc, _ = x.max(1)
    minc, _ = x.min(1)
    delta = maxc - minc
    d = delta.clamp_min(EPS)
    h = torch.where(maxc == r, (g - b) / d,
                    torch.where(maxc == g, 2.0 + (b - r) / d, 4.0 + (r - g) / d))
    h = torch.where(delta > 0, (h / 6.0) % 1.0, torch.zeros_like(h))
    s = torch.where(maxc > 0, delta / maxc.clamp_min(EPS), torch.zeros_like(maxc))
    return torch.stack([h, s, maxc], 1)


def hsv_to_rgb(x):
    h, s, v = x.unbind(1)
    h6 = h * 6.0
    i = torch.floor(h6)
    f = h6 - i
    p = v * (1 - s)
    q = v * (1 - s * f)
    t = v * (1 - s * (1 - f))
    i = (i.long() % 6).unsqueeze(1)
    r = torch.stack([v, q, p, p, t, v], 1).gather(1, i)
    g = torch.stack([t, v, v, q, p, p], 1).gather(1, i)
    b = torch.stack([p, p, t, v, v, q], 1).gather(1, i)
    return torch.cat([r, g, b], 1)


def hue_rotate(x, turns):
    """Rotate the HSV hue of each image by `turns` (fraction of the circle, 0.5 = 180 deg)."""
    hsv = rgb_to_hsv(x)
    hsv[:, 0] = (hsv[:, 0] + turns.view(-1, 1, 1)) % 1.0
    return hsv_to_rgb(hsv).clamp(0, 1)


def to_gray(x):
    lum = 0.299 * x[:, 0] + 0.587 * x[:, 1] + 0.114 * x[:, 2]
    return lum.unsqueeze(1).expand(-1, 3, -1, -1).contiguous()


def brightness_contrast(x, b_factor, c_factor):
    mean = x.mean(dim=(1, 2, 3), keepdim=True)
    x = (x - mean) * c_factor.view(-1, 1, 1, 1) + mean
    return (x * b_factor.view(-1, 1, 1, 1)).clamp(0, 1)


# ---------------------------------------------------------------- blur
def gaussian_blur(x, sigma):
    """Separable Gaussian blur with a different sigma (px) per image; sigma 0 = unchanged."""
    B, C, H, W = x.shape
    half = BLUR_KERNEL // 2
    coords = torch.arange(-half, half + 1, device=x.device, dtype=x.dtype)
    s = sigma.to(x.dtype).clamp_min(1e-3).view(-1, 1)
    k = torch.exp(-(coords.view(1, -1) ** 2) / (2 * s ** 2))
    k = k / k.sum(1, keepdim=True)                               # (B, K)
    k = k.repeat_interleave(C, 0)                                 # (B*C, K)
    y = x.reshape(1, B * C, H, W)
    y = F.conv2d(F.pad(y, (half, half, 0, 0), mode='reflect'), k.view(B * C, 1, 1, -1), groups=B * C)
    y = F.conv2d(F.pad(y, (0, 0, half, half), mode='reflect'), k.view(B * C, 1, -1, 1), groups=B * C)
    return y.view(B, C, H, W)


# ---------------------------------------------------------------- geometry
def resize(x, size):
    if x.shape[-1] == size:
        return x
    return F.interpolate(x, size=(size, size), mode='bilinear', antialias=True, align_corners=False)


def rotate_flip_zoom(x, angle_rad, flip, zoom):
    """Rotation by any angle, optional horizontal flip, zoom > 1 = zoom in. Reflection padding."""
    cos, sin = torch.cos(angle_rad), torch.sin(angle_rad)
    theta = torch.zeros(x.shape[0], 2, 3, device=x.device, dtype=x.dtype)
    theta[:, 0, 0] = cos * flip / zoom
    theta[:, 0, 1] = -sin / zoom
    theta[:, 1, 0] = sin * flip / zoom
    theta[:, 1, 1] = cos / zoom
    grid = F.affine_grid(theta, list(x.shape), align_corners=False)
    return F.grid_sample(x, grid, mode='bilinear', padding_mode='reflection', align_corners=False)


# ---------------------------------------------------------------- pipelines
def _apply_subset(x, mask, fn, *params):
    if mask.any():
        x = x.clone()
        x[mask] = fn(x[mask], *[p[mask] for p in params])
    return x


def test_like(x, hue_turns, blur_sigma):
    """Deterministic test-style distortion: per-image hue shift (0 = none) and blur (0 = none)."""
    x = _apply_subset(x, hue_turns != 0, hue_rotate, hue_turns)
    return _apply_subset(x, blur_sigma > 0, gaussian_blur, blur_sigma)


def train_transform(x, cfg, gen, gray=False):
    """x: (B, 3, 256, 256) float in [0, 1] on the device. cfg: the `aug` section of the config.
    gen: CPU torch.Generator, so a run is reproducible whatever the device."""
    B, dev = x.shape[0], x.device

    def u(lo=0.0, hi=1.0):
        return (torch.rand(B, generator=gen) * (hi - lo) + lo).to(dev)

    if gray:
        x = to_gray(x)
    elif cfg['hue_p'] > 0:
        mask = u() < cfg['hue_p']
        x = _apply_subset(x, mask, hue_rotate, u(-cfg['hue_max_turn'], cfg['hue_max_turn']))
    if cfg['blur_p'] > 0:
        lo, hi = cfg['blur_sigma']
        x = _apply_subset(x, u() < cfg['blur_p'], gaussian_blur, u(lo, hi))
    if cfg['brightness'] > 0 or cfg['contrast'] > 0:
        x = brightness_contrast(x, u(1 - cfg['brightness'], 1 + cfg['brightness']),
                                u(1 - cfg['contrast'], 1 + cfg['contrast']))
    x = resize(x, cfg['img_size'])
    if cfg['rotate']:
        flip = torch.where(u() < 0.5, -1.0, 1.0).to(x.dtype)
        x = rotate_flip_zoom(x, u(0, 2 * math.pi), flip, u(*cfg['zoom']))
    return x


def eval_transform(x, img_size, gray=False):
    if gray:
        x = to_gray(x)
    return resize(x, img_size)
