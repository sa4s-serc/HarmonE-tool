# utility/drift_utils.py
import os
from pathlib import Path
import numpy as np
from PIL import Image
from scipy.stats import entropy

def luminance_histogram(img, bins=64):
    """Compute luminance histogram (Y from RGB via Rec.601) normalized to sum=1."""
    try:
        if isinstance(img, (str, Path)):   # file path
            im = Image.open(img).convert("RGB")
        elif isinstance(img, Image.Image): # already a PIL image
            im = img.convert("RGB")
        else:
            return None
    except Exception:
        return None
    
    arr = np.asarray(im, dtype=np.float32)
    y = 0.299*arr[...,0] + 0.587*arr[...,1] + 0.114*arr[...,2]
    y = np.clip(y, 0, 255)
    hist, _ = np.histogram(y, bins=bins, range=(0,255), density=False)
    hist = hist.astype(np.float64)
    hist = hist / (hist.sum() + 1e-12)
    return hist

def kl_divergence(p, q):
    """KL(p || q) with small epsilon."""
    p = np.asarray(p, dtype=np.float64) + 1e-10
    q = np.asarray(q, dtype=np.float64) + 1e-10
    return entropy(p, q)

def window_hist_stats(image_paths, bins=64):
    """Aggregate luminance hist + simple moments over a list of images."""
    hsum = None
    means, stds = [], []
    for p in image_paths:
        h = luminance_histogram(p, bins=bins)
        if h is None: 
            continue
        if hsum is None:
            hsum = np.zeros_like(h, dtype=np.float64)
        hsum += h
        # reconstruct moments from histogram
        centers = np.linspace(0, 255, len(h), endpoint=False) + (255/len(h))/2
        mean = np.sum(h * centers)
        var = np.sum(h * (centers-mean)**2)
        means.append(mean)
        stds.append(np.sqrt(max(var, 1e-9)))
    if hsum is None:
        return None, None, None
    hsum /= (hsum.sum() + 1e-12)
    mean_val = float(np.mean(means)) if means else None
    std_val  = float(np.mean(stds))  if stds else None
    return hsum, mean_val, std_val
