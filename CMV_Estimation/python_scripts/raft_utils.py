"""
RAFT Inference Utility Functions
Image loading, preprocessing, visualization, and flow analysis utilities.

Author: Jonathan Huth
Date: January 2026
"""

import cv2
import numpy as np
import torch
from pathlib import Path
import torchvision.transforms.v2 as v2

def load_image(filepath, h=512, w=512):
    """Load and preprocess image (resize, normalize)."""
    img = cv2.imread(str(filepath))
    if img is None:
        raise FileNotFoundError(f"Cannot load {filepath}")
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (w, h))
    img = img.astype(np.float32) / 255.0
    return img


def preprocess_images(img1, img2):
    """Convert to torch tensors (1, 3, H, W)."""
    img1 = torch.from_numpy(img1).permute(2, 0, 1).unsqueeze(0)
    img2 = torch.from_numpy(img2).permute(2, 0, 1).unsqueeze(0)
    toImg = v2.ToImage()
    img1 = toImg(img1/255.0)
    img2 = toImg(img2/255.0)

    toDtype = v2.ToDtype(torch.float32, scale=True)
    img1_t = toDtype(img1)
    img2_t = toDtype(img2)
    
    mu = img1_t.mean() 
    sigma = img1_t.std()

    normalize = v2.Normalize(mean=[mu, mu, mu], std=[sigma, sigma, sigma])   # Inverse formula

    img1_t = normalize(img1)
    img2_t = normalize(img2)
    
    return img1_t, img2_t


def flow_to_xy_components(flow_np):
    """Extract x, y components from flow field."""
    Vx = flow_np[..., 0]
    Vy = flow_np[..., 1]
    return Vx, Vy
