"""
Pseudo-label generation from Farnebäck optical flow.
Generates confidence-weighted supervision for RAFT fine-tuning.
"""

import cv2
import numpy as np
import torch
from pathlib import Path
from python_scripts.pseudo_label_confidence import mask_low_confidence_pseudo_labels, compute_pseudo_label_confidence

def compute_farneback_flow(img1, img2, show_debug=False):
    """
    Compute optical flow using Farnebäck algorithm.
    
    Args:
        img1: First image (H, W, 3) uint8 or float32
        img2: Second image (H, W, 3) uint8 or float32
        show_debug: Print debug info
    
    Returns:
        flow: Optical flow (H, W, 2) float32
    """
    
    gray1 = cv2.cvtColor(img1, cv2.COLOR_RGB2GRAY)
    gray2 = cv2.cvtColor(img2, cv2.COLOR_RGB2GRAY)

    # Ensure uint8
    if gray1.dtype == np.float32:
        gray1 = (gray1 * 255).astype(np.uint8)
    if gray2.dtype == np.float32:
        gray2 = (gray2 * 255).astype(np.uint8)
    
    # Compute Farnebäck flow
    flow = cv2.calcOpticalFlowFarneback(
        gray1, gray2,
        None,
        pyr_scale=0.5,      # Pyramid scale
        levels=3,           # Pyramid levels
        winsize=15,         # Window size
        iterations=3,       # Iterations
        poly_n=5,          # Polynomial neighborhood
        poly_sigma=1.2,    # Polynomial sigma
        flags=0
    )
    
    if show_debug:
        mag, _ = cv2.cartToPolar(flow[..., 0], flow[..., 1])
        print(f"[DEBUG] Farnebäck flow magnitude - min: {mag.min():.3f}, "
              f"mean: {mag.mean():.3f}, max: {mag.max():.3f}")
    
    return flow


def compute_flow_confidence(flow):
    """
    Compute confidence score for optical flow.
    
    Higher confidence = more reliable flow estimate.
    
    Args:
        flow: Optical flow (H, W, 2)
        method: 'magnitude', 'gradient', or 'hybrid'
    
    Returns:
        confidence: (H, W) in range [0, 1]
    """
    u, v = flow[...,0], flow[...,1]
    mag = np.sqrt(u**2 + v**2)
    
    mag_conf = np.exp(-0.5 * ((mag - 1.0)/2.0)**2)  # Peak at 1px/frame
    
    u_dy, u_dx = np.gradient(u)      # Separate axis grads
    v_dy, v_dx = np.gradient(v)

    grad_mag = np.sqrt(u_dy**2 + u_dx**2 + v_dy**2 + v_dx**2)  # Noise penalty

    smooth_conf = np.exp(-0.1 * grad_mag)
    conf = 0.7 * mag_conf + 0.3 * smooth_conf
    confidence = np.clip(conf, 0, 1)
    
    return confidence


def generate_pseudo_labels(img1, img2, confidence_threshold, show_debug=False):
    """
    Generate pseudo-labels from Farnebäck flow with confidence weighting.
    
    Args:
        img1: First image (H, W, 3) numpy uint8 or float32
        img2: Second image (H, W, 3) numpy uint8 or float32
        confidence_threshold: Minimum confidence to keep flow
        show_debug: Print debug information
    
    Returns:
        flow_pseudo: Pseudo-label flow (H, W, 2) torch float32
        confidence_mask: Confidence weights (H, W) torch float32
        valid_mask: Binary mask of valid regions (H, W) torch float32
    """
    # Compute Farnebäck flow
    flow = compute_farneback_flow(img1, img2, show_debug=show_debug)
    
    _, valid_mask, confidence = mask_low_confidence_pseudo_labels(flow, confidence_threshold)

    flow = torch.from_numpy(flow).permute(2, 0, 1).float()

    if show_debug:
        print(f"[DEBUG] Pseudo-label generation:")
        print(f"  Confidence - min: {confidence.min():.3f}, "
              f"mean: {confidence.mean():.3f}, max: {confidence.max():.3f}")
        print(f"  Valid pixels: {valid_mask.sum():.0f} / {valid_mask.size} "
              f"({100*valid_mask.mean():.1f}%)")
    
    return flow, confidence, valid_mask
