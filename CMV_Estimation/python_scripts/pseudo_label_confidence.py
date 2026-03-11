"""
Confidence-based pseudo-label filtering.
Masks out low-quality pseudo-labels to prevent training on noise.
"""

import torch
import numpy as np
import cv2

def compute_pseudo_label_confidence(flow, method='magnitude_with_smoothness'):
    """
    Compute confidence scores for pseudo-label optical flow.
    
    Low confidence = unreliable pseudo-label → should be masked out during training.
    
    Args:
        flow: Optical flow (B, 2, H, W) or (H, W, 2) numpy
        method: 'magnitude' (simple) or 'magnitude_with_smoothness' (better)
    
    Returns:
        confidence: (B, H, W) or (H, W) float32, range [0, 1]
    """
    if isinstance(flow, torch.Tensor):
        flow = flow.cpu().numpy()
    
    # Handle 3D input (H, W, 2)
    if flow.ndim == 3:
        flow = np.expand_dims(flow, 0)
        squeeze_output = True
    else:
        squeeze_output = False
    
    B, H, W, C = flow.shape if flow.shape[-1] == 2 else (flow.shape[0], flow.shape[2], flow.shape[3], flow.shape[1])
    
    # Reshape if needed (B, 2, H, W)
    if flow.shape[1] == 2:
        flow = flow.transpose(0, 2, 3, 1)
    
    confidence_scores = []
    
    for b in range(flow.shape[0]):
        flow_b = flow[b]  # (H, W, 2)
        u, v = flow_b[:, :, 0], flow_b[:, :, 1]
        magnitude = np.sqrt(u**2 + v**2)
        
        if method == 'magnitude':
            # Simple: normalize magnitude
            # Very small flow (< 0.1) = low confidence
            # Very large flow (> 20) = also low confidence (likely outliers)
            confidence = np.exp(-0.5 * (magnitude - 1.0)**2 / 2.0)  # Peak at 1 px/frame
            confidence = np.clip(confidence, 0, 1)
        
        elif method == 'magnitude_with_smoothness':
            # Better: also penalize locally inconsistent flow
            
            # Magnitude confidence (Gaussian centered at 1 px/frame)
            mag_confidence = np.exp(-0.5 * ((magnitude - 1.0) / 2.0)**2)
            
            # Smoothness confidence (penalize very noisy flow)
            u_grad = np.sqrt((np.gradient(u, axis=0)**2 + np.gradient(u, axis=1)**2))
            v_grad = np.sqrt((np.gradient(v, axis=0)**2 + np.gradient(v, axis=1)**2))
            grad_magnitude = u_grad + v_grad
            
            # High gradient = noisy/inconsistent
            smooth_confidence = np.exp(-0.1 * grad_magnitude)
            
            # Combine
            confidence = (mag_confidence * 0.7 + smooth_confidence * 0.3)
            confidence = np.clip(confidence, 0, 1)
        
        else:
            raise ValueError(f"Unknown method: {method}")
        
        confidence_scores.append(confidence)
    
    confidence_array = np.stack(confidence_scores, axis=0)  # (B, H, W)
    
    if squeeze_output:
        confidence_array = confidence_array.squeeze(0)
    
    return confidence_array.astype(np.float32)


def mask_low_confidence_pseudo_labels(flow_pseudo, confidence_threshold=0.5):
    """
    Zero out pseudo-labels with confidence below threshold.
    
    Args:
        flow_pseudo: Optical flow (B, 2, H, W) torch tensor
        confidence_threshold: Minimum confidence [0, 1]
    
    Returns:
        flow_pseudo_masked: Same shape, low-confidence pixels zeroed
        confidence_mask: (B, H, W) binary mask of valid regions
    """
    # Compute confidence
    confidence = compute_pseudo_label_confidence(flow_pseudo, method='magnitude_with_smoothness')
    
    # Convert to torch if needed
    if isinstance(confidence, np.ndarray):
        confidence = torch.from_numpy(confidence).float()
    
    # Create mask
    valid_mask = (confidence >= confidence_threshold).float()
    
    # Apply to flow
    flow_pseudo = torch.from_numpy(flow_pseudo).permute(2, 0, 1).float()
    flow_masked = flow_pseudo.clone()
    for b in range(flow_pseudo.shape[0]):
        flow_masked[b, :] = flow_masked[b, :] * valid_mask[b].unsqueeze(0)
    
    return flow_masked, valid_mask, confidence


# Example usage
if __name__ == "__main__":
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    
    # Simulate weak pseudo-labels (like your Farnebäck output on dark clouds)
    flow_weak = np.random.randn(1, 512, 512, 2) * 0.3  # Mean mag ≈ 0.23
    flow_weak[0, 100:200, 100:200, :] = np.random.randn(100, 100, 2) * 3.0  # One region with real motion
    
    # Compute confidence
    confidence = compute_pseudo_label_confidence(flow_weak)
    
    # Apply masking
    flow_torch = torch.from_numpy(flow_weak).permute(0, 3, 1, 2)
    flow_masked, valid_mask, conf = mask_low_confidence_pseudo_labels(
        flow_torch, 
        confidence_threshold=0.3
    )
    
    # Visualize
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # Magnitude of original flow
    mag_orig = np.sqrt(flow_weak[0, :, :, 0]**2 + flow_weak[0, :, :, 1]**2)
    im0 = axes[0, 0].imshow(mag_orig, cmap='jet')
    axes[0, 0].set_title('Original Farnebäck Magnitude')
    plt.colorbar(im0, ax=axes[0, 0])
    
    # Confidence scores
    im1 = axes[0, 1].imshow(confidence[0], cmap='viridis', vmin=0, vmax=1)
    axes[0, 1].set_title('Confidence Scores (soft weights)')
    plt.colorbar(im1, ax=axes[0, 1])
    
    # Binary mask
    im2 = axes[1, 0].imshow(valid_mask[0].cpu().numpy(), cmap='gray')
    axes[1, 0].set_title('Valid Mask (threshold=0.3)')
    plt.colorbar(im2, ax=axes[1, 0])
    
    # Masked flow magnitude
    mag_masked = np.sqrt(flow_masked[0, 0].cpu().numpy()**2 + flow_masked[0, 1].cpu().numpy()**2)
    im3 = axes[1, 1].imshow(mag_masked, cmap='jet')
    axes[1, 1].set_title('Masked Farnebäck Magnitude')
    plt.colorbar(im3, ax=axes[1, 1])
    
    plt.tight_layout()
    plt.savefig('pseudo_label_masking.png', dpi=100, bbox_inches='tight')
    print("✓ Saved: pseudo_label_masking.png")
    
    # Statistics
    print(f"\nPseudo-label Quality Analysis:")
    print(f"Original - Mean magnitude: {mag_orig.mean():.4f}")
    print(f"Original - % near-zero: {(mag_orig < 0.1).sum()/mag_orig.size*100:.1f}%")
    print(f"\nAfter masking (threshold=0.3):")
    print(f"Valid pixels: {valid_mask.sum().item() / valid_mask.numel() * 100:.1f}%")
    print(f"Mean confidence: {confidence.mean():.4f}")
    print(f"Median confidence: {np.median(confidence):.4f}")
