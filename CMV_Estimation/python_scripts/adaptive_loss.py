"""
Adaptive Loss Weighting based on Scene Characteristics
Automatically adjusts loss weights for cloudy vs clear-sky images
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import cv2
import numpy as np


def compute_cloud_coverage(img):
    """
    Estimate cloud coverage percentage from image.
    
    Args:
        img: Image tensor (B, 3, H, W) in [-1, 1] or [0, 1]
    
    Returns:
        coverage: (B,) cloud coverage fraction [0, 1]
    """
    B = img.shape[0]
    coverage = []
    
    for b in range(B):
        # Convert to numpy
        img_np = img[b].cpu().numpy()  # (3, H, W)
        img_np = img_np.transpose(1, 2, 0)  # (H, W, 3)
        
        # Denormalize if needed
        if img_np.min() < 0:
            img_np = (img_np + 1.0) / 2.0
        
        img_np = (img_np * 255).astype(np.uint8)
        
        # Convert to HSV
        hsv = cv2.cvtColor(img_np, cv2.COLOR_RGB2HSV)
        
        # Clouds are bright (high V in HSV)
        # Threshold: V > 150 (out of 255)
        cloud_mask = hsv[:, :, 2] > 120
        cloud_coverage = cloud_mask.sum() / cloud_mask.size
        coverage.append(cloud_coverage)
    
    return torch.tensor(coverage, dtype=torch.float32)


class AdaptiveHybridFlowLoss(nn.Module):
    """
    Hybrid flow loss with adaptive weighting based on scene characteristics.
    
    For sparse/clear images: emphasize supervised loss
    For dense/cloudy images: balance all components
    """
    
    def __init__(self, base_lambda_photo=0.1, base_lambda_smooth=0.01, 
                 base_lambda_supervised=0.9):
        super().__init__()
        self.base_lambda_photo = base_lambda_photo
        self.base_lambda_smooth = base_lambda_smooth
        self.base_lambda_supervised = base_lambda_supervised
        
        from python_scripts.losses_finetuning import PhotometricLoss, SmoothnessLoss, SupervisedFlowLoss
        self.photo_loss = PhotometricLoss()
        self.smooth_loss = SmoothnessLoss()
        self.supervised_loss = SupervisedFlowLoss(use_confidence_weighting=True)
    
    def forward(self, flow_pred, img1, img2, flow_pseudo, 
                confidence_mask=None, valid_mask=None):
        """
        Compute adaptive hybrid loss based on image content.
        
        Args:
            flow_pred: Predicted flow (B, 2, H, W)
            img1: First image (B, 3, H, W)
            img2: Second image (B, 3, H, W)
            flow_pseudo: Pseudo-label flow (B, 2, H, W)
            confidence_mask: Confidence weights (B, H, W)
            valid_mask: Valid regions (B, H, W)
        
        Returns:
            (loss, loss_dict, weights)
        """
        # Estimate cloud coverage for adaptive weighting
        cloud_coverage = compute_cloud_coverage(img1)  # (B,)
        
        # Compute individual losses
        loss_photo = self.photo_loss(img1, img2, flow_pred)
        loss_smooth = self.smooth_loss(flow_pred)
        loss_supervised = self.supervised_loss(
            flow_pred, flow_pseudo,
            confidence_mask=confidence_mask,
            valid_mask=valid_mask
        )
        
        # Adaptive weighting based on cloud coverage
        B = img1.shape[0]
        weighted_losses = []
        weights_per_sample = []
        
        for b in range(B):
            coverage = cloud_coverage[b].item()  # 0 = clear sky, 1 = full clouds
            
            # Adapt loss weights
            if coverage < 0.2:  # Clear/sparse clouds
                # Emphasize supervised learning on cloudy regions
                lambda_photo = self.base_lambda_photo * 0.5  # Reduce photo loss
                lambda_smooth = self.base_lambda_smooth * 0.8  # Reduce smoothness
                lambda_supervised = 0.95  # Maximize supervised signal
            
            elif coverage > 0.7:  # Dense clouds
                # Balance all components (more photometric signal available)
                lambda_photo = self.base_lambda_photo * 1.5
                lambda_smooth = self.base_lambda_smooth * 1.2
                lambda_supervised = 0.8
            
            else:  # Medium coverage
                # Use default balanced weights
                lambda_photo = self.base_lambda_photo
                lambda_smooth = self.base_lambda_smooth
                lambda_supervised = self.base_lambda_supervised
            
            # Normalize weights to sum to 1
            total = lambda_photo + lambda_smooth + lambda_supervised
            lambda_photo /= total
            lambda_smooth /= total
            lambda_supervised /= total
            
            weighted_loss = (
                lambda_photo * loss_photo +
                lambda_smooth * loss_smooth +
                lambda_supervised * loss_supervised
            )
            weighted_losses.append(weighted_loss)
            weights_per_sample.append({
                'photo': lambda_photo,
                'smooth': lambda_smooth,
                'supervised': lambda_supervised,
                'coverage': coverage
            })
        
        # Average loss across batch
        total_loss = sum(weighted_losses) / B
        
        return total_loss, {
            'loss_photo': loss_photo.item(),
            'loss_smooth': loss_smooth.item(),
            'loss_supervised': loss_supervised.item(),
        }, weights_per_sample


class ClearSkyAwareLoss(nn.Module):
    """
    Loss that explicitly masks clear-sky regions to prevent spurious motion.
    """
    
    def __init__(self):
        super().__init__()
        from python_scripts.losses_finetuning import PhotometricLoss, SmoothnessLoss, SupervisedFlowLoss
        self.photo_loss = PhotometricLoss()
        self.smooth_loss = SmoothnessLoss()
    
    def identify_clear_sky_regions(self, img, brightness_threshold=120):
        """
        Identify clear-sky regions based on brightness.
        
        Args:
            img: Image (B, 3, H, W) in [-1, 1] or [0, 1]
            brightness_threshold: V threshold in HSV (0-255 scale)
        
        Returns:
            clear_sky_mask: (B, H, W) binary mask, 1 = clear sky
        """
        B, _, H, W = img.shape
        masks = []
        
        for b in range(B):
            img_np = img[b].cpu().numpy().transpose(1, 2, 0)
            
            # Denormalize
            if img_np.min() < 0:
                img_np = (img_np + 1.0) / 2.0
            img_np = (img_np * 255).astype(np.uint8)
            
            # Use V channel in HSV
            hsv = cv2.cvtColor(img_np, cv2.COLOR_RGB2HSV)
            clear_sky = (hsv[:, :, 2] > brightness_threshold).astype(np.float32)
            masks.append(clear_sky)
        
        return torch.from_numpy(np.stack(masks)).float()
    
    def forward(self, flow_pred, img1, img2,
                confidence_mask=None, valid_mask=None):
        """
        Compute loss with explicit clear-sky masking.
        
        Clear-sky regions: Use only smoothness loss
        Cloudy regions: Use full hybrid loss
        
        Args:
            flow_pred: Predicted flow (B, 2, H, W)
            img1: First image (B, 3, H, W)
            img2: Second image (B, 3, H, W)
            flow_pseudo: Pseudo-label flow (B, 2, H, W)
            confidence_mask: Confidence weights (B, H, W)
            valid_mask: Valid regions (B, H, W)
        
        Returns:
            (loss, loss_dict)
        """
        # Identify clear-sky regions
        clear_sky_mask = self.identify_clear_sky_regions(img1)  # (B, H, W)
        cloud_mask = 1.0 - clear_sky_mask  # Rename for clarity

        loss_photo = self.photo_loss(img1, img2, flow_pred)
        loss_smooth = self.smooth_loss(flow_pred)

        loss_clear = 0.5 * smooth_loss * clear_sky_mask
        loss_cloud = (self.lambda_photo * photo_loss + self.lambda_smooth * smooth_loss) * cloud_mask
        total_loss = (loss_clear.mean() + loss_cloud.mean())

        # # FLIPPED LOGIC:
        # loss_dark_clouds = 0.8 * loss_photo + 0.2 * loss_smooth
        # loss_bright = 0.9 * loss_supervised + 0.1 * loss_smooth
        # 
        # total_loss = dark_mask.mean() * loss_dark_clouds + clear_sky_mask.mean() * loss_bright  # Note: clear_sky_mask (bright)
        
        return total_loss, {'loss_photo': loss_photo.item(), 'loss_smooth': loss_smooth.item(),
                            'clear_fraction': clear_sky_mask.mean().item()}