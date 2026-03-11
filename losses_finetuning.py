"""
Loss functions for RAFT fine-tuning with confidence-weighted supervision.
Includes photometric loss and supervised loss with confidence masking.
"""
import argparse
import sys
import torch
import cv2

import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from pathlib import Path
from ezflow.models import build_model
from pytorch_msssim import ssim
from python_scripts.config_finetuning import Config
from python_scripts.evaluation_finetuning import ProxyMetrics
        
class PhotometricLoss(nn.Module):
    """
    Photometric loss - measures brightness consistency between warped image and target.
    This is an unsupervised loss since we don't have ground truth optical flow.
    """
    
    def __init__(self, beta=1.0):
        super().__init__()
        self.beta = beta  # Tune: 0.5-2.0
        
    def forward(self, img1, img2, flow, mask):
        """
        Compute photometric loss.
        
        Args:
        img1: First image (B, 3, H, W)
        img2: Second image (B, 3, H, W)
        flow: Predicted optical flow (B, 2, H, W)
        
        Returns:
        Photometric loss value
        """
        # Warp img1 towards img2 using predicted flow
        img1_warped = self.warp_image(img1, -flow)
        
        # Compute L1 loss between warped image and target
        l1_loss = F.smooth_l1_loss(img1_warped, img2, beta=self.beta, reduction='none')

        ssim_loss = 1 - ssim(img1_warped, img2, data_range=1.0)  

        

        # Broadcast SSIM to channels if needed
        if ssim_loss.dim() == 4 and ssim_loss.size(1) == 1:
            ssim_loss = ssim_loss.repeat(1, img1.size(1), 1, 1)

        loss = (1 - 0.85) * l1_loss + (0.85 / 2) * ssim_loss
        
        loss = loss * mask
        return loss.sum() 
        
    def warp_image(self, img, flow):
        """
        Warp image using optical flow.
        
        Args:
        img: Image to warp (B, C, H, W)
        flow: Optical flow (B, 2, H, W)
        
        Returns:
        Warped image (B, C, H, W)
        """
        B, C, H, W = img.size()
        
        # Create mesh grid
        xx = torch.arange(0, W).view(1, -1).repeat(H, 1)
        yy = torch.arange(0, H).view(-1, 1).repeat(1, W)
        xx = xx.view(1, 1, H, W).repeat(B, 1, 1, 1)
        yy = yy.view(1, 1, H, W).repeat(B, 1, 1, 1)
        grid = torch.cat((xx, yy), 1).float()
        
        if img.is_cuda:
            grid = grid.cuda()
        
        # Add flow to grid
        vgrid = grid + flow
        
        # Scale to [-1, 1] for grid_sample
        vgrid[:, 0, :, :] = 2.0 * vgrid[:, 0, :, :] / max(W - 1, 1) - 1.0
        vgrid[:, 1, :, :] = 2.0 * vgrid[:, 1, :, :] / max(H - 1, 1) - 1.0
        
        vgrid = vgrid.permute(0, 2, 3, 1)
        
        # Warp image
        output = F.grid_sample(img, vgrid, align_corners=True)
        
        return output


class SupervisedFlowLoss(nn.Module):
    """
    Supervised optical flow loss with confidence weighting.
    Uses pseudo-labels from Farnebäck with confidence-based masking.
    """
    
    def __init__(self, use_confidence_weighting=True, beta=1.0):
        super().__init__()
        self.use_confidence_weighting = use_confidence_weighting
        self.beta = beta  # Tune: 0.5-2.0
    
    def forward(self, flow_pred, flow_pseudo, mask):
        """
        Compute supervised loss with optional confidence weighting.
        
        Args:
            flow_pred: Predicted flow (B, 2, H, W)
            flow_pseudo: Pseudo-label flow from Farnebäck (B, 2, H, W)
            confidence_mask: Confidence weights (B, H, W), optional
            valid_mask: Binary mask of valid regions (B, H, W), optional
        
        Returns:
            Weighted loss value
        """

        mag_fb = torch.sqrt(flow_pseudo[:,0]**2 + flow_pseudo[:,1]**2)
        mag_raft = torch.sqrt(flow_pred[:,0]**2 + flow_pred[:,1]**2)
        diff = mag_fb - mag_raft
        mean_diff = abs(diff).mean()

        mask_pseudo = (abs(diff) < 0.2).unsqueeze(1).float()
        mask_photo  = (abs(diff) >= 0.2).unsqueeze(1).float()
        
        loss = F.smooth_l1_loss(flow_pred, flow_pseudo, beta=self.beta, 
                               reduction='none')  # B,2,H,W
        
        masked_loss = loss * mask_pseudo

        loss = masked_loss.sum() 

        return loss, mask_photo, mask_pseudo  # Auto-normalize

class HybridFlowLoss(nn.Module):
    """
    Hybrid loss combining photometric (unsupervised) and supervised components.
    """
    
    def __init__(self, model, num_pairs):
        super().__init__()
        self.model = model
        self.num_pairs = num_pairs

        self.photo_loss = PhotometricLoss()
        self.supervised_loss = SupervisedFlowLoss(use_confidence_weighting=True)
    
    def forward(self, flow_pred, img1, img2, flow_pseudo, 
                sup_mask):
        """
        Compute hybrid loss.
        
        Args:
            flow_pred: Predicted flow (B, 2, H, W)
            img1: First image (B, 3, H, W)
            img2: Second image (B, 3, H, W)
            flow_pseudo: Pseudo-label flow (B, 2, H, W)
            confidence_mask: Confidence weights (B, H, W)
            valid_mask: Valid regions (B, H, W)
        
        Returns:
            Weighted combination of losses
        """        

        
        
        # Supervised loss on high-confidence regions
        loss_sup, mask_photo, mask_pseudo = self.supervised_loss(
            flow_pred, flow_pseudo,
            sup_mask
        )

        # Photometric loss (unsupervised, for clear-sky regions)
        loss_photo = self.photo_loss(img1, img2, flow_pred, mask_photo)

        if self.num_pairs == 300:
            loss_sup = 1 * loss_sup   #Tune weights
            loss_photo = 1 * loss_photo

        elif self.num_pairs == 600:
            loss_sup = 1 * loss_sup   #Tune weights
            loss_photo = 1 * loss_photo
            
        elif self.num_pairs == 900:
            loss_sup = 1 * loss_sup   #Tune weights
            loss_photo = 1 * loss_photo

        elif self.num_pairs == 2000:
            loss_sup = 1 * loss_sup     #Tune weights
            loss_photo = 1 * loss_photo

        # Combine losses
        total_loss = (
            loss_sup + loss_photo
        )

        
        return total_loss, {
            'loss_supervised': loss_sup.item(),
            'loss_photo': loss_photo.item()
        }

    