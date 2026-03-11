"""
Utility functions for RAFT fine-tuning training loop.
Includes training iteration and checkpoint management.
"""

import torch
from pathlib import Path
import numpy as np
import cv2
import torchvision.transforms.v2 as v2

from python_scripts.config_finetuning import Config
from python_scripts.pseudo_labels import generate_pseudo_labels
from python_scripts.adaptive_loss import ClearSkyAwareLoss 

"""
Utility functions for RAFT fine-tuning training loop.
Includes training iteration and checkpoint management.
"""

import torch
from pathlib import Path

def train_one_epoch_with_pseudolabels(
    model,
    dataloader,
    optimizer,
    loss_fn,
    device,
    epoch,
    confidence_threshold=0.7,
    log_interval=10
):
    """
    Train model for one epoch with pseudo-label supervision from Farnebäck.
    
    Args:
        model: RAFT model
        dataloader: Training dataloader
        optimizer: Optimizer
        loss_fn: HybridFlowLoss function
        device: torch device
        epoch: Current epoch number
        confidence_threshold: Minimum confidence for pseudo-labels
        log_interval: How often to log progress
    
    Returns:
        (avg_loss, loss_components_dict)
    """

    total_loss = 0.0
    loss_components_sum = {'photo': 0.0, 'supervised': 0.0}
    num_batches = len(dataloader)
    
    for batch_idx, batch_data in enumerate(dataloader):
        # Parse batch data
        if isinstance(batch_data, (tuple, list)):
            img1 = batch_data[0]
            img2 = batch_data[1]
        else:
            img1 = batch_data['img1']
            img2 = batch_data['img2']
        
        # Move to device
        img1 = img1.to(device)
        img2 = img2.to(device)
        
        # Generate pseudo-labels from Farnebäck
        # Convert back to numpy for Farnebäck
        img1_np = img1.cpu().numpy() 
        img2_np = img2.cpu().numpy() 

        toImg = v2.ToImage()
        img1 = toImg(img1/255.0)
        img2 = toImg(img2/255.0)

        toDtype = v2.ToDtype(torch.float32, scale=True)
        img1 = toDtype(img1)
        img2 = toDtype(img2)
        
        mu1 = img1.mean() 
        sigma1 = img1.std()
        normalize = v2.Normalize(mean=[mu1, mu1, mu1], std=[sigma1, sigma1, sigma1])
        img1 = normalize(img1) 

        mu2 = img2.mean() 
        sigma2 = img2.std()
        normalize = v2.Normalize(mean=[mu2, mu2, mu2], std=[sigma2, sigma2, sigma2])
        img2 = normalize(img2)

        # Transpose to (B, H, W, 3) for opencv
        img1_np = img1_np.transpose(0, 2, 3, 1)
        img2_np = img2_np.transpose(0, 2, 3, 1)
        
        # Generate pseudo-labels batch-wise
        B, H, W, _ = img1_np.shape
        flow_pseudo_list = []
        confidence_list = []
        valid_list = []
        
        for b in range(B):
            flow, conf, valid = generate_pseudo_labels(
                img1_np[b], img2_np[b],
                confidence_threshold=confidence_threshold
            )

            flow_pseudo_list.append(flow)
            confidence_list.append(conf)
            valid_list.append(valid)
        
        flow_pseudo = torch.stack(flow_pseudo_list).to(device)  # (B, 2, H, W)
        confidence_mask = torch.stack(confidence_list).to(device)
        valid_mask = torch.stack(valid_list).to(device)  # (B, H, W)
        sup_mask = (confidence_mask > confidence_threshold).float()
        sup_mask *= confidence_mask
        # Zero gradients
        optimizer.zero_grad()
        
        # Forward pass
        flow_predictions = model(img1, img2)
        flow_pred = flow_predictions['flow_preds'][-1]  # (B, 2, H, W)

        # Compute hybrid loss
        loss, loss_dict = loss_fn(
            flow_pred=flow_pred,
            img1=img1,
            img2=img2,
            flow_pseudo=flow_pseudo,
            sup_mask=sup_mask
        )
        
        # Backward pass
        loss.backward()
        optimizer.step()
        
        # Accumulate losses
        total_loss += loss.item()
        loss_components_sum['photo'] += loss_dict['loss_photo']
        loss_components_sum['supervised'] += loss_dict['loss_supervised']
        
        # Log progress
        if (batch_idx + 1) % log_interval == 0:
            avg_loss_so_far = total_loss / (batch_idx + 1)
            print(f"  Batch [{batch_idx + 1}/{num_batches}] - Loss: {loss.item():.4f} (Avg: {avg_loss_so_far:.4f})")
    
    # Return average losses
    avg_loss = total_loss / num_batches
    avg_components = {
        'photo': loss_components_sum['photo'] / num_batches,
        'supervised': loss_components_sum['supervised'] / num_batches
    }
    
    return avg_loss, avg_components


def save_checkpoint(
    model,
    optimizer,
    epoch,
    loss,  # Add to dict
    save_dir,
    checkpoint_name="checkpoint.pth"
):
    """
    Save model checkpoint.
    
    Args:
        model: Model to save
        optimizer: Optimizer state
        epoch: Current epoch
        loss: Current loss
        save_dir: Directory to save checkpoint
        checkpoint_name: Name of checkpoint file
    """
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    
    checkpoint_path = save_dir / checkpoint_name
    
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
    }, checkpoint_path)
    
    print(f"  ✓ Checkpoint saved: {checkpoint_path}")

def save_eval_checkpoint(
    model,
    optimizer,
    epoch,
    loss,  # Add to dict
    save_dir,
    fbce,
    proxy_epe,
    checkpoint_name="checkpoint.pth"
):
    """
    Save model checkpoint.
    
    Args:
        model: Model to save
        optimizer: Optimizer state
        epoch: Current epoch
        loss: Current loss
        save_dir: Directory to save checkpoint
        checkpoint_name: Name of checkpoint file
    """
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    
    checkpoint_path = save_dir / checkpoint_name
    
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
        'fbce': fbce,
        'proxy_epe': proxy_epe,
    }, checkpoint_path)
    
    print(f"  ✓ Checkpoint saved: {checkpoint_path}")