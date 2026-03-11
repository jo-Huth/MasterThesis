"""
RAFT Inference Configuration
Centralized configuration for RAFT optical flow inference.

Author: Jonathan Huth
Date: January 2026
"""

import torch
from pathlib import Path

class InferenceConfig:
    """Configuration for RAFT inference."""
    
    # Data paths
    DATA_DIR = "data/Images/EvaluationSet"
    CHECKPOINT = 'checkpoints/raft_finetune_300.pth'  # or raft_finetune_2000.pt, etc.
    OUTPUT_DIR = "results/flowRaftPy"
    
    # Model
    IMG_H, IMG_W = 512, 512  # RAFT input size
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    BATCH_SIZE = 1
    
    # Output options
    SAVE_FLOW = True       # Save as .npy
    SAVE_CSV = True        # Save CSV summary
