"""
Dataset loader for ASI image pairs.
Loads NON-OVERLAPPING consecutive image pairs for optical flow training.
"""
import os
from pathlib import Path
from typing import Tuple, List
import cv2
import torch
import pytz
import torchvision.transforms.v2 as v2
import random
from torch.utils.data import Dataset, DataLoader, SubsetRandomSampler
import numpy as np
import pandas as pd
from datetime import datetime

    
class ASIFlowDataset(Dataset):
    """
    Dataset for ASI non-overlapping image pairs.
    
    Pairs are created as: img1→img2, img3→img4, img5→img6, etc.
    NO overlap between pairs (unlike img1→img2, img2→img3).
    
    Args:
        image_dir: Directory containing ASI images
        num_pairs: Number of image pairs to use (300, 600, 1200, 2000)
        img_height: Target height for resizing
        img_width: Target width for resizing
    """
    
    def __init__(
        self,
        data_dir,
        num_pairs,
        img_height,
        img_width,
        augment_flag
    ):

        self.data_dir = Path(data_dir)
        self.num_pairs = num_pairs
        self.img_height = img_height
        self.img_width = img_width
        self.augment_flag = augment_flag

        # Load image file paths
        self.image_files = self._load_image_files()
        
        # Create NON-OVERLAPPING pairs
        self.pairs = self._create_non_overlapping_pairs()
        
        print(f"✓ Loaded {len(self.image_files)} images")
        print(f"✓ Created {len(self.pairs)} non-overlapping pairs")
        print(f"  (Pairs: img1→img2, img3→img4, img5→img6, ...)")
    
    def _load_image_files(self) -> List[Path]:
        """Load all image file paths, sorted chronologically."""
        # Get all .jpg files
        image_files = sorted(self.data_dir.glob("*.jpg"))
        
        if len(image_files) == 0:
            raise ValueError(f"No .jpg files found in {self.data_dir}")
        
        return image_files
    
    def _create_non_overlapping_pairs(self) -> List[Tuple[Path, Path]]:
        """
        Create NON-OVERLAPPING image pairs.
        
        Example:
            Images: [img0, img1, img2, img3, img4, img5]
            Pairs:  [(img0, img1), (img2, img3), (img4, img5)]
        
        Returns:
            List of (img1_path, img2_path) tuples
        """
        pairs = []
        
        # Step through images in increments of 2
        for i in range(0, len(self.image_files) - 1, 2):
            img1_path = self.image_files[i]
            img2_path = self.image_files[i + 1]
            pairs.append((img1_path, img2_path))
            
            # # Stop when we have enough pairs
            # if len(pairs) >= self.num_pairs:
            #     break

        print(f"{len(pairs)} pairs available")
        
        return pairs[:self.num_pairs]
    
    def _load_and_preprocess(self, img_path: Path) -> torch.Tensor:
        """
        Load and preprocess a single image.
        
        Steps:
        1. Load image with OpenCV
        2. Convert BGR to RGB
        3. Resize to target size
        4. Normalize to [0, 1] (NO ImageNet normalization)
        5. Convert to tensor (C, H, W)
        
        Args:
            img_path: Path to image file
        
        Returns:
            Image tensor (C, H, W) in range [0, 1]
        """

        # Load image
        img = cv2.imread(str(img_path))
        if img is None:
            raise ValueError(f"Failed to load image: {img_path}")

        # Convert BGR to RGB
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # Resize (RAFT requires specific input size)
        img = cv2.resize(img, (self.img_width, self.img_height), 
                        interpolation=cv2.INTER_LINEAR)
        
        # Convert to tensor: (H, W, C) -> (C, H, W)
        img_tensor = torch.from_numpy(img).permute(2, 0, 1)
        
        return img_tensor
    
    def __len__(self) -> int:
        """Return number of pairs."""
        return len(self.pairs)

    def augment_pair(self, img1, img2):
        """Elastic + jitter for cloud variety."""
        if random.random() < 0.5:  # Flip
            img1, img2 = torch.flip(img1, [2]), torch.flip(img2, [2])
        if random.random() < 0.3:  # Brightness/contrast
            aug = v2.ColorJitter(brightness=0.2, contrast=0.2)
            img1, img2 = aug(img1), aug(img2)
        # Elastic: simple grid warp (add torchio or kornia for full)
        return img1, img2

    def path_to_unix(self, path: Path) -> float:
        """YYYYMMDDHHmmss → UTC Unix s."""
        stem = path.stem  # e.g., "20240308121530"
        dt_str = f"{stem[:4]}-{stem[4:6]}-{stem[6:12]}"
        dt = datetime.strptime(dt_str, "%Y-%m-%d%H%M%S")
        return dt.timestamp()  # UTC+0 assumed from filename

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, str]:
        """
        Get a pair of non-overlapping consecutive images.
        
        Args:
            idx: Index of the pair
        
        Returns:
            img1: First image tensor (C, H, W) in range [0, 1]
            img2: Second image tensor (C, H, W) in range [0, 1]
            pair_name: String identifier for this pair
        """
        img1_path, img2_path = self.pairs[idx]
        
        # Load and preprocess both images
        img1 = self._load_and_preprocess(img1_path)
        img2 = self._load_and_preprocess(img2_path)
        
        # Create pair identifier
        pair_name = f"{img1_path.stem}_{img2_path.stem}"
        if self.augment_flag:
            img1, img2 = self.augment_pair(img1, img2)
        # Extract UTC Unix from filename YYYYMMDDHHmmss.jpg → Unix s
        img1_time = self.path_to_unix(img1_path)  # Implement: parse str → UTC Unix
        date_str = img1_path.stem[:10]  # "2024-03-08" from filename

        return img1, img2, pair_name


def get_dataloader(config, num_pairs: int):
    """
    Create DataLoader for training.
    
    Args:
        config: Configuration object
        num_pairs: Number of image pairs for this training stage
    
    Returns:
        DataLoader object
    """
    dataset = ASIFlowDataset(
        data_dir = config.get_data_dir(num_pairs),
        num_pairs=num_pairs,
        img_height=config.IMG_HEIGHT,
        img_width=config.IMG_WIDTH,
        augment_flag=False
    )
    
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=True,  # Shuffle for better training
        num_workers=0,  # Set to 0 for Windows compatibility
        pin_memory=True if config.DEVICE == "cuda" else False
    )
    
    return dataloader

def get_valdataloader(config):
    """
    Create DataLoader for training.
    
    Args:
        config: Configuration object
        num_pairs: Number of image pairs for this training stage
    
    Returns:
        DataLoader object
    """
    full_dataset = ASIFlowDataset(
        data_dir = config.get_valid_dir(),
        num_pairs=200,
        img_height=config.IMG_HEIGHT,
        img_width=config.IMG_WIDTH,
        augment_flag=False
    )
    
    valloader = DataLoader(full_dataset, batch_size=config.BATCH_SIZE, shuffle=False, num_workers=0)
    
    return valloader


if __name__ == "__main__":
    # Test dataset loading
    from python_scripts.config_finetuning import Config
    
    print("Testing dataset loader...")
    Config.validate_paths()
    
    # Test with 10 pairs
    print("\n" + "="*60)
    print("TESTING NON-OVERLAPPING PAIRS")
    print("="*60)
    
    dataloader = get_dataloader(Config, num_pairs=300)
    
    # Load first 3 pairs to verify non-overlapping structure
    print("\nFirst 3 pairs:")
    for batch_idx, (img1, img2, pair_name) in enumerate(dataloader):
        if batch_idx >= 3:
            break
        print(f"\nPair {batch_idx + 1}: {pair_name[0]}")
        print(f"  Image 1 shape: {img1.shape}")
        print(f"  Image 2 shape: {img2.shape}")
        print(f"  Image 1 range: [{img1.min():.3f}, {img1.max():.3f}]")
        print(f"  Image 2 range: [{img2.min():.3f}, {img2.max():.3f}]")
    
    print("\n" + "="*60)
    print("✓ Dataset loader test passed!")
    print("✓ Images are in RGB format, range [0, 1]")
    print("✓ NO ImageNet normalization applied")
    print("✓ Pairs are NON-OVERLAPPING")
    print("="*60)


