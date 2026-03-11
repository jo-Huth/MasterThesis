"""
Configuration for RAFT Fine-Tuning
Central configuration for all training parameters and paths.
"""
import os
from pathlib import Path

class Config:
    """Training configuration for RAFT fine-tuning on ASI data."""
    
    # ============================================
    # PATHS
    # ============================================
    # Base directory (CMV_Estimation folder)
    BASE_DIR = Path(r"C:\Users\DaZipfe\Documents\GitHub\MasterThesis\CMV_Estimation")
    
    # Data paths - will be set per training stage
    DATA_DIR_BASE = BASE_DIR / "data" 
    IMAGE_DIR_BASE = DATA_DIR_BASE / "Images"

    # Map training size to dataset folder
    IMAGE_DIRS = {
        300: IMAGE_DIR_BASE / "TrainingSet2",
        600: IMAGE_DIR_BASE / "TrainingSet1",      # Assumed folder name
        900: IMAGE_DIR_BASE / "TrainingSet3",     # Assumed folder name
        2000: IMAGE_DIR_BASE / "TrainingSet4"      # Assumed folder name
    }

    VALIDATION_DIR = IMAGE_DIR_BASE / "ValidationSet"

    # Checkpoint paths
    CHECKPOINT_DIR = BASE_DIR / "checkpoints"
    PRETRAINED_RAFT = CHECKPOINT_DIR / "raft_kubric.pth"
    
    # Results paths
    RESULTS_BASE = BASE_DIR / "results" / "flowRaftPy"
    
    # Training stages - checkpoint output directories
    TRAINING_STAGES = {
        300: RESULTS_BASE / "raft_finetune_300",
        600: RESULTS_BASE / "raft_finetune_600",
        900: RESULTS_BASE / "raft_finetune_900",
        2000: RESULTS_BASE / "raft_finetune_2000"
    }

    # Progressive training: which checkpoint to load for each stage
    CHECKPOINT_TO_LOAD = {
        300: PRETRAINED_RAFT,  # Start from Kubric pretrained
        600: CHECKPOINT_DIR  / "raft_finetune_300.pth",
        900: CHECKPOINT_DIR  / "raft_finetune_600.pth",
        2000: CHECKPOINT_DIR / "raft_finetune_900.pth"
    }

    # ============================================
    # TRAINING PARAMETERS
    # ============================================
    # Progressive training sizes (number of image pairs)
    TRAIN_SIZES = [300, 600, 900, 2000]
    
    # Each stage trains on 300 NEW image pairs
    IMAGES_PER_STAGE = 300

    # Model input size (RAFT standard)
    IMG_HEIGHT = 512
    IMG_WIDTH = 512
    
    # Training hyperparameters
    BATCH_SIZE = 1  # Increase if GPU memory allows
    LEARNING_RATE = 0.00001  # Lower for fine-tuning
    NUM_EPOCHS = 10  # Adjust based on convergence
    

    # Optimizer
    WEIGHT_DECAY = 0.0001
    EPSILON = 1e-8
    
    # Early stopping
    PATIENCE = 5  # Stop if no improvement after 10 epochs

    # Device
    DEVICE = "cuda"  # or "cpu" if no GPU  

    # ============================================
    # LOGGING
    # ============================================
    SAVE_INTERVAL = 5  # Save checkpoint every N epochs
    LOG_INTERVAL = 10  # Print loss every N batches
    
    @classmethod
    def get_data_dir(cls, num_pairs: int) -> Path:
        """Get data directory for given training size."""
        if num_pairs not in cls.IMAGE_DIRS:
            raise ValueError(f"No data directory configured for {num_pairs} pairs")
        return cls.IMAGE_DIRS[num_pairs]

    @classmethod
    def get_valid_dir(cls) -> Path:
        """Get data directory for given training size."""
        return cls.VALIDATION_DIR
       
    @classmethod
    def get_results_dir(cls, num_pairs: int) -> Path:
        """Get results directory for given training size."""
        if num_pairs not in cls.TRAINING_STAGES:
            raise ValueError(f"No results directory configured for {num_pairs} pairs")
        return cls.TRAINING_STAGES[num_pairs]
    
    @classmethod
    def get_checkpoint_to_load(cls, num_pairs: int) -> Path:
        """Get checkpoint to load for given training size."""
        if num_pairs not in cls.CHECKPOINT_TO_LOAD:
            raise ValueError(f"No checkpoint configured for {num_pairs} pairs")
        return cls.CHECKPOINT_TO_LOAD[num_pairs]
    
    @classmethod
    def create_directories(cls):
        """Create all necessary directories."""
        # Create results directories for each training stage
        for stage_dir in cls.TRAINING_STAGES.values():
            stage_dir.mkdir(parents=True, exist_ok=True)
            print(f"✓ Created directory: {stage_dir}")
    
    @classmethod
    def get_cloud_csv(cls, date_str: str) -> Path:
        """Get CSV for date (e.g., '2024-03-08.csv')."""
        return cls.CLOUD_HEIGHT_DIR / f"{date_str}.csv"

    @classmethod
    def validate_paths(cls):
        """Validate that required paths exist."""
        # Check pretrained model
        assert cls.PRETRAINED_RAFT.exists(), f"Pretrained checkpoint not found: {cls.PRETRAINED_RAFT}"
        
        # Check data directories for configured stages
        for train_size, data_dir in cls.IMAGE_DIRS.items():
            assert data_dir.exists(), f"Data directory not found: {data_dir}"
            print(f"✓ Data directory for {train_size} pairs: {data_dir}")
        
        print("✓ All required paths validated")
    
    @classmethod
    def print_config(cls):
        """Print current configuration."""
        print("\n" + "="*70)
        print("RAFT FINE-TUNING CONFIGURATION - PROGRESSIVE TRAINING")
        print("="*70)
        print(f"Base directory: {cls.BASE_DIR}")
        print(f"Pretrained model: {cls.PRETRAINED_RAFT}")
        print(f"\nTraining stages (progressive fine-tuning):")
        for size in cls.TRAIN_SIZES:
            data_dir = cls.IMAGE_DIRS[size]
            checkpoint_to_load = cls.CHECKPOINT_TO_LOAD[size]
            results_dir = cls.TRAINING_STAGES[size]
            print(f"\n  Stage {size} pairs:")
            print(f"    - Load checkpoint from: {checkpoint_to_load.name}")
            print(f"    - Train on images from: {data_dir.name}")
            print(f"    - Save results to: {results_dir.name}")
        
        print(f"\nTraining parameters:")
        print(f"  Image size: {cls.IMG_HEIGHT}x{cls.IMG_WIDTH}")
        print(f"  Batch size: {cls.BATCH_SIZE}")
        print(f"  Learning rate: {cls.LEARNING_RATE}")
        print(f"  Epochs per stage: {cls.NUM_EPOCHS}")
        print(f"  Device: {cls.DEVICE}")
        print("="*70 + "\n")

