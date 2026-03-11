"""
RAFT Fine-Tuning on ASI Data - Main Training Script
Progressive training with configurable dataset sizes.
"""

import argparse
import sys
from pathlib import Path

# Add python_scripts to path

from python_scripts.config_finetuning import Config
from python_scripts.dataloader_finetuning import get_dataloader
from python_scripts.losses_finetuning import PhotometricLoss, SmoothnessLoss, ClearSkyAwareLoss
from python_scripts.utils_finetuning import train_one_epoch, save_checkpoint

import torch
from ezflow.models import build_model


def main():
    """Main training function."""
    
    # Parse command-line arguments
    parser = argparse.ArgumentParser(
        description="RAFT Fine-Tuning on ASI Data - Progressive Training"
    )
    parser.add_argument(
        "--stage",
        type=int,
        choices=[300, 600, 1200, 2000],
        default=300,
        help="Training stage (number of image pairs): 300, 600, 1200, or 2000"
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=None,
        help=f"Number of epochs (default: {Config.NUM_EPOCHS} from config)"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=None,
        help=f"Batch size (default: {Config.BATCH_SIZE} from config)"
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=None,
        help=f"Learning rate (default: {Config.LEARNING_RATE} from config)"
    )
    
    args = parser.parse_args()
    
    # Override config if arguments provided
    num_pairs = args.stage
    num_epochs = args.epochs if args.epochs is not None else Config.NUM_EPOCHS
    batch_size = args.batch_size if args.batch_size is not None else Config.BATCH_SIZE
    learning_rate = args.lr if args.lr is not None else Config.LEARNING_RATE
    
    print("\n" + "="*70)
    print(f"RAFT FINE-TUNING - STAGE {num_pairs} PAIRS")
    print("="*70)
    print(f"Training pairs: {num_pairs}")
    print(f"Epochs: {num_epochs}")
    print(f"Batch size: {batch_size}")
    print(f"Learning rate: {learning_rate}")
    print(f"Device: {Config.DEVICE}")
    print("="*70 + "\n")
    
    # Validate paths and create directories
    try:
        Config.validate_paths()
        Config.create_directories()
    except AssertionError as e:
        print(f"\n❌ ERROR: {e}")
        print("\nPlease ensure:")
        print(f"  1. Pretrained checkpoint exists: {Config.PRETRAINED_RAFT}")
        print(f"  2. Data directory exists: {Config.get_data_dir(num_pairs)}")
        sys.exit(1)
    
    # Get device
    device = torch.device(Config.DEVICE if torch.cuda.is_available() else "cpu")
    print(f"✓ Using device: {device}\n")
    
     # Create dataloader
    print(f"Loading data from: {Config.get_data_dir(num_pairs)}")
    try:
        dataloader = get_dataloader(
            config=Config,
            num_pairs=num_pairs
        )
        print(f"✓ DataLoader created with {len(dataloader)} batches per epoch\n")
    except Exception as e:
        print(f"\n❌ ERROR creating dataloader: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
    
    # Load RAFT model
    print("Loading RAFT model...")
    try:
        model = build_model("RAFT", default=True)
        
        # Load checkpoint
        checkpoint_path = Config.get_checkpoint_to_load(num_pairs)
        print(f"Loading checkpoint: {checkpoint_path}")
        
        if checkpoint_path.exists():
            state_dict = torch.load(checkpoint_path, map_location=device)
            model.load_state_dict(state_dict)
            print(f"✓ Loaded checkpoint: {checkpoint_path.name}\n")
        else:
            print(f"⚠️  WARNING: Checkpoint not found: {checkpoint_path}")
            print("   Using randomly initialized weights\n")
        
        model = model.to(device)
        model.train()
        
    except Exception as e:
        print(f"\n❌ ERROR loading model: {e}")
        sys.exit(1)
    
    # Setup optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=learning_rate,
        weight_decay=Config.WEIGHT_DECAY,
        eps=Config.EPSILON
    )
    
    # Setup learning rate scheduler
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=num_epochs
    )
    
    # Setup loss functions
    loss_fn = ClearSkyAwareLoss()
    
    # Training loop
    print("\n" + "="*70)
    print("STARTING TRAINING")
    print("="*70 + "\n")
    
    best_loss = float('inf')
    patience_counter = 0
    
    for epoch in range(1, num_epochs + 1):
        print(f"\nEpoch {epoch}/{num_epochs}")
        print("-" * 50)
        
        # Train one epoch
        avg_loss, loss_components = train_one_epoch(
            model=model,
            dataloader=dataloader,
            optimizer=optimizer,
            loss_fn=loss_fn,
            device=device,
            epoch=epoch,
            log_interval=Config.LOG_INTERVAL
        )
        
        # Update learning rate
        scheduler.step()
        current_lr = optimizer.param_groups[0]['lr']
        
        print(f"\nEpoch {epoch} Summary:")
        print(f"  Total Loss: {avg_loss:.4f}")
        print(f"  Photometric Loss: {loss_components['photo']:.4f}")
        print(f"  Smoothness Loss: {loss_components['smooth']:.4f}")
        print(f"  Learning Rate: {current_lr:.6f}")
        
        # Save checkpoint if improved
        if avg_loss < best_loss:
            best_loss = avg_loss
            patience_counter = 0
            
            # Save checkpoint
            results_dir = Config.get_results_dir(num_pairs)
            checkpoint_name = f"raft_finetune_{num_pairs}.pth"
            save_checkpoint(
                model=model,
                optimizer=optimizer,
                epoch=epoch,
                loss=avg_loss,
                save_dir=results_dir,
                checkpoint_name=checkpoint_name
            )
            print(f"  ✓ Saved best checkpoint (loss improved)")
        else:
            patience_counter += 1
            print(f"  No improvement ({patience_counter}/{Config.PATIENCE})")
        
        # Early stopping
        if patience_counter >= Config.PATIENCE:
            print(f"\n⚠️  Early stopping triggered after {epoch} epochs")
            print(f"   No improvement for {Config.PATIENCE} consecutive epochs")
            break
    
    # Training complete
    print("\n" + "="*70)
    print("TRAINING COMPLETE")
    print("="*70)
    print(f"Best loss: {best_loss:.4f}")
    print(f"Final checkpoint saved to: {Config.get_results_dir(num_pairs)}")
    print(f"Checkpoint name: raft_finetune_{num_pairs}.pth")
    print("="*70 + "\n")


if __name__ == "__main__":
    main()