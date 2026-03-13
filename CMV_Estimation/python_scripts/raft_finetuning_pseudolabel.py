"""
RAFT Fine-Tuning with Pseudo-Label Supervision
Progressive training with confidence-weighted Farnebäck pseudo-labels
"""

import argparse
import sys
from pathlib import Path
import gc

from python_scripts.evaluation_finetuning import validate
from python_scripts.config_finetuning import Config
from python_scripts.dataloader_finetuning import get_dataloader, get_valdataloader
from python_scripts.losses_finetuning import HybridFlowLoss
from python_scripts.pseudo_labels import generate_pseudo_labels
from python_scripts.utils_finetuning import train_one_epoch_with_pseudolabels, save_checkpoint, save_eval_checkpoint
from python_scripts.adaptive_loss import ClearSkyAwareLoss

from torch.utils.data import DataLoader, SubsetRandomSampler
import torch
import numpy as np
import cv2
from ezflow.models import build_model


def main():
    """Main training function with pseudo-label supervision."""
    
    # Parse command-line arguments
    parser = argparse.ArgumentParser(
        description="RAFT Fine-Tuning on ASI Data with Pseudo-Labels"
    )
    parser.add_argument(
        "--stage",
        type=int,
        choices=[300, 600, 900, 2000],
        default=2000,
        help="Training stage (number of image pairs): 300, 600, 900, or 2000"
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=Config.NUM_EPOCHS,
        help=f"Number of epochs (default: {Config.NUM_EPOCHS} from config)"
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=Config.LEARNING_RATE,
        help=f"Learning rate (default: {Config.LEARNING_RATE} from config)"
    )
    
    args = parser.parse_args()
    
    # Override config if arguments provided
    num_pairs = args.stage
    num_epochs = args.epochs
    learning_rate = args.lr 

    print("\n" + "="*70)
    print(f"RAFT FINE-TUNING WITH PSEUDO-LABELS - STAGE {num_pairs} PAIRS")
    print("="*70)
    print(f"Training pairs: {num_pairs}")
    print(f"Epochs: {num_epochs}")
    print(f"Learning rate: {learning_rate}")
    print(f"Device: {Config.DEVICE}")
    print("="*70 + "\n")
    
    # Validate paths and create directories
    try:
        Config.validate_paths()
        Config.create_directories()
    except AssertionError as e:
        print(f"\n ERROR: {e}")
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
        print(f" DataLoader created with {len(dataloader)} batches per epoch\n")
    except Exception as e:
        print(f"\n ERROR creating dataloader: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

    # Create valloader
    try:
        valloader = get_valdataloader(
            config=Config
        )
        print(f" ValDataLoader created with {len(valloader)}\n")
    except Exception as e:
        print(f"\n ERROR creating valloader: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

    # Load RAFT model
    print("Loading RAFT model...")
    try:
        model = build_model("RAFT", default=True)
        
        # Load checkpoint (start from pretrained Kubric or previous stage)
        checkpoint_path = Config.get_checkpoint_to_load(num_pairs)
        print(f"Loading checkpoint: {checkpoint_path}")
        
        if checkpoint_path.exists():
            checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
            if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
                model.load_state_dict(checkpoint['model_state_dict'])
                print(f"✓ Loaded fine-tuned checkpoint (epoch {checkpoint.get('epoch', '?')})\n")
            else:
                model.load_state_dict(checkpoint)
                print(f"✓ Loaded checkpoint\n")
        else:
            print(f"  Using pretrained Kubric weights\n")
        
        model = model.to(device)
        
    except Exception as e:
        print(f"\n ERROR loading model: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

    # Load Kubric RAFT model
    try:
        kubric_model = build_model("RAFT", default=True)
        # Load checkpoint (start from pretrained Kubric or previous stage)
        checkpoint_path = Config.PRETRAINED_RAFT
        
        if checkpoint_path.exists():
            checkpoint = torch.load(checkpoint_path, map_location=device)
            print(f"Loading checkpoint: {checkpoint_path}")
        else:
            print(f" no checkpoints loaded\n")
        
        kubric_model = kubric_model.to(device)

    except Exception as e:
        print(f"\n ERROR loading model: {e}")
        import traceback
        traceback.print_exc()
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
    
    # Setup hybrid loss (photometric + supervised with confidence weighting)
    loss_fn = HybridFlowLoss(model, num_pairs)  # Now passes correct signature!
    
    # Training loop
    print("\n" + "="*70)
    print("STARTING TRAINING WITH PSEUDO-LABEL SUPERVISION")
    print("="*70 + "\n")
    validate.QUIET = True
    metrics = validate(model, valloader, device)
    fbce = metrics['FBCE']
    proxy_epe = metrics['Proxy-EPE']
    syn_epe = metrics['Synth-EPE']
    # outliers = metrics['outliers']

    # print(metrics['outliers'])  # List worst pairs → viz/save
    print(f"Val FBCE: {fbce:.3f} | Proxy-EPE: {proxy_epe:.2f} | Syn-EPE: {syn_epe:.2f}")
    best_loss = float('inf')
    best_fbce = fbce 
    best_proxy = proxy_epe
    patience_counter = 0
    
    for epoch in range(1, num_epochs + 1):
        print(f"\nEpoch {epoch}/{num_epochs}")
        print("-" * 50)
        
        model.train(True)

        # Train one epoch with pseudo-labels
        avg_loss, loss_components = train_one_epoch_with_pseudolabels(
            model=model,
            dataloader=dataloader,
            optimizer=optimizer,
            loss_fn=loss_fn,
            device=device,
            epoch=epoch,
            log_interval=Config.LOG_INTERVAL
        )
        
        model.train(False)

        # Update learning rate
        scheduler.step()
        current_lr = optimizer.param_groups[0]['lr']
        
        print(f"\nEpoch {epoch} Summary:")
        print(f"  Total Loss: {avg_loss:.4f}")
        print(f"  Photometric Loss: {loss_components['photo']:.4f}")
        print(f"  Supervised Loss: {loss_components['supervised']:.4f}")
        
        
        validate.QUIET = True

        metrics = validate(model, valloader, device)
        fbce = metrics['FBCE']
        proxy_epe = metrics['Proxy-EPE']
        syn_epe = metrics['Synth-EPE']
        print(f"Val FBCE: {fbce:.3f} | Proxy-EPE: {proxy_epe:.2f} | Syn-EPE: {syn_epe:.2f}")
        
        if avg_loss < best_loss:
            best_loss = avg_loss
            patience_counter = 0
            
            # Save checkpoint best loss
            results_dir = Config.get_results_dir(num_pairs)
            checkpoint_name = f"raft_finetune_{num_pairs}_best_loss.pth"
            save_eval_checkpoint(
                model=model,
                optimizer=optimizer,
                epoch=epoch,
                loss=avg_loss, # Add to dict
                save_dir=results_dir,
                fbce=fbce,
                proxy_epe=proxy_epe,
                checkpoint_name=checkpoint_name
            )
            
            results_dir = Config.CHECKPOINT_DIR
            checkpoint_name = f"raft_finetune_{num_pairs}.pth"
            save_checkpoint(
                model=model,
                optimizer=optimizer,
                epoch=epoch,
                loss=avg_loss, # Add to dict
                save_dir=results_dir,
                checkpoint_name=checkpoint_name
            )
    
        else:
            patience_counter += 1
            print(f"  No improvement ({patience_counter}/{Config.PATIENCE})")  
            
        # best metric save
        if (fbce + proxy_epe) < (best_proxy + best_fbce):
            best_fbce = fbce
            best_proxy = proxy_epe

            # Save checkpoint
            results_dir = Config.CHECKPOINT_DIR
            checkpoint_name = f"raft_finetune_{num_pairs}_best_metric.pth"
            save_eval_checkpoint(
                model=model,
                optimizer=optimizer,
                epoch=epoch,
                loss=avg_loss, # Add to dict
                save_dir=results_dir,
                fbce=fbce,
                proxy_epe=proxy_epe,
                checkpoint_name=checkpoint_name
            )

        # Optional: early stop if plateau
        if epoch > 50 and fbce > prev_fbce + 0.01:  # No improvement
            print("Early stopping")
            break
        prev_fbce = fbce

        # Early stopping
        if patience_counter >= Config.PATIENCE:
            print(f"\n Early stopping triggered after {epoch} epochs")
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
    # At END of every script:
    del model, optimizer, dataloader  # Explicit del large objects
    torch.cuda.empty_cache()          # Free PyTorch cache
    gc.collect()                      # Python GC
    print(f"GPU allocated: {torch.cuda.memory_allocated()/1e9:.1f} GB")

if __name__ == "__main__":
    main()
