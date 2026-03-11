"""
RAFT Optical Flow Inference Pipeline with Confidence Filtering
Main inference engine: loads model, processes image pairs, saves results.
Author: Jonathan Huth
Date: January 2026
"""

import os
import sys
import csv
import gc
import torch
import scipy.io
import argparse
import time
import psutil  # pip install psutil (for RAM)
import GPUtil  # GPU utilization
from contextlib import contextmanager
import numpy as np
from pathlib import Path
from ezflow.models import build_model
from python_scripts.raft_config import InferenceConfig
from python_scripts.raft_utils import load_image, preprocess_images, flow_to_xy_components

def run_inference(checkpoint_path, image_dir, output_dir, device,
                  save_flow=True, save_csv=True):
    """
    Run RAFT inference on all image pairs with optional confidence filtering.
    
    Args:
        checkpoint_path: Path to RAFT checkpoint (.pt file)
        image_dir: Directory containing ASI images
        output_dir: Directory to save flow fields (.mat)
        device: torch device (cuda or cpu)
        save_flow: Whether to save flow fields
        save_csv: Whether to save CSV summary
        confidence_threshold: Minimum flow magnitude threshold (pixels/frame)
        apply_filtering: Whether to apply confidence-based filtering
    """
    cfg = InferenceConfig
    
    # Create output directories
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    print(f"\n{'='*80}")
    print(f"RAFT Optical Flow Inference")
    print(f"{'='*80}\n")
    print(f"[INFO] Device: {device}")
    print(f"[INFO] Checkpoint: {checkpoint_path}")
    print(f"[INFO] Image directory: {image_dir}")
    print(f"[INFO] Output directory: {output_dir}")
    
    # Load model
    print("[INFO] Loading RAFT model...")
    model = build_model("RAFT", default=True).to(device)
    model.eval()
    
    # Load checkpoint if it exists
    if checkpoint_path and os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
        
        # Check if it's a training checkpoint
        if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
            print(f"[INFO] Loaded fine-tuned checkpoint (epoch {checkpoint.get('epoch', 'unknown')})")
        else:
            # Raw state dict
            model.load_state_dict(checkpoint)
            print(f"[INFO] Loaded checkpoint")
    else:
        print(f"[INFO] Using pretrained Kubric weights")
    
    # Load images
    print("\n[INFO] Loading image pairs...")
    image_dir = Path(image_dir)
    image_files = sorted(image_dir.glob("*.jpg"))
    
    if not image_files:
        print(f"ERROR: No JPG files found in {image_dir}")
        sys.exit(1)
    
    n_images = len(image_files)
    n_pairs = n_images // 2
    print(f"[INFO] Found {n_images} images ({n_pairs} consecutive pairs)\n")
    
    # Inference loop
    results = []
    inference_times = []
    total_inference_time = 0
    peak_gpu_mem = 0.0
    ram_gb = 0.0

    with torch.no_grad():
        for pair_idx in range(0, n_images - 1, 2):  # 0, 2, 4, ...
            if pair_idx + 1 >= n_images:  # Safety check
                break
            
            timestamp = image_files[pair_idx].stem.split('_')[0]
            
            # Load and preprocess images
            img1 = load_image(image_files[pair_idx], cfg.IMG_H, cfg.IMG_W)
            img2 = load_image(image_files[pair_idx + 1], cfg.IMG_H, cfg.IMG_W)
            
            img1_t, img2_t = preprocess_images(img1, img2)
            img1_t = img1_t.to(device)
            img2_t = img2_t.to(device)

            # TIME FORWARD PASS ONLY
            torch.cuda.synchronize()  # GPU sync before
            start_time = time.time()

            outputs = model(img1_t, img2_t)
            flow_pred = outputs['flow_preds'][-1]  # Final iteration (most accurate)
            flow_np = flow_pred[0].permute(1, 2, 0).cpu().numpy()  # (H, W, 2)

            torch.cuda.synchronize()  # GPU sync after
            inference_time = time.time() - start_time
            inference_times.append(inference_time)

            if torch.cuda.is_available():
                peak_gpu_mem = max(peak_gpu_mem, torch.cuda.max_memory_allocated() / 1e9)
            
            ram_gb = max(ram_gb, psutil.Process().memory_info().rss / 1e9)

            # Extract u, v components
            Vx, Vy = flow_to_xy_components(flow_np)
            
            # Compute statistics
            mag = np.sqrt(Vx**2 + Vy**2)
            
            # Only compute stats on non-zero (valid) pixels
            valid_pixels = mag > 0
            if valid_pixels.sum() > 0:
                mean_mag = np.mean(mag[valid_pixels])
                max_mag = np.max(mag[valid_pixels])
            else:
                mean_mag = 0.0
                max_mag = 0.0
            
            # Save flow field (.mat format for MATLAB compatibility)
            if save_flow:
                checkpoints = f"{checkpoint_path}".split('/')[1]
                checkpoints = checkpoints.split('.')[0]
                flow_path = f"{output_dir}/{checkpoints}/flowRaftPy_{timestamp}"
                Path(flow_path).mkdir(parents=True, exist_ok=True)
                flow_path = os.path.join(flow_path, f"flowRaftPy_{timestamp}.mat")
                
                flow_dict = {
                    'Vx': flow_np[:, :, 0], 
                    'Vy': flow_np[:, :, 1],
                }
                scipy.io.savemat(flow_path, flow_dict)
            
            # Record result
            results.append({
                'pair_idx': pair_idx // 2 + 1,
                'image1': str(image_files[pair_idx].name),
                'image2': str(image_files[pair_idx + 1].name),
                'mean_magnitude': mean_mag,
                'max_magnitude': max_mag,
                'x_mean': np.mean(Vx[valid_pixels]) if valid_pixels.sum() > 0 else 0.0,
                'y_mean': np.mean(Vy[valid_pixels]) if valid_pixels.sum() > 0 else 0.0,
                'inference_time_s': inference_time
            })
            
            if (pair_idx + 1) % 10 == 0 or pair_idx == 0:
                print(f"[{pair_idx+1:4d}/{n_pairs}] Mean flow: {mean_mag:.4f}, "
                      f"Max: {max_mag:.4f}")
    
            # # Save CSV summary
            # if save_csv and results:
            #     flow_path = f"{output_dir}/{checkpoints}"
            #     csv_path = os.path.join(flow_path, "inference_summary.csv")
            #     with open(csv_path, 'w', newline='') as f:
            #         writer = csv.DictWriter(f, fieldnames=results[0].keys())
            #         writer.writeheader()
            #         writer.writerows(results)
            #     print(f"\n[SAVED] CSV summary: {csv_path}")
            # else:
            #     csv_path = None

    mean_time = np.mean(inference_times)
    fps = n_pairs / sum(inference_times)
    print(f'\n=== COMPUTATIONAL METRICS ===')
    print(f'Mean inference time/pair: {mean_time:.3f}s')
    print(f'Total time for {n_pairs} pairs: {sum(inference_times):.1f}s')
    print(f'Mean FPS: {fps:.1f}')
    print(f'Peak GPU memory: {peak_gpu_mem:.1f} GB')
    print(f'Peak RAM usage: {ram_gb:.1f} GB')

    print(f"\n{'='*80}")
    print(f"INFERENCE COMPLETE")
    print(f"{'='*80}\n")
    print(f"[OUTPUT] Flow fields: {output_dir}/")
    # if csv_path is not None:
    #     print(f"[OUTPUT] Summary: {csv_path}")
    print()
    # At END of every script:
    del model  # Explicit del large objects
    torch.cuda.empty_cache()          # Free PyTorch cache
    gc.collect()                      # Python GC
    print(f"GPU allocated: {torch.cuda.memory_allocated()/1e9:.1f} GB")
    
    timing_results = {
        'mean_fps': float(fps),
        'mean_time_per_pair': float(mean_time),
        'total_time': float(sum(inference_times)),
        'peak_gpu_gb': float(peak_gpu_mem),
        'peak_ram_gb': float(ram_gb),
        'num_pairs': int(n_pairs),
        'inference_times': inference_times  # list of per-pair times
    }
    return timing_results

def main():
    cfg = InferenceConfig
    
    parser = argparse.ArgumentParser(
        description="RAFT Inference"
    )
    
    parser.add_argument(
        "--checkpoints",
        type=str,
        default=cfg.CHECKPOINT,
        help="Checkpoints for RAFT model"
    )

    parser.add_argument(
        "--images",
        type=str,
        default=cfg.DATA_DIR,
        help="Images for RAFT inference"
    )
    
    output_dir = cfg.OUTPUT_DIR

    # Parse arguments
    args = parser.parse_args()
    
    if args.images != cfg.DATA_DIR:
        output_dir = 'results/synFlow/flowRaftPy'

    timing_results = run_inference(
        checkpoint_path=args.checkpoints,
        image_dir=args.images,
        output_dir=output_dir,
        device=cfg.DEVICE,
        save_flow=cfg.SAVE_FLOW,
        save_csv=cfg.SAVE_CSV,
    )
    return timing_results

if __name__ == "__main__":
    timing_results = main()

    