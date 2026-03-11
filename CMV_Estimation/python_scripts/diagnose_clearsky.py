"""
Diagnostic script to analyze clear-sky regions in ASI images.
Simpler version without command-line arguments - just edit the paths below.
"""

import numpy as np
import cv2
from pathlib import Path
import matplotlib.pyplot as plt


def analyze_clear_sky_detection(img_path, brightness_threshold=120):
    """
    Analyze how well clear-sky detection works on your images.
    
    Args:
        img_path: Path to image file
        brightness_threshold: HSV V threshold (0-255)
    """
    # Load image
    img = cv2.imread(str(img_path))
    if img is None:
        print(f"ERROR: Could not load image: {img_path}")
        return
    
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    # Convert to HSV
    hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
    
    # Apply threshold
    clear_sky_mask = hsv[:, :, 2] < brightness_threshold  # V channel
    cloud_mask = ~clear_sky_mask
    
    # Compute statistics
    clear_sky_fraction = clear_sky_mask.sum() / clear_sky_mask.size
    cloud_fraction = cloud_mask.sum() / cloud_mask.size
    
    print(f"\n{'='*70}")
    print(f"Image: {Path(img_path).name}")
    print(f"{'='*70}")
    print(f"Clear-sky fraction: {clear_sky_fraction*100:.1f}%")
    print(f"Cloud fraction: {cloud_fraction*100:.1f}%")
    print(f"\nV channel (brightness) statistics:")
    print(f"  Min: {hsv[:,:,2].min()}")
    print(f"  Mean: {hsv[:,:,2].mean():.1f}")
    print(f"  Max: {hsv[:,:,2].max()}")
    print(f"\nCurrent threshold: {brightness_threshold}")
    print(f"{'='*70}")
    
    return clear_sky_mask, cloud_mask, hsv


def test_brightness_thresholds(img_path):
    """
    Test different brightness thresholds to find the best one.
    """
    img = cv2.imread(str(img_path))
    if img is None:
        print(f"ERROR: Could not load image: {img_path}")
        return
    
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
    
    print(f"\n{'='*70}")
    print("Testing Different Brightness Thresholds")
    print(f"{'='*70}\n")
    
    thresholds = [100, 120, 140, 150, 160, 180, 200]
    
    print(f"{'Threshold':<15} {'Clear-sky %':<20} {'Clouds %':<20}")
    print("-" * 55)
    
    for thresh in thresholds:
        clear_mask = hsv[:, :, 2] < thresh
        clear_pct = clear_mask.sum() / clear_mask.size * 100
        cloud_pct = 100 - clear_pct
        marker = " ← CURRENT" if thresh == 150 else ""
        print(f"{thresh:<15} {clear_pct:<20.1f} {cloud_pct:<20.1f}{marker}")
    
    print(f"\n{'='*70}")
    print("💡 RECOMMENDATION:")
    print("   - If clear-sky % > 80%, INCREASE threshold")
    print("   - If clear-sky % < 20%, DECREASE threshold")
    print("   - Target: 40-70% clear sky for balanced detection")
    print(f"{'='*70}\n")


def visualize_detection(img_path, brightness_threshold=150):
    """
    Create visualization of clear-sky detection.
    """
    img = cv2.imread(str(img_path))
    if img is None:
        return
    
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
    
    clear_sky_mask = hsv[:, :, 2] < brightness_threshold
    
    # Create figure
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    
    # Original image
    axes[0].imshow(img)
    axes[0].set_title('Original Image', fontsize=12, fontweight='bold')
    axes[0].axis('off')
    
    # V channel (brightness)
    im1 = axes[1].imshow(hsv[:, :, 2], cmap='gray')
    axes[1].set_title('V Channel (Brightness/Value)', fontsize=12, fontweight='bold')
    axes[1].axis('off')
    plt.colorbar(im1, ax=axes[1])
    
    # Clear sky mask
    im2 = axes[2].imshow(clear_sky_mask, cmap='RdYlGn_r')
    axes[2].set_title(f'Clear Sky Mask (V > {brightness_threshold})\nGreen=Clear, Red=Cloud', 
                     fontsize=12, fontweight='bold')
    axes[2].axis('off')
    plt.colorbar(im2, ax=axes[2], label='Clear Sky')
    
    plt.tight_layout()
    
    output_path = 'clear_sky_analysis.png'
    plt.savefig(output_path, dpi=100, bbox_inches='tight')
    print(f"✓ Saved visualization: {output_path}")
    plt.close()


# ============================================================================
# MAIN - Edit these paths!
# ============================================================================

if __name__ == "__main__":
    # YOUR IMAGE PATHS HERE
    img_path = "data/Images/ValidationSet/20250803105345_11.jpg"
    
    print("\n" + "="*70)
    print("CLEAR-SKY REGION ANALYSIS")
    print("="*70)
    
    # Check if file exists
    if not Path(img_path).exists():
        print(f"\n❌ ERROR: Image not found at: {img_path}")
        print(f"\nPlease check:")
        print(f"  1. File path is correct")
        print(f"  2. You're running from the correct directory")
        print(f"  3. File exists: {Path(img_path).resolve()}")
    else:
        print(f"\n✓ Found image: {img_path}\n")
        
        # Run diagnostics
        result = analyze_clear_sky_detection(img_path, brightness_threshold=140)
        test_brightness_thresholds(img_path)
        visualize_detection(img_path, brightness_threshold=140)
        
        print("\n" + "="*70)
        print("ANALYSIS COMPLETE")
        print("="*70)
        print(f"\nNext steps:")
        print(f"1. Review 'clear_sky_analysis.png' visualization")
        print(f"2. Check the threshold recommendations above")
        print(f"3. If needed, adjust brightness_threshold in ClearSkyAwareLoss")
        print(f"   (currently set to 150)")
        print()