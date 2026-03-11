import cv2
import numpy as np
import matplotlib.pyplot as plt

img1 = cv2.imread("data/Images/ValidationSet/20250425094615_11.jpg")
img2 = cv2.imread("data/Images/ValidationSet/20250425094600_11.jpg")

# Your threshold
hsv1 = cv2.cvtColor(img1, cv2.COLOR_BGR2HSV)
dark_clouds = hsv1[:, :, 2] <= 100  # Very dark areas

# Compute Farnebäck
gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

flow = cv2.calcOpticalFlowFarneback(gray1, gray2, None, 0.5, 3, 15, 3, 5, 1.2, 0)
mag = np.sqrt(flow[:, :, 0]**2 + flow[:, :, 1]**2)

# Check magnitude in dark cloud regions
dark_magnitudes = mag[dark_clouds]

print(f"Farnebäck flow in dark clouds:")
print(f"  Mean magnitude: {dark_magnitudes.mean():.4f} px/frame")
print(f"  Std: {dark_magnitudes.std():.4f}")
print(f"  Max: {dark_magnitudes.max():.4f}")
print(f"  % near-zero (< 0.1): {(dark_magnitudes < 0.1).sum()/len(dark_magnitudes)*100:.1f}%")

# Visualize
fig, axes = plt.subplots(1, 3, figsize=(15, 5))

axes[0].imshow(cv2.cvtColor(img1, cv2.COLOR_BGR2RGB))
axes[0].set_title('Image')

axes[1].imshow(dark_clouds, cmap='gray')
axes[1].set_title('Dark Clouds (V <= 100)')

axes[2].imshow(mag, cmap='jet')
axes[2].set_title('Farnebäck Flow Magnitude')

plt.tight_layout()
plt.savefig('pseudo_label_quality.png')
print("\n✓ Saved: pseudo_label_quality.png")