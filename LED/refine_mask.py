import cv2
import numpy as np
import matplotlib.pyplot as plt
import glob
import os
from PIL import Image
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--scene_path', type=str, required=True, help='Path to scene directory')
args = parser.parse_args()
scene_path = args.scene_path
vis_path = os.path.join(scene_path, 'vis')
vis_mask_path = os.path.join(vis_path, 'masks.npy')
image_paths = glob.glob(os.path.join(scene_path, 'processed_images', '*.png'))
assert image_paths, "No images found!"

# Load the float mask from .npy
float_mask = np.load(vis_mask_path)
print(f"Loaded mask shape: {float_mask.shape}")

# Load one sample image for overlay preview
sample_image = Image.open(image_paths[0]).convert('RGB')
sample_image_np = np.array(sample_image)

# ---------- 1. INTERACTIVE THRESHOLD ADJUSTMENT ----------
min_thresh = 0.5
max_thresh = 0.7

while True:
    # Apply thresholding to float_mask
    binary_mask = np.where((float_mask >= min_thresh) & (float_mask <= max_thresh), 255, 0).astype(np.uint8)

    # Overlay mask
    overlay = sample_image_np.copy()
    overlay[binary_mask == 255] = [255, 0, 0]
    overlayed_image = cv2.addWeighted(sample_image_np, 0.5, overlay, 0.5, 0)

    # Show threshold preview
    plt.figure(figsize=(14, 6))
    plt.subplot(1, 3, 1)
    plt.imshow(float_mask, cmap='jet')
    plt.title('Original Attention Mask')
    plt.axis('off')

    plt.subplot(1, 3, 2)
    plt.imshow(binary_mask, cmap='gray')
    plt.title(f'Thresholded Mask [{min_thresh}, {max_thresh}]')
    plt.axis('off')

    plt.subplot(1, 3, 3)
    plt.imshow(overlayed_image)
    plt.title('Overlay on Image')
    plt.axis('off')
    plt.tight_layout()
    plt.show()

    response = input("Are you satisfied with the thresholded mask? (y/n): ").strip().lower()
    if response == 'y':
        break
    min_thresh = float(input("Enter new min threshold (0.0 ~ 1.0): ").strip())
    max_thresh = float(input("Enter new max threshold (0.0 ~ 1.0): ").strip())

# ---------- 2. INTERACTIVE DILATION REFINE ----------
kernel_size = 3
iterations = 3

def apply_dilation_and_overlay(mask, image_np, k_size, iters):
    kernel = np.ones((k_size, k_size), np.uint8)
    dilated = cv2.dilate(mask, kernel, iterations=iters)
    refined = cv2.bitwise_and(dilated, dilated)
    overlay = image_np.copy()
    overlay[refined == 255] = [255, 0, 0]
    blended = cv2.addWeighted(image_np, 0.5, overlay, 0.5, 0)
    return refined, blended

while True:
    refined_mask, overlayed_dilated = apply_dilation_and_overlay(binary_mask, sample_image_np, kernel_size, iterations)

    plt.figure(figsize=(14, 6))
    plt.subplot(1, 3, 1)
    plt.imshow(binary_mask, cmap='gray')
    plt.title('Original Thresholded Mask')
    plt.axis('off')

    plt.subplot(1, 3, 2)
    plt.imshow(refined_mask, cmap='gray')
    plt.title(f'Dilated Mask (Kernel={kernel_size}, Iter={iterations})')
    plt.axis('off')

    plt.subplot(1, 3, 3)
    plt.imshow(overlayed_dilated)
    plt.title('Dilated Mask Overlay')
    plt.axis('off')
    plt.tight_layout()
    plt.show()

    response = input("Are you satisfied with the refined mask? (y/n): ").strip().lower()
    if response == 'y':
        break
    kernel_size = int(input("Enter new kernel size: ").strip())
    iterations = int(input("Enter new iteration count: ").strip())

# ---------- 3. SAVE FINAL MASK ----------
refined_mask_path = os.path.join(vis_path, 'refine_masks_cv.png')
cv2.imwrite(refined_mask_path, refined_mask)
print(f"Saved refined mask to {refined_mask_path}")

# ---------- 4. GENERATE AND SAVE GS MASK ----------
masks_output_dir = os.path.join(scene_path, 'masks')
os.makedirs(masks_output_dir, exist_ok=True)
#gs_mask = 1 - (refined_mask / 255.0)
gs_mask = (refined_mask == 0).astype(np.uint8)

gs_mask_img = Image.fromarray((gs_mask).astype(np.uint8))

for img_path in image_paths:
    base_name = os.path.basename(img_path)
    gs_mask_img.save(os.path.join(masks_output_dir, base_name + '.png'))

print(f"Saved inverse GS masks to {masks_output_dir}")
