import os
import numpy as np
import cv2
from skimage.transform import resize

# Define the correct folder path
mask_folder = "data/aptiv_single"

def adjust_mask(mask_path, output_path, shift_x=-120, shift_y=10):
    """ Load, resize, shift, and save the adjusted mask """
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)  # Load mask

    if mask is None:
        print(f"❌ ERROR: Failed to load {mask_path}. Check if the file exists!")
        return  # Exit function if mask is not found

    mask = resize(mask, (800, 1000), anti_aliasing=False) > 0.5  # Resize & binarize
    mask = np.roll(mask, shift=(shift_x, shift_y), axis=(0, 1))  # Apply shift
    mask = (mask * 255).astype('uint8')  # Convert to uint8
    cv2.imwrite(output_path, mask)  # Save adjusted mask
    print(f"✅ Saved adjusted mask: {output_path}")

# Update mask paths with the correct folder
mask_paths = [os.path.join(mask_folder, f"mask{i+1}.png") for i in range(4)]
output_paths = [os.path.join(mask_folder, f"mask_adjusted_{i+1}.png") for i in range(4)]

# Process all masks
for i in range(4):
    adjust_mask(mask_paths[i], output_paths[i], shift_x=-120, shift_y=10)
