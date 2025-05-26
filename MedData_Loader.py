# MedData_Loader.py
import os
import cv2
import numpy as np

def load_images(folder, gray=True, target_size=(128, 128), threshold=0.8):
    """
    Loads images and generates corresponding binary masks using a pixel threshold.
    Returns: (images, masks)
    - images: float32, shape (N, H, W, 1) if gray else (N, H, W, 3), normalized [0,1]
    - masks: uint8, shape (N, H, W, 1), values 0 or 1
    """
    images, masks = [], []

    for fname in os.listdir(folder):
        if fname.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp')):
            path = os.path.join(folder, fname)
            flag = cv2.IMREAD_GRAYSCALE if gray else cv2.IMREAD_COLOR
            img = cv2.imread(path, flag)

            if img is None:
                continue  # skip corrupted or unreadable images

            if target_size:
                img = cv2.resize(img, target_size)

            img_norm = img.astype(np.float32) / 255.0
            images.append(img_norm[..., None] if gray else img_norm)

            # Generate binary mask from the grayscale image using a pixel threshold
            mask = (img_norm > threshold).astype(np.uint8)
            masks.append(mask[..., None])  # ensure shape is (H, W, 1)

    return np.array(images), np.array(masks)
