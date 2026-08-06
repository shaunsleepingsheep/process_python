"""I/O helpers shared by the processing notebooks.

Only dataset traversal / load / save logic lives here — the actual
image-processing algorithms stay in the notebooks.
"""
import os

import cv2

DATASET_ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "raw_data", "COVID-19_Radiography_Dataset")
CLASSES = ["COVID", "Lung_Opacity", "Normal", "Viral Pneumonia"]


def list_dataset(classes=None):
    """Return a list of (class_name, image_path, mask_path) tuples."""
    classes = classes or CLASSES
    items = []
    for class_name in classes:
        images_dir = os.path.join(DATASET_ROOT, class_name, "images")
        masks_dir = os.path.join(DATASET_ROOT, class_name, "masks")
        for filename in sorted(os.listdir(images_dir)):
            image_path = os.path.join(images_dir, filename)
            mask_path = os.path.join(masks_dir, filename)
            items.append((class_name, image_path, mask_path if os.path.exists(mask_path) else None))
    return items


def load_gray(path):
    """Load an image as grayscale (uint8 numpy array)."""
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(path)
    return img


def save_result(img, out_root, technique, class_name, filename):
    """Save a processed image to output/<out_root>/<technique>/<class_name>/<filename>."""
    base_dir = os.path.dirname(os.path.abspath(__file__))
    out_dir = os.path.join(base_dir, "output", out_root, technique, class_name)
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, filename)
    cv2.imwrite(out_path, img)
    return out_path
