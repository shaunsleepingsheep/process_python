# Các hàm xử lý ảnh dùng chung cho ứng dụng.

import cv2
import numpy as np

# Xử lý cơ bản (01_basic_processing)

def rotate_image(img, angle):
    """Xoay ảnh quanh tâm, giữ nguyên kích thước, nền đen."""
    h, w = img.shape[:2]
    center = (w / 2, h / 2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    return cv2.warpAffine(img, M, (w, h), borderValue=0)

def flip_image(img, mode="horizontal"):
    """Lật ảnh. mode: 'horizontal' | 'vertical' | 'both'"""
    flip_code = {"horizontal": 1, "vertical": 0, "both": -1}[mode]
    return cv2.flip(img, flip_code)

def crop_image(img, mask=None, margin: int = 10):
    """Cắt ảnh theo bounding box của mask phổi (kèm margin).
    Nếu không có mask, center-crop 80% kích thước ảnh."""
    h, w = img.shape[:2]
    if mask is not None:
        mask_resized = cv2.resize(mask, (w, h), interpolation=cv2.INTER_NEAREST)
        ys, xs = np.where(mask_resized > 0)
        if len(xs) == 0:
            return img.copy()
        x0, x1 = max(xs.min() - margin, 0), min(xs.max() + margin, w)
        y0, y1 = max(ys.min() - margin, 0), min(ys.max() + margin, h)
        return img[y0:y1, x0:x1]
    else:
        cx, cy = w // 2, h // 2
        half_w, half_h = int(w * 0.4), int(h * 0.4)
        return img[cy - half_h:cy + half_h, cx - half_w:cx + half_w]

def denoise_image(img, method: str = "median"):
    """Khử nhiễu. method: 'gaussian' | 'median' | 'nlm' (Non-local Means)."""
    if method == "gaussian":
        return cv2.GaussianBlur(img, (5, 5), 0)
    elif method == "median":
        return cv2.medianBlur(img, 5)
    elif method == "nlm":
        return cv2.fastNlMeansDenoising(img, h=10)
    raise ValueError(method)

def edge_detect(img, method: str = "canny"):
    """Tách biên. method: 'canny' | 'sobel' | 'laplacian'"""
    if method == "canny":
        return cv2.Canny(img, 50, 150)
    elif method == "sobel":
        sx = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=3)
        sy = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=3)
        mag = cv2.magnitude(sx, sy)
        return cv2.convertScaleAbs(mag)
    elif method == "laplacian":
        lap = cv2.Laplacian(img, cv2.CV_64F, ksize=3)
        return cv2.convertScaleAbs(lap)
    raise ValueError(method)

def background_subtract(img, mask=None):
    """Tách nền: dùng mask phổi có sẵn nếu có, ngược lại dùng Otsu threshold + morphology."""
    h, w = img.shape[:2]
    if mask is not None:
        mask_resized = cv2.resize(mask, (w, h), interpolation=cv2.INTER_NEAREST)
        return cv2.bitwise_and(img, img, mask=mask_resized)
    else:
        _, otsu_mask = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
        otsu_mask = cv2.morphologyEx(otsu_mask, cv2.MORPH_CLOSE, kernel)
        return cv2.bitwise_and(img, img, mask=otsu_mask)

# Tăng cường ảnh (02_enhancement)

def hist_eq(img):
    """Histogram Equalization toàn cục - baseline so sánh."""
    return cv2.equalizeHist(img)

def clahe_enhance(img, clip_limit: float = 2.0, tile_grid_size: tuple = (8, 8)):
    """CLAHE - tăng tương phản cục bộ, hạn chế khuếch đại nhiễu. Chuẩn cho ảnh y tế."""
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
    return clahe.apply(img)

def gamma_correction(img, gamma: float = 1.5):
    """gamma > 1: làm sáng vùng tối; gamma < 1: làm tối vùng sáng."""
    inv_gamma = 1.0 / gamma
    table = np.array([((i / 255.0) ** inv_gamma) * 255 for i in range(256)]).astype("uint8")
    return cv2.LUT(img, table)

def unsharp_mask(img, sigma: float = 2.0, amount: float = 1.5):
    """Làm rõ nét các vệt mờ phổi bằng unsharp masking."""
    blurred = cv2.GaussianBlur(img, (0, 0), sigma)
    sharpened = cv2.addWeighted(img, 1 + amount, blurred, -amount, 0)
    return sharpened

def recommended_pipeline(img):
    """CLAHE (tăng tương phản cục bộ) + khử nhiễu nhẹ - tránh khuếch đại nhiễu quá mức."""
    enhanced = clahe_enhance(img)
    return cv2.medianBlur(enhanced, 3)