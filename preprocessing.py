# preprocessing.py
import cv2
import numpy as np

# Standard size for matching grid consistency
STD_SIZE = (300, 300)

def enhance_fingerprint(image_path, blur_sigma=0, return_binarized=False, do_align=False):
    """
    Enhances the fingerprint image.
    - do_align: if True calls align_fingerprint_orientation (useful only for specific cases).
    """
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        print(f"Warning: Could not read {image_path}")
        return None if not return_binarized else (None, None)

    if blur_sigma > 0:
        img = cv2.GaussianBlur(img, (0, 0), blur_sigma)

    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    img_eq = clahe.apply(img)

    if do_align:
        img_aligned, _ = align_fingerprint_orientation(img_eq)
    else:
        img_aligned = img_eq

    img_roi = get_fingerprint_roi(img_aligned)
    try:
        img_fixed = cv2.resize(img_roi, STD_SIZE, interpolation=cv2.INTER_AREA)
    except Exception:
        img_fixed = cv2.resize(img_aligned, STD_SIZE, interpolation=cv2.INTER_AREA)

    if return_binarized:
        binary_img = cv2.adaptiveThreshold(
            img_fixed, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY_INV, 15, 2
        )
        return img_fixed, binary_img
    else:
        return img_fixed

def get_fingerprint_roi(img):
    _, binary = cv2.threshold(img, 240, 255, cv2.THRESH_BINARY_INV)
    coords = cv2.findNonZero(binary)
    if coords is None:
        return img
    x, y, w, h = cv2.boundingRect(coords)
    crop = img[y:y+h, x:x+w]
    return crop

def align_fingerprint_orientation(img_gray):
    """Estimate and rotate to make ridges vertical; returns (rotated_img, rotation_angle)"""
    gx = cv2.Sobel(img_gray, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(img_gray, cv2.CV_32F, 0, 1, ksize=3)

    g2x = gx ** 2
    g2y = gy ** 2
    gxy = gx * gy

    sigma = 5
    g2x = cv2.GaussianBlur(g2x, (0, 0), sigma)
    g2y = cv2.GaussianBlur(g2y, (0, 0), sigma)
    gxy = cv2.GaussianBlur(gxy, (0, 0), sigma)

    mask = img_gray < 240
    if np.sum(mask) == 0:
        return img_gray, 0.0

    mean_g2x = np.mean(g2x[mask])
    mean_g2y = np.mean(g2y[mask])
    mean_gxy = np.mean(gxy[mask])

    angle_rad = 0.5 * np.arctan2(2 * mean_gxy, mean_g2x - mean_g2y)
    angle_deg = np.degrees(angle_rad)
    rotation_angle = -angle_deg

    img_rotated = rotate_image_lossless(img_gray, rotation_angle)
    return img_rotated, rotation_angle

def rotate_image_lossless(image, angle):
    """
    Rotates an image without cropping it.
    Includes a safety crop to remove dark edges from the source.
    """
    # 1. SAFETY CROP: Shave off 2 pixels from each side
    # This removes scanning artifacts or dark borders from the original file
    h_orig, w_orig = image.shape[:2]
    safe_img = image[2:h_orig-2, 2:w_orig-2]
    
    # 2. Add EXTENDED White Padding
    # We add more padding (5px) to be safe against interpolation artifacts
    image_padded = cv2.copyMakeBorder(
        safe_img, 
        5, 5, 5, 5, 
        cv2.BORDER_CONSTANT, 
        value=255 # White
    )

    h, w = image_padded.shape[:2]
    center = (w // 2, h // 2)

    # 3. Calculate Rotation Matrix
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    cos = np.abs(M[0, 0])
    sin = np.abs(M[0, 1])
    
    # Calculate new bounding box dimensions
    new_w = int((h * sin) + (w * cos))
    new_h = int((h * cos) + (w * sin))
    
    # Adjust translation
    M[0, 2] += (new_w / 2) - center[0]
    M[1, 2] += (new_h / 2) - center[1]
    
    # 4. Warp with White Background
    rotated = cv2.warpAffine(
        image_padded, 
        M, 
        (new_w, new_h),
        flags=cv2.INTER_LANCZOS4,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=255 # STRICTLY WHITE
    )
    
    return rotated