# feature_extraction.py
import numpy as np
import cv2
from skimage.feature import local_binary_pattern

# ------------------------------
# Serialization Helpers
# ------------------------------
def serialize_keypoints(keypoints):
    """Convert OpenCV Keypoints to a list of tuples for pickling."""
    if keypoints is None:
        return []
    return [(kp.pt, kp.size, kp.angle, kp.response, kp.octave, kp.class_id) for kp in keypoints]

def deserialize_keypoints(data):
    """Convert list of tuples back to OpenCV Keypoints."""
    if not data:
        return []
    keypoints = []
    for point in data:
        kp = cv2.KeyPoint(x=point[0][0], y=point[0][1], size=point[1], angle=point[2], 
                          response=point[3], octave=point[4], class_id=point[5])
        keypoints.append(kp)
    return keypoints

# ------------------------------
# SIFT Implementation (Rotation Invariant)
# ------------------------------
def extract_sift_features(image):
    """
    Extracts SIFT keypoints and descriptors.
    Returns: (keypoints, descriptors)
    """
    if image is None:
        return None, None
    
    # SIFT requires uint8
    if image.dtype != np.uint8:
        image = image.astype(np.uint8)

    # Initialize SIFT detector
    sift = cv2.SIFT_create()
    
    # Detect keypoints and compute descriptors
    keypoints, descriptors = sift.detectAndCompute(image, None)
    
    return keypoints, descriptors

# ------------------------------
# LBP Implementation (For Clustering)
# ------------------------------
# LBP parameters
N_POINTS = 24
RADIUS = 3
METHOD = 'uniform'
EPS = 1e-10
FFT_COEFFS = 64

def lbp_n_bins(P, method):
    if method == 'uniform':
        return P + 2
    else:
        return 2 ** P

def extract_raw_lbp_histogram_from_gray(gray_image, P=N_POINTS, R=RADIUS, method=METHOD):
    if gray_image is None or gray_image.size == 0:
        return None
    lbp = local_binary_pattern(gray_image, P=P, R=R, method=method)
    n_bins = lbp_n_bins(P, method)
    hist, _ = np.histogram(lbp.ravel(), bins=np.arange(0, n_bins + 1), range=(0, n_bins))
    hist = hist.astype(np.float64)
    s = hist.sum()
    if s > 0:
        hist /= (s + EPS)
    return hist

def fft_rotation_invariant_descriptor(hist, n_coeffs=FFT_COEFFS, normalize=True):
    if hist is None:
        return None
    spectrum = np.fft.fft(hist)
    mag = np.abs(spectrum)
    n_keep = min(n_coeffs, mag.shape[0])
    desc = mag[:n_keep].astype(np.float64)
    if normalize:
        norm = np.linalg.norm(desc) + EPS
        desc = desc / norm
    return desc

def extract_lbp_features_for_clustering(image):
    """
    Only extracts the global features needed for the Clustering/PCA step.
    """
    if image is None: 
        return None
    if len(image.shape) > 2:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image

    raw_hist = extract_raw_lbp_histogram_from_gray(gray)
    prefilt_desc = fft_rotation_invariant_descriptor(raw_hist)
    
    return prefilt_desc