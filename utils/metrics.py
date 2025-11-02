import numpy as np
import cv2
from skimage.metrics import structural_similarity as ssim

def bit_error_rate(original_bits, recovered_bits):
    """Compute Bit Error Rate (BER)."""
    if len(original_bits) != len(recovered_bits):
        min_len = min(len(original_bits), len(recovered_bits))
        original_bits = original_bits[:min_len]
        recovered_bits = recovered_bits[:min_len]
    errors = np.sum(np.array(original_bits) != np.array(recovered_bits))
    return errors / len(original_bits)

def psnr(img1, img2):
    """Compute Peak Signal-to-Noise Ratio (PSNR) between two images."""
    return cv2.PSNR(img1, img2)

def ssim_score(img1, img2):
    """Compute SSIM between two images."""
    if img1.shape != img2.shape:
        img2 = cv2.resize(img2, (img1.shape[1], img1.shape[0]))
    return ssim(cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY),
                cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY))

def recovery_probability(successes, total):
    """Compute recovery probability (fraction of successful recoveries)."""
    return successes / total if total > 0 else 0.0
