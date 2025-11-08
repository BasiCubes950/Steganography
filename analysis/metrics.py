import cv2
import numpy as np
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import mean_squared_error as mse
from skimage.measure import shannon_entropy
from brisque import BRISQUE
from stega_lib.bit_utils import img_to_bits

# Initialize BRISQUE model
# Note: This might require downloading model files on first run.
brisque_model = BRISQUE(url=False) # Use local model files

def calculate_ber(original_img: np.ndarray, extracted_img: np.ndarray) -> float:
    """
    Calculates the Bit Error Rate (BER) between two images.
    """
    # Ensure images have the same shape, resize if necessary (e.g., failed extraction)
    if original_img.shape != extracted_img.shape:
        extracted_img = cv2.resize(extracted_img, (original_img.shape[1], original_img.shape[0]))
        
    original_bits, _ = img_to_bits(original_img)
    extracted_bits, _ = img_to_bits(extracted_img)
    
    # Ensure bitstreams are the same length
    min_len = min(len(original_bits), len(extracted_bits))
    original_bits = original_bits[:min_len]
    extracted_bits = extracted_bits[:min_len]
    
    if min_len == 0:
        return 1.0  # Total failure
        
    ber = np.mean(original_bits != extracted_bits)
    return ber


def run_all_metrics(original_img: np.ndarray, extracted_img: np.ndarray) -> dict:
    """
    Calculates all required quality and error metrics.
    """
    # Ensure images have the same shape for comparison
    if original_img.shape != extracted_img.shape:
        extracted_img = cv2.resize(
            extracted_img, 
            (original_img.shape[1], original_img.shape[0]),
            interpolation=cv2.INTER_NEAREST
        )
        
    # --- Reference-based Metrics ---
    try:
        psnr_val = psnr(original_img, extracted_img, data_range=255)
    except Exception:
        psnr_val = 0.0 # Handle case of all-black image
        
    try:
        # data_range is deprecated, use win_size if needed, but multichannel=True is key
        ssim_val = ssim(original_img, extracted_img, data_range=255, channel_axis=2)
    except Exception:
        ssim_val = 0.0

    mse_val = mse(original_img, extracted_img)
    
    # --- Bit Error Metrics ---
    ber_val = calculate_ber(original_img, extracted_img)
    recovery_prob = 1.0 - ber_val
    
    # --- No-Reference / Content Metrics ---
    # Convert to grayscale for entropy and BRISQUE
    if len(extracted_img.shape) == 3 and extracted_img.shape[2] == 3:
        extracted_gray = cv2.cvtColor(extracted_img, cv2.COLOR_BGR2GRAY)
    else:
        extracted_gray = extracted_img # Assume already grayscale or failed
        
    entropy_val = shannon_entropy(extracted_gray)
    
    try:
        # BRISQUE expects grayscale
        brisque_val = brisque_model.score(extracted_gray)
    except Exception:
        # Can fail on pure black/white images
        brisque_val = 100.0 # Max (worst) score
        
    return {
        "PSNR": psnr_val,
        "SSIM": ssim_val,
        "MSE": mse_val,
        "BER": ber_val,
        "RecoveryProbability": recovery_prob,
        "Entropy": entropy_val,
        "BRISQUE": brisque_val
    }