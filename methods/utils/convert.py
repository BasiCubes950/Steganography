import cv2
import numpy as np
import os

def image_to_bits(image_path):
    """Load an image and convert it to a binary bit array."""
    img = cv2.imread(image_path, cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(f"Image not found: {image_path}")
    _, buffer = cv2.imencode('.png', img)
    return np.unpackbits(np.frombuffer(buffer, dtype=np.uint8)), img.shape

def bits_to_image(bits, shape, output_path=None):
    """Rebuild image from bit array and save to disk.

    If `output_path` is not provided, the image will be saved to
    `results/images/reconstructed.png`. The function will create the
    parent directory if it doesn't exist.
    """
    # Ensure we have a target path and that the parent directory exists
    if output_path is None:
        # prefer existing capitalization if present; fall back to 'results/images'
        preferred_dirs = [os.path.join("Results", "images"), os.path.join("results", "images")]
        output_dir = None
        for d in preferred_dirs:
            if os.path.isdir(d):
                output_dir = d
                break
        if output_dir is None:
            output_dir = os.path.join("results", "images")
        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(output_dir, "reconstructed.png")
    else:
        parent = os.path.dirname(output_path)
        if parent:
            os.makedirs(parent, exist_ok=True)

    byte_data = np.packbits(bits)
    img_array = np.frombuffer(byte_data, dtype=np.uint8)
    img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
    if img is None:
        print("Warning: Could not decode image. Possibly corrupted or truncated.")
    else:
        cv2.imwrite(output_path, img)
        print(f"✅ Reconstructed secret image saved to {output_path}")
