import cv2
import numpy as np

def image_to_bits(image_path):
    """Load an image and convert it to a binary bit array."""
    img = cv2.imread(image_path, cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(f"Image not found: {image_path}")
    _, buffer = cv2.imencode('.png', img)
    return np.unpackbits(np.frombuffer(buffer, dtype=np.uint8)), img.shape

def bits_to_image(bits, shape, output_path):
    """Rebuild image from bit array."""
    byte_data = np.packbits(bits)
    img_array = np.frombuffer(byte_data, dtype=np.uint8)
    img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
    if img is None:
        print("Warning: Could not decode image. Possibly corrupted or truncated.")
    else:
        cv2.imwrite(output_path, img)
        print(f"✅ Reconstructed secret image saved to {output_path}")
