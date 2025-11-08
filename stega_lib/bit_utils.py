import cv2
import numpy as np


def img_to_bits(img: np.ndarray) -> (np.ndarray, tuple):
    """
    Converts an image to a flat bit array and returns its shape.
    
    Returns:
        (bit_array, shape)
    """
    shape = img.shape
    flat_bytes = img.flatten()
    bit_array = np.unpackbits(flat_bytes)
    return bit_array, shape


def bits_to_img(bit_array: np.ndarray, shape: tuple) -> np.ndarray:
    """
    Converts a flat bit array back into an image of the given shape.
    """
    num_bytes_needed = int(np.prod(shape))
    # Ensure bit_array is a multiple of 8
    num_bits_needed = num_bytes_needed * 8
    
    if len(bit_array) < num_bits_needed:
        # Pad with zeros if data is incomplete (e.g., failed extraction)
        padding = np.zeros(num_bits_needed - len(bit_array), dtype=np.uint8)
        bit_array = np.concatenate((bit_array, padding))
    else:
        # Truncate if there's extra data
        bit_array = bit_array[:num_bits_needed]

    byte_array = np.packbits(bit_array)
    img = byte_array.reshape(shape)
    return img.astype(np.uint8)


def create_header(shape: tuple, bit_len: int) -> np.ndarray:
    """
    Creates a 128-bit header containing image shape and total bit length.
    (32 bits for H, 32 for W, 32 for C, 32 for bit_len)
    """
    h, w, c = shape
    header_data = np.array([h, w, c, bit_len], dtype=np.uint32)
    header_bits = np.unpackbits(header_data.view(np.uint8))
    # 4 * 32-bit uints = 16 * 8-bit uints = 128 bits
    return header_bits


def parse_header(header_bits: np.ndarray) -> (tuple, int):
    """
    Parses the 128-bit header to get image shape and total bit length.
    """
    if len(header_bits) != 128:
        raise ValueError(f"Header must be 128 bits, but got {len(header_bits)}")
        
    header_bytes = np.packbits(header_bits)
    header_data = header_bytes.view(np.uint32)
    h, w, c, bit_len = header_data
    return (int(h), int(w), int(c)), int(bit_len)