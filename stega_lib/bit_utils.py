import cv2
import numpy as np


def img_to_bits(img: np.ndarray, chunk_size=1000000) -> (np.ndarray, tuple):
    """
    Converts an image to a flat bit array and returns its shape.
    Memory-efficient version using chunked processing.
    
    Returns:
        (bit_array, shape)
    """
    shape = img.shape
    total_bytes = img.size
    total_bits = total_bytes * 8
    
    # Pre-allocate result array
    bit_array = np.zeros(total_bits, dtype=np.uint8)
    
    # Process in chunks
    remaining = total_bytes
    pos = 0
    
    while remaining > 0:
        # Calculate chunk size
        current_chunk = min(chunk_size, remaining)
        end_pos = pos + current_chunk
        
        # Process chunk
        chunk_bits = np.unpackbits(img.ravel()[pos:end_pos])
        bit_array[pos*8:(end_pos*8)] = chunk_bits
        
        # Update counters
        remaining -= current_chunk
        pos = end_pos
        
        # Free memory
        del chunk_bits
    
    return bit_array, shape


def bits_to_img(bit_array: np.ndarray, shape: tuple, chunk_size=1000000) -> np.ndarray:
    """
    Converts a flat bit array back into an image of the given shape.
    Memory-efficient version using chunked processing.
    """
    num_bytes_needed = int(np.prod(shape))
    # Ensure bit_array is a multiple of 8
    num_bits_needed = num_bytes_needed * 8
    
    if len(bit_array) < num_bits_needed:
        # Pad with zeros if data is incomplete
        bit_array = np.pad(bit_array, (0, num_bits_needed - len(bit_array)))
    else:
        # Truncate if there's extra data
        bit_array = bit_array[:num_bits_needed]

    # Pre-allocate result array
    byte_array = np.zeros(num_bytes_needed, dtype=np.uint8)
    
    # Process in chunks
    remaining_bits = len(bit_array)
    pos = 0
    
    while remaining_bits > 0:
        # Calculate chunk size (must be multiple of 8)
        current_chunk = min(chunk_size * 8, remaining_bits)
        current_chunk = (current_chunk // 8) * 8  # Ensure multiple of 8
        end_pos = pos + current_chunk
        
        # Process chunk
        byte_pos = pos // 8
        byte_end = end_pos // 8
        byte_array[byte_pos:byte_end] = np.packbits(bit_array[pos:end_pos])
        
        # Update counters
        remaining_bits -= current_chunk
        pos = end_pos
    
    img = byte_array.reshape(shape)
    return img


def create_header(shape: tuple, bit_len: int) -> np.ndarray:
    """
    Creates a 128-bit header containing image shape and total bit length.
    Memory-efficient version.
    (32 bits for H, 32 for W, 32 for C, 32 for bit_len)
    """
    h, w, c = shape
    header_data = np.array([h, w, c, bit_len], dtype=np.uint32)
    header_bits = np.unpackbits(header_data.view(np.uint8))
    header_data = None  # Help garbage collection
    return header_bits


def parse_header(header_bits: np.ndarray) -> (tuple, int):
    """
    Parses the 128-bit header to get image shape and total bit length.
    Memory-efficient version.
    """
    if len(header_bits) != 128:
        raise ValueError(f"Header must be 128 bits, but got {len(header_bits)}")
    
    # Process header efficiently
    header_bytes = np.packbits(header_bits)
    header_data = header_bytes.view(np.uint32)
    h, w, c, bit_len = map(int, header_data)  # Convert to Python ints to free memory
    header_bytes = None  # Help garbage collection
    header_data = None  # Help garbage collection
    
    return (h, w, c), bit_len