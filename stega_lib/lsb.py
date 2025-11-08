import numpy as np


def embed_bits_in_lsb(carrier: np.ndarray, data_bits: np.ndarray) -> np.ndarray:
    """
    Embeds a stream of bits into the LSBs of a carrier numpy array.
    Memory-efficient version.
    
    Args:
        carrier: The data to hide in (e.g., frame, diff, flow).
        data_bits: The bits to embed (0s and 1s).
        
    Returns:
        The modified carrier array.
    """
    carrier_flat = carrier.flatten()
    data_len = len(data_bits)
    
    if data_len > len(carrier_flat):
        raise ValueError("Data to embed is larger than carrier capacity.")
        
    # Create a memory-efficient view to avoid copying large arrays
    modified_carrier_flat = carrier_flat.view()
    
    # Clear the LSBs of the carrier (in place)
    modified_carrier_flat[:data_len] &= 0xFE
    
    # Set the LSBs using the data bits (in place)
    modified_carrier_flat[:data_len] |= data_bits
    
    return modified_carrier_flat.reshape(carrier.shape)


def extract_bits_from_lsb(carrier: np.ndarray, num_bits: int, chunk_size=1000000) -> np.ndarray:
    """
    Extracts a stream of bits from the LSBs of a carrier numpy array.
    Memory-efficient version using chunked processing.
    
    Args:
        carrier: The data to extract from (e.g., frame, diff, flow).
        num_bits: The number of bits to extract.
        chunk_size: Size of chunks to process at a time.
        
    Returns:
        The extracted bit array.
    """
    if num_bits > carrier.size:
        # Requesting more bits than available
        num_bits = carrier.size
    
    # Pre-allocate the result array
    result = np.zeros(num_bits, dtype=np.uint8)
    
    # Extract in chunks to manage memory
    remaining = num_bits
    pos = 0
    
    while remaining > 0:
        # Calculate chunk size
        current_chunk = min(chunk_size, remaining)
        end_pos = pos + current_chunk
        
        # Process chunk
        result[pos:end_pos] = carrier.ravel()[pos:end_pos] & 1
        
        # Update counters
        remaining -= current_chunk
        pos = end_pos
        
    return result