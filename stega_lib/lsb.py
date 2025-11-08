import numpy as np


def embed_bits_in_lsb(carrier: np.ndarray, data_bits: np.ndarray) -> np.ndarray:
    """
    Embeds a stream of bits into the LSBs of a carrier numpy array.
    
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
        
    # Create a copy to avoid modifying the original array in place
    modified_carrier_flat = carrier_flat.copy()
    
    # Clear the LSBs of the carrier
    modified_carrier_flat[:data_len] &= 0xFE
    
    # Set the LSBs using the data bits
    modified_carrier_flat[:data_len] |= data_bits
    
    return modified_carrier_flat.reshape(carrier.shape)


def extract_bits_from_lsb(carrier: np.ndarray, num_bits: int) -> np.ndarray:
    """
    Extracts a stream of bits from the LSBs of a carrier numpy array.
    
    Args:
        carrier: The data to extract from (e.g., frame, diff, flow).
        num_bits: The number of bits to extract.
        
    Returns:
        The extracted bit array.
    """
    carrier_flat = carrier.flatten()
    
    if num_bits > len(carrier_flat):
        # Requesting more bits than available, extract all possible
        num_bits = len(carrier_flat)
        
    # Extract LSBs
    extracted_bits = carrier_flat[:num_bits] & 1
    
    return extracted_bits