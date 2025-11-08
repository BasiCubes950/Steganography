"""
Motion-Vector Based Extraction
Extracts hidden image from quantized optical flow vectors with LSB extraction.
"""
import cv2
import numpy as np
from pathlib import Path
import os
import sys

# Scale factor for flow vectors (must match embedding)
FLOW_SCALE = 10.0
# Header size constant
HEADER_SIZE_BITS = 72  # 9 bytes for H, W, C, and Data Length (4 bytes)


def parse_header(header_bits: np.ndarray):
    """Parse header to extract shape and data length."""
    if len(header_bits) < HEADER_SIZE_BITS:
        raise ValueError(f"Header bits too short. Expected {HEADER_SIZE_BITS}, got {len(header_bits)}")
        
    header_bytes = np.packbits(header_bits[:HEADER_SIZE_BITS])
    
    # 2 bytes for H, 2 bytes for W, 1 byte for C, 4 bytes for Data Length
    # H (16-bit): [0] [1]
    h = (int(header_bytes[0]) << 8) | int(header_bytes[1])
    # W (16-bit): [2] [3]
    w = (int(header_bytes[2]) << 8) | int(header_bytes[3])
    # C (8-bit): [4]
    c = int(header_bytes[4])
    
    # Data Length (32-bit): [5] [6] [7] [8]
    data_length = (int(header_bytes[5]) << 24) | (int(header_bytes[6]) << 16) | \
                  (int(header_bytes[7]) << 8) | int(header_bytes[8])
                  
    return (h, w, c), data_length


def extract_bits_from_lsb(carrier: np.ndarray, num_bits: int):
    """Extract bits from the LSBs of carrier array."""
    carrier_flat = carrier.flatten()
    extracted = []
    
    # We only care about the LSB of the first 'num_bits' worth of elements
    for i in range(min(num_bits, len(carrier_flat))):
        extracted.append(carrier_flat[i] & 1)
    
    return np.array(extracted, dtype=np.uint8)


def extract_motion_vector(video_path, output_path, secret_shape=None):
    """
    Extract secret image from optical flow motion vectors using LSB extraction.
    """
    print(f"Starting motion vector extraction...")
    print(f"Extracting from: {video_path}")
    
    # Open video
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise ValueError(f"Cannot open video: {video_path}")
    
    # Get video properties
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    print(f"Video: {width}x{height}, {total_frames} frames")
    
    if total_frames < 2:
        raise ValueError("Video must have at least 2 frames to calculate flow")
    
    # Read first frame and extract header
    ret, first_frame = cap.read()
    if not ret:
        raise ValueError("Cannot read first frame")
    
    # 1. Extract header bits (72 bits) from the first frame's spatial LSBs
    header_bits = extract_bits_from_lsb(first_frame, HEADER_SIZE_BITS)
    
    try:
        shape, data_length = parse_header(header_bits)
    except ValueError as e:
        print(f"ERROR parsing header: {e}")
        cap.release()
        raise
        
    print(f"Extracted header - Shape: {shape}, Data length: {data_length} bits")
    
    if secret_shape is not None and secret_shape != shape:
        print(f"Warning: Provided shape {secret_shape} doesn't match header shape {shape}")
    
    # Initialize extraction
    extracted_bits = []
    frame_idx = 1
    prev_frame = first_frame
    prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
    
    print("Extracting data from motion vectors...")
    
    # Process remaining frames
    while cap.isOpened() and len(extracted_bits) < data_length:
        ret, current_frame = cap.read()
        if not ret:
            break
        
        # Calculate optical flow
        current_gray = cv2.cvtColor(current_frame, cv2.COLOR_BGR2GRAY)
        
        # Farneback Optical Flow calculation
        flow = cv2.calcOpticalFlowFarneback(
            prev_gray, current_gray, None,
            pyr_scale=0.5, levels=3, winsize=15,
            iterations=3, poly_n=5, poly_sigma=1.2, flags=0
        )
        
        # Quantize flow (same as during embedding)
        # We must use a type that can handle negative values, such as np.int16
        flow_quant = np.round(flow * FLOW_SCALE).astype(np.int16)
        
        # Extract bits from LSBs of quantized flow
        bits_needed = data_length - len(extracted_bits)
        bits_in_frame = min(flow_quant.size, bits_needed)
        
        frame_bits = extract_bits_from_lsb(flow_quant, bits_in_frame)
        extracted_bits.extend(frame_bits)
        
        # Update for next iteration
        prev_gray = current_gray
        frame_idx += 1
        
        if frame_idx % 20 == 0:
            sys.stdout.write(f"\rProcessing frame {frame_idx}/{total_frames}, extracted {len(extracted_bits)}/{data_length} bits")
            sys.stdout.flush()

    cap.release()
    sys.stdout.write("\n") # Newline after progress bar
    
    print(f"Extraction finished. Total extracted bits: {len(extracted_bits)}")
    
    # Handle incomplete extraction
    if len(extracted_bits) < data_length:
        print(f"Warning: Only extracted {len(extracted_bits)}/{data_length} bits. Padding with zeros.")
        # Pad with zeros
        extracted_bits.extend([0] * (data_length - len(extracted_bits)))
    
    # 2. Convert bits back to image
    extracted_bits_array = np.array(extracted_bits[:data_length], dtype=np.uint8)
    extracted_bytes = np.packbits(extracted_bits_array)
    
    # Calculate expected number of bytes
    h, w, c = shape
    expected_bytes = h * w * c
    
    if len(extracted_bytes) < expected_bytes:
        print(f"Warning: Expected {expected_bytes} bytes, got {len(extracted_bytes)}. Padding with zeros.")
        # Pad with zeros
        padding = np.zeros(expected_bytes - len(extracted_bytes), dtype=np.uint8)
        extracted_bytes = np.concatenate([extracted_bytes, padding])
    
    # Reshape to image
    # Ensure the image is interpreted as unsigned 8-bit integers (0-255)
    extracted_img = extracted_bytes[:expected_bytes].reshape(shape).astype(np.uint8)
    
    # Save extracted image
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output_path), extracted_img)
    
    print(f"✅ Extracted image saved to: {output_path}")
    
    return extracted_img


# --- RUNNER SCRIPT ---

if __name__ == "__main__":
    
    # --- CONFIGURATION ---
    # !! UPDATE THIS PATH TO YOUR STEGO VIDEO FILE !!
    # This should be the video created using the motion-vector embedding method.
    STEGO_VIDEO_PATH = Path("output/stego/motion_vector_stego.mp4") 
    
    OUTPUT_IMAGE_PATH = "output/extracted/extracted_mv_image.png"
    
    if not STEGO_VIDEO_PATH.exists():
        print(f"ERROR: Stego video not found at '{STEGO_VIDEO_PATH}'.")
        print("Please update the STEGO_VIDEO_PATH variable with the correct file path.")
    else:
        print("\n--- Starting Motion Vector Extraction Test ---")

        # --- EXECUTION ---
        try:
            extract_motion_vector(str(STEGO_VIDEO_PATH), str(OUTPUT_IMAGE_PATH))
            print("\n✅ Extraction Test Finished Successfully.")
        except ValueError as e:
            print(f"\n❌ Extraction Failed (Video/Data Error): {e}")
        except Exception as e:
            print(f"\n❌ Extraction Failed: An unexpected error occurred: {e}")