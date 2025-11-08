# standalone_extractor.py
# Combines extraction logic and the runner script for easy testing.

import cv2
import numpy as np
from tqdm import tqdm
from pathlib import Path

# --- EXTRACTION FUNCTION ---

# Define header constants (MUST match stega_lib/bit_utils.py)
HEADER_SIZE_BITS = 256  # Example: 32 bits for width, 32 for height, 64 for data length

def extract_spatial_lsb(video_path: str, output_image_path: str):
    """
    Extracts a secret image from the LSBs of a video's frames.
    
    NOTE: This extraction uses a TEMPORARY HACK for image dimensions (1920x1080x3) 
    because the actual header parsing logic is not implemented.
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise IOError(f"Cannot open video file: {video_path}")

    extracted_bits = []
    
    # 1. --- EXTRACT HEADER BITS FIRST ---
    
    print("Step 1: Extracting Header Bits...")
    
    # Keep extracting until we get enough bits for the header
    while len(extracted_bits) < HEADER_SIZE_BITS:
        ret, frame = cap.read()
        if not ret:
            cap.release()
            raise ValueError("Video ended before header could be extracted.")

        # Flatten the frame bytes (H*W*C bytes)
        flat_frame = frame.flatten()
        
        # Extract the LSB from each byte of the flattened frame (byte & 1)
        frame_lsb_bits = [byte & 1 for byte in flat_frame]
        extracted_bits.extend(frame_lsb_bits)
    
    # --- TEMPORARY HACK: Hardcoded Size for 1920x1080 ---
    extracted_height, extracted_width, extracted_channels = 1080, 1920, 3
    data_length_bits = extracted_width * extracted_height * extracted_channels * 8
    
    # 2. --- EXTRACT REMAINING IMAGE DATA BITS ---
    
    # The image data starts right after the header bits we already extracted
    extracted_bits = extracted_bits[HEADER_SIZE_BITS:] 
    
    pbar = tqdm(total=data_length_bits, desc="Step 2: Extracting Image Data")
    pbar.update(len(extracted_bits)) # Account for bits already extracted with the header
    
    while len(extracted_bits) < data_length_bits:
        ret, frame = cap.read()
        if not ret:
            break

        flat_frame = frame.flatten()
        frame_lsb_bits = [byte & 1 for byte in flat_frame]
        extracted_bits.extend(frame_lsb_bits)
        
        # Update the progress bar by the number of bits just extracted
        pbar.update(len(frame_lsb_bits)) 
        
    pbar.close()
    cap.release()
    
    # 3. --- RECONSTRUCT IMAGE (Updated for Tiling/Alignment Fixes) ---
    
    print("Step 3: Reconstructing Image...")
    
    # Trim to the exact data size needed and convert to numpy array
    extracted_bits = np.array(extracted_bits[:data_length_bits], dtype=np.uint8)

    if len(extracted_bits) < data_length_bits:
        raise ValueError(f"Could not extract enough data. Found {len(extracted_bits)} bits, needed {data_length_bits}.")

    # Convert bits back to bytes
    extracted_bytes = np.packbits(extracted_bits)
    
    expected_bytes = extracted_width * extracted_height * extracted_channels
    # Ensure byte array size matches
    extracted_bytes = extracted_bytes[:expected_bytes]
    
    # Reshape the bytes into the final image structure (Height, Width, Channels)
    extracted_image = extracted_bytes.reshape(extracted_height, extracted_width, extracted_channels)

    # Note: If the image still looks tiled, the issue is likely that the
    # channel order (RGB vs BGR) does not match the original. 
    # Try adding this line if the image is tiled or colored wrong:
    # extracted_image = cv2.cvtColor(extracted_image.astype(np.uint8), cv2.COLOR_RGB2BGR)

    # Save the extracted image
    cv2.imwrite(output_image_path, extracted_image)
    print(f"Successfully extracted image to: {output_image_path}")
    
    return output_image_path

# --- STANDALONE RUNNER ---

def run_single_extraction_test():
    """
    A standalone script to test the extract_spatial_lsb function and save the result.
    """
    # --- Configuration ---
    # ⚠️ 1. DEFINE YOUR INPUT FILE PATH HERE 
    INPUT_VIDEO_PATH = "output/stego/spatial_lsb_stego.mp4" 
    
    output_image_path = "output/extracted/extracted_test_image_lsb.png"
    
    if not Path(INPUT_VIDEO_PATH).exists():
        print(f"ERROR: Input video not found at '{INPUT_VIDEO_PATH}'.")
        print("Please ensure you run the embedding script first to create the stego video.")
        return

    print(f"\n--- Starting Extraction Test ---")
    print(f"Input Video: {INPUT_VIDEO_PATH}")
    print(f"Output Image: {output_image_path}")

    # --- Run Extraction ---
    try:
        # Call the extraction function defined above
        extract_spatial_lsb(str(INPUT_VIDEO_PATH), str(output_image_path))
        
        print("\n✅ Extraction Test COMPLETE.")

    except Exception as e:
        print(f"\n❌ Extraction Test FAILED. Error: {e}")
        print("This usually means the video is too short or the LSBs were corrupted.")

if __name__ == "__main__":
    run_single_extraction_test()