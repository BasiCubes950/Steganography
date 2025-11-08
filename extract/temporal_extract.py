"""
Temporal Steganography Extraction Script (Placeholder)

Extracts a secret image using a simplified temporal LSB method.
NOTE: The actual logic (frame_interval, bit extraction method) MUST be
synchronized with the corresponding 'temporal_embed' script.
"""
import cv2
import numpy as np
from pathlib import Path
import sys

def extract_temporal(video_path: str, output_image_path: str):
    """
    Extracts a secret image using a temporal steganography method (e.g.,
    LSB from every Nth frame, or spread across frames).

    NOTE: The logic here is a placeholder and MUST match the 'temporal_embed' logic.
    """
    print(f"Starting temporal extraction from: {video_path}")
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise IOError(f"Cannot open video file: {video_path}")

    # --- CONFIGURATION (MUST MATCH EMBEDDER) ---
    frame_interval = 3
    # Assuming a fixed, small secret image size for this placeholder.   1080, 1920
    extracted_width, extracted_height, extracted_channels = 1920, 1080, 3
    total_bits_needed = extracted_width * extracted_height * extracted_channels * 8
    # -------------------------------------------
    
    print(f"Attempting to extract {total_bits_needed} bits (128x128x3 image)...")
    
    extracted_bits = []
    frame_count = 0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    while len(extracted_bits) < total_bits_needed:
        ret, frame = cap.read()
        if not ret:
            break
            
        if frame_count % frame_interval == 0:
            # Placeholder: extract ALL LSBs from the frame and append
            # WARNING: This uses the entire frame capacity in one go.
            frame_flat = frame.flatten()
            
            # The actual LSB extraction logic:
            frame_lsb_bits = frame_flat & 1 
            extracted_bits.extend(frame_lsb_bits.tolist())
            
            sys.stdout.write(f"\rProcessing frame {frame_count}/{total_frames}, extracted {len(extracted_bits)} bits")
            sys.stdout.flush()
            
        frame_count += 1
        
    cap.release()
    sys.stdout.write("\n") # Newline after progress bar

    # --- Processing Extracted Bits ---
    extracted_bits = extracted_bits[:total_bits_needed]
    
    if len(extracted_bits) < total_bits_needed:
        print(f"Warning: Could not extract all bits. Found {len(extracted_bits)}, needed {total_bits_needed}.")
        # Pad with zeros if extraction failed early
        extracted_bits.extend([0] * (total_bits_needed - len(extracted_bits)))

    try:
        extracted_bytes = np.packbits(np.array(extracted_bits, dtype=np.uint8))
        
        # Ensure we only use the exact number of bytes needed for the reshape
        expected_bytes = extracted_width * extracted_height * extracted_channels
        
        extracted_image = extracted_bytes[:expected_bytes].reshape(
            extracted_height, extracted_width, extracted_channels
        ).astype(np.uint8)
        
        # Save extracted image
        output_path = Path(output_image_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(output_path), extracted_image)
        
        print(f"Successfully extracted temporal image to: {output_image_path}")
        return output_image_path
        
    except ValueError as e:
        print(f"Error during bit-to-image conversion or reshaping: {e}")
        return None


# --- RUNNER SCRIPT ---

if __name__ == "__main__":
    
    # Placeholder paths for testing (update these if you use real files)
    STEGO_VIDEO_PATH = Path("output/stego/temporal_stego.mp4") 
    OUTPUT_IMAGE_PATH = Path("output/extracted/extracted_temporal_image.png")

    if not STEGO_VIDEO_PATH.exists():
        print(f"ERROR: Placeholder stego video not found at '{STEGO_VIDEO_PATH}'.")
        print("Please update STEGO_VIDEO_PATH to a valid video file to test extraction.")
    else:
        try:
            extract_temporal(str(STEGO_VIDEO_PATH), str(OUTPUT_IMAGE_PATH))
            print("\n✅ Temporal Extraction Test Finished.")
        except IOError as e:
            print(f"\n❌ Extraction Failed (I/O Error): {e}")
        except Exception as e:
            print(f"\n❌ Extraction Failed: An unexpected error occurred: {e}")