import cv2
import numpy as np
from tqdm import tqdm
from pathlib import Path
import os

# --- NOTE: The following imports must be available in your local environment ---
# from stega_lib.bit_utils import img_to_bits, create_header
# from stega_lib.lsb import embed_bits_in_lsb 

# You must provide definitions for img_to_bits and create_header for this to run
# if they are not in your stega_lib. Since this is a self-contained file, 
# I will assume those functions are available when you run this script.
# If you get a 'ModuleNotFoundError' you must ensure stega_lib is in your path.

def embed_spatial_lsb(base_video_path: str, secret_image_path: str, output_video_path: str):
    """
    Embeds a secret image into the spatial LSBs of a base video.
    """
    # 1. Prepare secret data
    secret_img = cv2.imread(secret_image_path)
    if secret_img is None:
        raise FileNotFoundError(f"Secret image not found at {secret_image_path}")
        
    # --- TEMPORARY MOCK FOR MISSING LIBRARY IMPORTS ---
    # Since stega_lib is not provided, we mock the required data/shape
    # For a 1920x1080 image, the bits needed are ~49.7 million.
    try:
        # Attempt to use real library functions if available
        # NOTE: This will fail if stega_lib is not set up correctly
        from stega_lib.bit_utils import img_to_bits, create_header
        data_bits, shape = img_to_bits(secret_img)
        header_bits = create_header(shape, len(data_bits))
        total_bitstream = np.concatenate((header_bits, data_bits))
        total_bits_to_embed = len(total_bitstream)
    except ImportError:
        # Fallback/Mock implementation to allow testing the core logic
        print("Warning: stega_lib helper functions not found. Using placeholder bitstream.")
        # Mock shape: (1080, 1920, 3)
        mock_data_len = secret_img.size * 8
        mock_header_len = 128 # Must match HEADER_SIZE_BITS in extraction
        total_bits_to_embed = mock_data_len + mock_header_len
        
        # Create mock data (random bits) to satisfy the rest of the script
        total_bitstream = np.random.randint(0, 2, size=total_bits_to_embed, dtype=np.uint8)
        # -------------------------------------------------------------------
        

    # 2. Setup video streams
    cap = cv2.VideoCapture(base_video_path)
    if not cap.isOpened():
        raise IOError(f"Cannot open base video: {base_video_path}")

    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # Try different codecs in order of preference
    # 'mp4v' is a common safe fallback if 'avc1' or 'x264' fail
    codecs = ['avc1', 'mp4v', 'x264']
    writer = None
    
    # Validate fps
    if fps is None or fps <= 0:
        # Default to 30 FPS if invalid
        fps = 30.0
        print(f"Warning: Invalid FPS read, defaulting to {fps} FPS.")
        
    for codec in codecs:
        try:
            fourcc = cv2.VideoWriter_fourcc(*codec)
            writer = cv2.VideoWriter(output_video_path, fourcc, fps, (frame_width, frame_height))
            if writer.isOpened():
                print(f"Successfully opened VideoWriter with codec: {codec}")
                break
        except Exception as e:
            print(f"Failed to initialize codec {codec}: {e}")
            if writer:
                writer.release()
                
    if writer is None or not writer.isOpened():
        raise IOError(f"Could not open VideoWriter for output: {output_video_path} with any available codec")

    # 3. Perform embedding
    bit_idx = 0
    frame_idx = 0
    
    pbar = tqdm(total=total_frames, desc="Embedding Spatial LSB")
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
            
        pbar.update(1)
        
        if bit_idx >= total_bits_to_embed:
            # All data embedded, just write remaining frames
            writer.write(frame)
            continue
            
        # Calculate how many bits we can/need to write in this frame
        capacity_per_frame = frame.size  # H * W * 3
        bits_to_embed = min(capacity_per_frame, total_bits_to_embed - bit_idx)
        
        if bits_to_embed > 0:
            # Get the chunk of bits to embed
            bit_chunk = total_bitstream[bit_idx : bit_idx + bits_to_embed]
            
            # Embed the bit chunk into the frame
            
            # Embed by modifying a *copy* of the frame's flat representation
            flat_frame = frame.flatten()
            
            # The actual LSB manipulation logic:
            flat_frame[:bits_to_embed] &= 0xFE # Clears the LSB
            flat_frame[:bits_to_embed] |= bit_chunk # Sets the new LSB
            
            mod_frame = flat_frame.reshape(frame.shape)
            writer.write(mod_frame)
            
            bit_idx += bits_to_embed
        else:
            # Should be covered above, but safe to include
            writer.write(frame)
            
        frame_idx += 1

    pbar.close()
    cap.release()
    writer.release()
    
    if bit_idx < total_bits_to_embed:
        print(f"Warning: Video ended before all data could be embedded.")
        print(f"Embedded {bit_idx} / {total_bits_to_embed} bits.")
    else:
        print(f"✅ Embedding complete. Total bits written: {total_bits_to_embed}.")


# --- RUNNER SCRIPT ---

if __name__ == "__main__":
    
    # --- CONFIGURATION ---
    # !! UPDATE THESE PATHS TO YOUR LOCAL FILES !!
    BASE_VIDEO_PATH = Path("data/data2.mp4") 
    SECRET_IMAGE_PATH = Path("data/secret.png") 
    OUTPUT_FOLDER = Path("output")
    OUTPUT_VIDEO_NAME = "spatial_lsb_stego.mp4"
    
    OUTPUT_VIDEO_PATH = OUTPUT_FOLDER / OUTPUT_VIDEO_NAME

    # --- SETUP ---
    OUTPUT_FOLDER.mkdir(parents=True, exist_ok=True)
    
    if not BASE_VIDEO_PATH.exists():
        print(f"ERROR: Base video not found at '{BASE_VIDEO_PATH}'.")
        print("Please check the BASE_VIDEO_PATH setting.")
    elif not SECRET_IMAGE_PATH.exists():
        print(f"ERROR: Secret image not found at '{SECRET_IMAGE_PATH}'.")
        print("Please check the SECRET_IMAGE_PATH setting.")
    else:
        print("\n--- Starting Spatial LSB Embedding ---")
        print(f"Base Video: {BASE_VIDEO_PATH}")
        print(f"Secret Image: {SECRET_IMAGE_PATH}")
        print(f"Output Video: {OUTPUT_VIDEO_PATH}")

        # --- EXECUTION ---
        try:
            embed_spatial_lsb(str(BASE_VIDEO_PATH), str(SECRET_IMAGE_PATH), str(OUTPUT_VIDEO_PATH))
            print("\n✅ Embedding Script Finished Successfully.")
        except IOError as e:
            print(f"\n❌ Embedding Failed (I/O Error): {e}")
            print("Check file paths, permissions, and video codec availability.")
        except ValueError as e:
            print(f"\n❌ Embedding Failed (Data Error): {e}")
        except Exception as e:
            print(f"\n❌ Embedding Failed: An unexpected error occurred: {e}")