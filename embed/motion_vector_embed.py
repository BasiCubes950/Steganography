"""
Motion-Vector Based Embedding
Embeds secret image by modifying quantized optical flow vectors using LSB embedding,
then re-synthesizes frames via warping.
"""
import cv2
import numpy as np
from pathlib import Path
import os
import sys

# Scale factor for flow vectors before LSB embedding
FLOW_SCALE = 10.0
# Header size constant
HEADER_SIZE_BITS = 72  # 9 bytes for H, W, C, and Data Length (4 bytes)


def img_to_bits(img: np.ndarray):
    """Convert image to bit array and return shape info."""
    flat = img.flatten()
    bits = np.unpackbits(flat)
    return bits, img.shape


def create_header(shape: tuple, data_length: int):
    """Create a simple header with shape and data length info."""
    # Store: height (16 bits), width (16 bits), channels (8 bits), data_length (32 bits)
    h, w, c = shape
    
    # Check if image dimensions fit in 16 bits
    if h > 65535 or w > 65535:
        raise ValueError("Image dimensions exceed 16-bit capacity for header.")
    
    header = np.array([
        (h >> 8) & 0xFF, h & 0xFF,
        (w >> 8) & 0xFF, w & 0xFF,
        c & 0xFF,
        (data_length >> 24) & 0xFF,
        (data_length >> 16) & 0xFF,
        (data_length >> 8) & 0xFF,
        data_length & 0xFF
    ], dtype=np.uint8)
    return np.unpackbits(header)


def parse_header(header_bits: np.ndarray):
    """Parse header to extract shape and data length."""
    if len(header_bits) < HEADER_SIZE_BITS:
        # This function is primarily for validation/testing, not core embedding, 
        # but good to keep it robust.
        raise ValueError(f"Header bits too short. Expected {HEADER_SIZE_BITS}, got {len(header_bits)}")
        
    header_bytes = np.packbits(header_bits[:HEADER_SIZE_BITS])
    
    h = (int(header_bytes[0]) << 8) | int(header_bytes[1])
    w = (int(header_bytes[2]) << 8) | int(header_bytes[3])
    c = int(header_bytes[4])
    data_length = (int(header_bytes[5]) << 24) | (int(header_bytes[6]) << 16) | \
                  (int(header_bytes[7]) << 8) | int(header_bytes[8])
                  
    return (h, w, c), data_length


def embed_bits_in_lsb(carrier: np.ndarray, bits: np.ndarray):
    """Embed bits into the LSBs of carrier array (safe for int16 or uint8)."""
    carrier_flat = carrier.flatten().copy()

    # Ensure bits are uint8 (0 or 1)
    bits = bits.astype(np.uint8)

    bits_to_embed = min(len(bits), len(carrier_flat))

    # Perform operation safely using bitwise masking
    # Convert to int32 to avoid overflow/underflow when carrier is signed
    temp = carrier_flat[:bits_to_embed].astype(np.int32)
    temp = (temp & ~1) | bits[:bits_to_embed]

    # Cast back to original type (e.g. int16 or uint8)
    carrier_flat[:bits_to_embed] = temp.astype(carrier_flat.dtype)

    return carrier_flat.reshape(carrier.shape)



def extract_bits_from_lsb(carrier: np.ndarray, num_bits: int):
    """Extract bits from the LSBs of carrier array."""
    carrier_flat = carrier.flatten()
    extracted = []
    
    # We only care about the LSB of the first 'num_bits' worth of elements
    for i in range(min(num_bits, len(carrier_flat))):
        extracted.append(carrier_flat[i] & 1)
    
    return np.array(extracted, dtype=np.uint8)


def embed_motion_vector(video_path: str, secret_image_path: str, output_path: str):
    """
    Embed secret image by modifying simulated optical flow (motion) vectors
    and re-synthesizing frames using warping.
    """
    print(f"Starting motion vector embedding...")
    
    # 1. Prepare secret data
    secret_img = cv2.imread(str(secret_image_path))
    if secret_img is None:
        raise ValueError(f"Cannot read secret image: {secret_image_path}")
    
    data_bits, shape = img_to_bits(secret_img)
    header_bits = create_header(shape, len(data_bits))
    total_bitstream = np.concatenate((header_bits, data_bits))
    total_bits_to_embed = len(total_bitstream)
    
    print(f"Secret image shape: {shape}")
    print(f"Total data bits (including header): {total_bits_to_embed}")
    
    # 2. Setup video streams
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise ValueError(f"Cannot open video: {video_path}")
    
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    if fps is None or fps <= 0:
        fps = 30.0
        print(f"Warning: Invalid FPS read, defaulting to {fps} FPS.")
    
    print(f"Video: {frame_width}x{frame_height}, {fps} fps, {total_frames} frames")
    
    # Create output directory
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    
    # Setup video writer (Using 'mp4v' as a common codec)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter(str(output_path), fourcc, fps, (frame_width, frame_height))
    
    if not writer.isOpened():
        raise IOError(f"Could not open VideoWriter: {output_path}. Try changing the codec.")
    
    # Create coordinate grid for remapping (needed for frame synthesis)
    grid_y, grid_x = np.mgrid[0:frame_height, 0:frame_width]
    grid_x = grid_x.astype(np.float32)
    grid_y = grid_y.astype(np.float32)
    
    # 3. Perform embedding
    bit_idx = 0
    frame_idx = 0
    prev_mod_frame = None
    prev_mod_gray = None
    
    print(f"Embedding data into motion vectors...")
    
    while cap.isOpened():
        ret, current_orig_frame = cap.read()
        if not ret:
            break
        
        # --- FRAME 1: Embed Header Spatially ---
        if frame_idx == 0:
            # Embed header into the LSBs of the first frame's spatial data
            # NOTE: We use the spatial LSB method here, which must match the extraction logic
            mod_frame = embed_bits_in_lsb(current_orig_frame, header_bits)
            
            writer.write(mod_frame)
            prev_mod_frame = mod_frame
            prev_mod_gray = cv2.cvtColor(prev_mod_frame, cv2.COLOR_BGR2GRAY)
            frame_idx += 1
            continue
        
        # --- Subsequent Frames: Embed Data into Flow Vectors ---
        
        if bit_idx >= len(data_bits):
            # All data embedded, write remaining frames unchanged
            writer.write(current_orig_frame)
            
            # Update previous frame references to maintain continuity
            prev_mod_frame = current_orig_frame
            prev_mod_gray = cv2.cvtColor(prev_mod_frame, cv2.COLOR_BGR2GRAY)
            frame_idx += 1
            continue
        
        # Calculate optical flow between previous MODIFIED frame and current ORIGINAL frame
        current_orig_gray = cv2.cvtColor(current_orig_frame, cv2.COLOR_BGR2GRAY)
        
        flow = cv2.calcOpticalFlowFarneback(
            prev_mod_gray, current_orig_gray, None,
            pyr_scale=0.5, levels=3, winsize=15,
            iterations=3, poly_n=5, poly_sigma=1.2, flags=0
        )
        
        # Quantize flow: multiply by scale, round, and cast to int16 (required for LSB)
        flow_quant = np.round(flow * FLOW_SCALE).astype(np.int16)
        
        # Calculate capacity and bits to embed in this frame
        # flow_quant is H x W x 2 (UV flow components)
        capacity_per_frame = flow_quant.size
        
        # We are only embedding the main data, skipping the header bits here (bit_idx vs total_bitstream)
        bits_remaining = len(data_bits) - bit_idx
        bits_to_embed = min(capacity_per_frame, bits_remaining)
        
        if bits_to_embed > 0:
            bit_chunk = data_bits[bit_idx:bit_idx + bits_to_embed]
            
            # Embed bits into the LSBs of quantized flow vectors
            mod_flow_quant = embed_bits_in_lsb(flow_quant, bit_chunk)
            
            # De-quantize to get modified flow (back to float32)
            mod_flow = mod_flow_quant.astype(np.float32) / FLOW_SCALE
            
            # Re-synthesize frame by warping previous modified frame using modified flow
            map_x = grid_x + mod_flow[..., 0]
            map_y = grid_y + mod_flow[..., 1]
            
            mod_frame_float = cv2.remap(
                prev_mod_frame, map_x, map_y,
                interpolation=cv2.INTER_LINEAR,
                borderMode=cv2.BORDER_REPLICATE
            )
            
            # --- FIX: Clip and convert synthesized frame to uint8 ---
            # Clip values to ensure they are within the [0, 255] range
            mod_frame = np.clip(mod_frame_float, 0, 255).astype(np.uint8)
            # -----------------------------------------------------

            writer.write(mod_frame)
            
            # The next flow calculation MUST use the newly synthesized frame
            prev_mod_frame = mod_frame
            prev_mod_gray = cv2.cvtColor(mod_frame, cv2.COLOR_BGR2GRAY)
            bit_idx += bits_to_embed
        else:
            # Should be unreachable if the outer loop condition is correct, but safe fallback
            writer.write(current_orig_frame)
            prev_mod_frame = current_orig_frame
            prev_mod_gray = cv2.cvtColor(current_orig_frame, cv2.COLOR_BGR2GRAY)
        
        frame_idx += 1
        
        if frame_idx % 20 == 0:
            sys.stdout.write(f"\rProcessing frame {frame_idx}/{total_frames}, embedded {bit_idx}/{len(data_bits)} bits")
            sys.stdout.flush()
    
    cap.release()
    writer.release()
    sys.stdout.write("\n") # Newline after progress bar
    
    if bit_idx < len(data_bits):
        print(f"Warning: Video ended before all data could be embedded.")
        print(f"Embedded {bit_idx}/{len(data_bits)} bits.")
    else:
        print(f"✅ Successfully embedded all {bit_idx} bits!")
    
    metadata = {
        'method': 'motion_vector',
        'bits_embedded': bit_idx,
        'total_bits': len(data_bits),
        'frames_used': frame_idx,
        'total_frames': total_frames,
        'flow_scale': FLOW_SCALE,
        'secret_shape': shape,
        'header_bits': len(header_bits)
    }
    
    print(f"Stego video saved to: {output_path}")
    
    return metadata

# --- RUNNER SCRIPT ---

if __name__ == "__main__":
    
    # --- CONFIGURATION ---
    # !! UPDATE THESE PATHS TO YOUR LOCAL FILES !!
    BASE_VIDEO_PATH = Path("data/data2.mp4") 
    SECRET_IMAGE_PATH = Path("data/secret.png") 

    OUTPUT_VIDEO_PATH = "output/stego/motion_vector_stego.mp4"
    
    if not BASE_VIDEO_PATH.exists():
        print(f"ERROR: Base video not found at '{BASE_VIDEO_PATH}'.")
        print("Please check the BASE_VIDEO_PATH setting.")
    elif not SECRET_IMAGE_PATH.exists():
        print(f"ERROR: Secret image not found at '{SECRET_IMAGE_PATH}'.")
        print("Please check the SECRET_IMAGE_PATH setting.")
    else:
        print("\n--- Starting Motion Vector Embedding Test ---")
        print(f"Base Video: {BASE_VIDEO_PATH}")
        print(f"Secret Image: {SECRET_IMAGE_PATH}")
        print(f"Output Video: {OUTPUT_VIDEO_PATH}")

        # --- EXECUTION ---
        try:
            metadata = embed_motion_vector(str(BASE_VIDEO_PATH), str(SECRET_IMAGE_PATH), str(OUTPUT_VIDEO_PATH))
            print("\n✅ Embedding Script Finished Successfully.")
            print("\nMetadata:")
            for key, value in metadata.items():
                print(f"  {key}: {value}")
        except ValueError as e:
            print(f"\n❌ Embedding Failed (Data Error): {e}")
        except IOError as e:
            print(f"\n❌ Embedding Failed (I/O Error): {e}")
            print("Check file paths, permissions, and ensure the 'mp4v' codec is supported on your system.")
        except Exception as e:
            print(f"\n❌ Embedding Failed: An unexpected error occurred: {e}")