import cv2
import numpy as np
from tqdm import tqdm

from stega_lib.bit_utils import img_to_bits, create_header
from stega_lib.lsb import embed_bits_in_lsb, extract_bits_from_lsb

def embed_temporal(base_video_path: str, secret_image_path: str, output_video_path: str):
    """
    Embeds a secret image into the LSBs of the *difference*
    between consecutive frames.
    
    f_new[n] = f_new[n-1] + modified_diff(f[n], f_new[n-1])
    """
    # 1. Prepare secret data
    secret_img = cv2.imread(secret_image_path)
    if secret_img is None:
        raise FileNotFoundError(f"Secret image not found at {secret_image_path}")
        
    data_bits, shape = img_to_bits(secret_img)
    header_bits = create_header(shape, len(data_bits))
    
    # 2. Setup video streams
    cap = cv2.VideoCapture(base_video_path)
    if not cap.isOpened():
        raise IOError(f"Cannot open base video: {base_video_path}")

    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter(output_video_path, fourcc, fps, (frame_width, frame_height))

    # 3. Perform embedding
    bit_idx = 0
    frame_idx = 0
    prev_mod_frame = None
    
    pbar = tqdm(total=total_frames, desc="Embedding Temporal")
    
    while cap.isOpened():
        ret, current_orig_frame = cap.read()
        if not ret:
            break
        
        pbar.update(1)

        if frame_idx == 0:
            # Embed header into the LSBs of the first frame (spatially)
            header_capacity = current_orig_frame.size
            if len(header_bits) > header_capacity:
                raise ValueError("Frame 0 is not large enough to hold header.")
                
            mod_frame = embed_bits_in_lsb(current_orig_frame, header_bits)
            writer.write(mod_frame)
            prev_mod_frame = mod_frame
            frame_idx += 1
            continue
            
        if bit_idx >= len(data_bits):
            # All data embedded, just write remaining frames
            writer.write(current_orig_frame)
            prev_mod_frame = current_orig_frame # Keep track for next diff
            frame_idx += 1
            continue

        # Calculate frame difference using 16-bit signed integers to avoid clipping
        diff = current_orig_frame.astype(np.int16) - prev_mod_frame.astype(np.int16)
        
        capacity_per_frame = diff.size
        bits_to_embed = min(capacity_per_frame, len(data_bits) - bit_idx)
        
        if bits_to_embed > 0:
            bit_chunk = data_bits[bit_idx : bit_idx + bits_to_embed]
            
            # Embed bits into the LSBs of the difference array
            mod_diff = embed_bits_in_lsb(diff, bit_chunk)
            
            # Reconstruct the new frame by adding the modified diff
            # to the *previous modified* frame.
            new_frame_s16 = prev_mod_frame.astype(np.int16) + mod_diff
            
            # Clip back to 8-bit unsigned range
            mod_frame = np.clip(new_frame_s16, 0, 255).astype(np.uint8)
            
            writer.write(mod_frame)
            prev_mod_frame = mod_frame
            bit_idx += bits_to_embed
        else:
            writer.write(current_orig_frame)
            prev_mod_frame = current_orig_frame
            
        frame_idx += 1

    pbar.close()
    cap.release()
    writer.release()
    
    if bit_idx < len(data_bits):
        print(f"Warning: Video ended before all data could be embedded.")
        print(f"Embedded {bit_idx} / {len(data_bits)} bits.")