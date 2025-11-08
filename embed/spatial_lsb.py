import cv2
import numpy as np
from tqdm import tqdm

from stega_lib.bit_utils import img_to_bits, create_header
from stega_lib.lsb import embed_bits_in_lsb

def embed_spatial_lsb(base_video_path: str, secret_image_path: str, output_video_path: str):
    """
    Embeds a secret image into the spatial LSBs of a base video.
    """
    # 1. Prepare secret data
    secret_img = cv2.imread(secret_image_path)
    if secret_img is None:
        raise FileNotFoundError(f"Secret image not found at {secret_image_path}")
        
    data_bits, shape = img_to_bits(secret_img)
    header_bits = create_header(shape, len(data_bits))
    total_bitstream = np.concatenate((header_bits, data_bits))
    total_bits_to_embed = len(total_bitstream)

    # 2. Setup video streams
    cap = cv2.VideoCapture(base_video_path)
    if not cap.isOpened():
        raise IOError(f"Cannot open base video: {base_video_path}")

    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # Use 'mp4v' for .mp4, which is widely supported
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter(output_video_path, fourcc, fps, (frame_width, frame_height))

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
            # We need to pad the chunk if it's smaller than the frame's capacity
            padded_chunk = np.zeros(capacity_per_frame, dtype=np.uint8)
            padded_chunk[:bits_to_embed] = bit_chunk
            
            # Embed by modifying a *copy* of the frame's flat representation
            flat_frame = frame.flatten()
            flat_frame[:bits_to_embed] &= 0xFE
            flat_frame[:bits_to_embed] |= bit_chunk
            
            mod_frame = flat_frame.reshape(frame.shape)
            writer.write(mod_frame)
            
            bit_idx += bits_to_embed
        else:
            # This case should be covered by the (bit_idx >= total_bits_to_embed) check,
            # but as a fallback, write the original frame.
            writer.write(frame)
            
        frame_idx += 1

    pbar.close()
    cap.release()
    writer.release()
    
    if bit_idx < total_bits_to_embed:
        print(f"Warning: Video ended before all data could be embedded.")
        print(f"Embedded {bit_idx} / {total_bits_to_embed} bits.")