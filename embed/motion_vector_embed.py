import cv2
import numpy as np
from tqdm import tqdm

from stega_lib.bit_utils import img_to_bits, create_header
from stega_lib.lsb import embed_bits_in_lsb

# Scale factor for flow vectors before LSB embedding
FLOW_SCALE = 10.0

def embed_motion_vector(base_video_path: str, secret_image_path: str, output_video_path: str):
    """
    Embeds a secret image by modifying simulated optical flow (motion) vectors
    and re-synthesizing frames.
    
    f_new[n] = warp(f_new[n-1], modified_flow(f[n], f_new[n-1]))
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
    writer = cv2.VideoWriter(output_video_path, fourcc, (frame_width, frame_height))
    
    # Create coordinate grid for remapping
    grid_y, grid_x = np.mgrid[0:frame_height, 0:frame_width]
    grid_x = grid_x.astype(np.float32)
    grid_y = grid_y.astype(np.float32)

    # 3. Perform embedding
    bit_idx = 0
    frame_idx = 0
    prev_mod_frame = None
    prev_mod_gray = None
    
    pbar = tqdm(total=total_frames, desc="Embedding Motion Vector")
    
    while cap.isOpened():
        ret, current_orig_frame = cap.read()
        if not ret:
            break
            
        pbar.update(1)

        if frame_idx == 0:
            # Embed header into the LSBs of the first frame (spatially)
            mod_frame = embed_bits_in_lsb(current_orig_frame, header_bits)
            writer.write(mod_frame)
            prev_mod_frame = mod_frame
            prev_mod_gray = cv2.cvtColor(prev_mod_frame, cv2.COLOR_BGR2GRAY)
            frame_idx += 1
            continue
            
        if bit_idx >= len(data_bits):
            # All data embedded, just write remaining frames
            writer.write(current_orig_frame)
            prev_mod_frame = current_orig_frame
            prev_mod_gray = cv2.cvtColor(prev_mod_frame, cv2.COLOR_BGR2GRAY)
            frame_idx += 1
            continue

        # Calculate optical flow based on *original* frame and *previous modified* frame
        current_orig_gray = cv2.cvtColor(current_orig_frame, cv2.COLOR_BGR2GRAY)
        
        flow = cv2.calcOpticalFlowFarneback(
            prev_mod_gray, current_orig_gray, None, 0.5, 3, 15, 3, 5, 1.2, 0
        )
        
        # Quantize flow: Multiply by scale, round, and cast to int16
        flow_quant = np.round(flow * FLOW_SCALE).astype(np.int16)
        
        capacity_per_frame = flow_quant.size
        bits_to_embed = min(capacity_per_frame, len(data_bits) - bit_idx)
        
        if bits_to_embed > 0:
            bit_chunk = data_bits[bit_idx : bit_idx + bits_to_embed]
            
            # Embed bits into the LSBs of the quantized flow vectors
            mod_flow_quant = embed_bits_in_lsb(flow_quant, bit_chunk)
            
            # De-quantize to get modified flow
            mod_flow = mod_flow_quant.astype(np.float32) / FLOW_SCALE
            
            # Re-synthesize the new frame by warping the previous modified frame
            # using the modified flow
            map_x = grid_x + mod_flow[..., 0]
            map_y = grid_y + mod_flow[..., 1]
            
            mod_frame = cv2.remap(
                prev_mod_frame, map_x, map_y, 
                interpolation=cv2.INTER_LINEAR, 
                borderMode=cv2.BORDER_REPLICATE
            )
            
            writer.write(mod_frame)
            prev_mod_frame = mod_frame
            prev_mod_gray = cv2.cvtColor(mod_frame, cv2.COLOR_BGR2GRAY)
            bit_idx += bits_to_embed
        else:
            writer.write(current_orig_frame)
            prev_mod_frame = current_orig_frame
            prev_mod_gray = cv2.cvtColor(current_orig_frame, cv2.COLOR_BGR2GRAY)
            
        frame_idx += 1

    pbar.close()
    cap.release()
    writer.release()
    
    if bit_idx < len(data_bits):
        print(f"Warning: Video ended before all data could be embedded.")
        print(f"Embedded {bit_idx} / {len(data_bits)} bits.")