import cv2
import numpy as np
from tqdm import tqdm

from stega_lib.bit_utils import parse_header, bits_to_img
from stega_lib.lsb import extract_bits_from_lsb

# Must be the same scale factor used in embedding
FLOW_SCALE = 10.0

def extract_motion_vector(stego_video_path: str, output_image_path: str):
    """
    Extracts a secret image from the LSBs of simulated optical flow vectors.
    """
    cap = cv2.VideoCapture(stego_video_path)
    if not cap.isOpened():
        raise IOError(f"Cannot open stego video: {stego_video_path}")

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total_frames == 0:
        raise IOError(f"Video file is empty or corrupted: {stego_video_path}")

    # 1. Read first frame and extract header
    ret, frame0 = cap.read()
    if not ret:
        raise ValueError("Cannot read first frame to extract header.")
        
    try:
        header_bits = extract_bits_from_lsb(frame0, 128)
        shape, total_data_bits = parse_header(header_bits)
    except Exception as e:
        print(f"Error parsing header, extraction will likely fail: {e}")
        shape, total_data_bits = (100, 100, 3), 1000
    
    total_bits_to_extract = total_data_bits
    
    # 2. Continue extracting data bits from flow vectors
    all_data_bits = []
    bits_extracted = 0
    
    prev_frame = frame0
    prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
    
    pbar = tqdm(total=total_bits_to_extract, desc="Extracting Motion Vector")
    
    frame_idx = 1
    while cap.isOpened() and bits_extracted < total_bits_to_extract:
        ret, current_frame = cap.read()
        if not ret:
            break
            
        current_gray = cv2.cvtColor(current_frame, cv2.COLOR_BGR2GRAY)
            
        # Calculate optical flow. This will be noisy due to recompression!
        flow = cv2.calcOpticalFlowFarneback(
            prev_gray, current_gray, None, 0.5, 3, 15, 3, 5, 1.2, 0
        )
        
        # Quantize flow vectors
        flow_quant = np.round(flow * FLOW_SCALE).astype(np.int16)
        
        capacity_per_frame = flow_quant.size
        bits_to_get = min(capacity_per_frame, total_bits_to_extract - bits_extracted)
        
        if bits_to_get > 0:
            frame_bits = extract_bits_from_lsb(flow_quant, bits_to_get)
            all_data_bits.append(frame_bits)
            bits_extracted += bits_to_get
            pbar.update(bits_to_get)
        else:
            break
            
        prev_frame = current_frame
        prev_gray = current_gray
        frame_idx += 1
        
    pbar.close()
    cap.release()
    
    if not all_data_bits:
         print("Warning: No data bits were extracted.")
         final_bitstream = np.array([], dtype=np.uint8)
    else:
        final_bitstream = np.concatenate(all_data_bits)

    # 3. Reconstruct image
    try:
        img = bits_to_img(final_bitstream, shape)
        cv2.imwrite(output_image_path, img)
    except Exception as e:
        print(f"Failed to reconstruct or save image: {e}")
        black_img = np.zeros(shape, dtype=np.uint8)
        cv2.imwrite(output_image_path, black_img)