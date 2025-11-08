import cv2
import numpy as np
from tqdm import tqdm

from stega_lib.bit_utils import parse_header, bits_to_img
from stega_lib.lsb import extract_bits_from_lsb

def extract_spatial_lsb(stego_video_path: str, output_image_path: str):
    """
    Extracts a secret image from the spatial LSBs of a stego video.
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
        # Default to a small, plausible size to avoid memory errors
        shape, total_data_bits = (100, 100, 3), 1000
    
    total_bits_to_extract = total_data_bits
    
    # 2. Continue extracting data bits
    all_data_bits = []
    bits_extracted = 0
    
    # We already "used" frame 0, but it also contains data bits after the header
    capacity_frame0 = frame0.size
    bits_in_frame0 = min(capacity_frame0 - 128, total_bits_to_extract)
    
    if bits_in_frame0 > 0:
        data_from_frame0 = extract_bits_from_lsb(frame0.flatten()[128:], bits_in_frame0)
        all_data_bits.append(data_from_frame0)
        bits_extracted += bits_in_frame0

    pbar = tqdm(total=total_bits_to_extract, desc="Extracting Spatial LSB")
    pbar.update(bits_extracted)
    
    frame_idx = 1
    while cap.isOpened() and bits_extracted < total_bits_to_extract:
        ret, frame = cap.read()
        if not ret:
            break
            
        capacity_per_frame = frame.size
        bits_to_get = min(capacity_per_frame, total_bits_to_extract - bits_extracted)
        
        if bits_to_get > 0:
            frame_bits = extract_bits_from_lsb(frame, bits_to_get)
            all_data_bits.append(frame_bits)
            bits_extracted += bits_to_get
            pbar.update(bits_to_get)
        else:
            # This happens if bits_to_get is 0 or negative (shouldn't occur)
            break
        
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
        # Save a black image as a failure token
        black_img = np.zeros(shape, dtype=np.uint8)
        cv2.imwrite(output_image_path, black_img)