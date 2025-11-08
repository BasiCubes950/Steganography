import cv2
import numpy as np
import time
import os
from utils.convert import image_to_bits, bits_to_image

def embed_message_temporally(video_path, output_path, message, intensity_shift=3):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise FileNotFoundError(f"Cannot open video: {video_path}")

    fourcc = cv2.VideoWriter_fourcc(*'MJPG')
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    out = cv2.VideoWriter(output_path, fourcc, fps, (w, h))

    # Convert message to bits
    bits = np.unpackbits(np.frombuffer(message.encode('utf-8'), dtype=np.uint8))
    total_bits = len(bits)
    bit_index = 0

    print(f"Embedding {total_bits} bits temporally...")

    ret, prev_frame = cap.read()
    if not ret:
        raise ValueError("No frames found in input video.")

    out.write(prev_frame)

    while True:
        ret, curr_frame = cap.read()
        if not ret:
            break

        if bit_index < total_bits:
            # Adjust brightness depending on bit (temporal difference)
            shift = intensity_shift if bits[bit_index] else -intensity_shift
            modified_frame = cv2.convertScaleAbs(curr_frame, alpha=1.0, beta=shift)
            bit_index += 1
        else:
            modified_frame = curr_frame

        out.write(modified_frame)
        prev_frame = curr_frame

    cap.release()
    out.release()
    print(f"✅ Temporal embedding complete ({bit_index}/{total_bits} bits used). Saved to {output_path}")

def extract_message_temporally(stego_path, original_path, message_length, intensity_shift=3, output_image_path=None, shape=None):
    cap_s = cv2.VideoCapture(stego_path)
    cap_o = cv2.VideoCapture(original_path)

    bits = []
    ret_s, prev_s = cap_s.read()
    ret_o, prev_o = cap_o.read()

    if not (ret_s and ret_o):
        raise ValueError("Could not read input videos.")

    while True:
        ret_s, curr_s = cap_s.read()
        ret_o, curr_o = cap_o.read()
        if not (ret_s and ret_o):
            break

        # Measure temporal brightness difference
        diff_s = np.mean(curr_s.astype(np.int16) - prev_s.astype(np.int16))
        diff_o = np.mean(curr_o.astype(np.int16) - prev_o.astype(np.int16))

        # Detect whether the brightness shifted up or down
        delta = diff_s - diff_o
        bits.append(1 if delta > 0 else 0)

        if len(bits) >= message_length * 8:
            break

        prev_s, prev_o = curr_s, curr_o

    cap_s.release()
    cap_o.release()

    bits = np.array(bits[:message_length * 8], dtype=np.uint8)

    # If caller requested image reconstruction, save the image and return None
    if output_image_path is not None and shape is not None:
        try:
            bits_to_image(bits, shape, output_image_path)
            return None
        except Exception as e:
            print(f"[ERROR] Could not reconstruct image: {e}")

    bytes_arr = np.packbits(bits)
    message = bytes_arr.tobytes().decode('utf-8', errors='ignore')
    return message

def main():
    input_path = "data/data1.mp4"
    output_path = "Results/temporal_output.avi"
    secret_message = "Secret message!"
    message_length = len(secret_message)

    start = time.time()
    print("Embedding message temporally...")
    embed_message_temporally(input_path, output_path, secret_message)
    print(f"Embedding done in {time.time() - start:.2f} sec")

    print("Extracting message...")
    recovered = extract_message_temporally(output_path, input_path, message_length)
    print(f"Recovered message: {recovered}")

main()
