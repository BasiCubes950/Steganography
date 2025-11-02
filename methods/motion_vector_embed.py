import cv2
import numpy as np
import time
import os

def embed_message_in_motion_vectors(video_path, output_path, message):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise FileNotFoundError(f"Cannot open video: {video_path}")

    fourcc = cv2.VideoWriter_fourcc(*'MJPG')
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    out = cv2.VideoWriter(output_path, fourcc, fps, (w, h))

    # Read first frame
    ret, prev_frame = cap.read()
    if not ret:
        raise ValueError("Could not read first frame.")
    prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)

    bits = np.unpackbits(np.frombuffer(message.encode('utf-8'), dtype=np.uint8))
    bit_index = 0
    total_bits = len(bits)

    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_num = 1

    print(f"Embedding {total_bits} bits into {frame_count} frames...")

    while True:
        ret, curr_frame = cap.read()
        if not ret:
            break

        frame_num += 1
        curr_gray = cv2.cvtColor(curr_frame, cv2.COLOR_BGR2GRAY)

        # ↓↓ Optical flow is the expensive step ↓↓
        flow = cv2.calcOpticalFlowFarneback(
            prev_gray, curr_gray, None,
            0.5, 3, 15, 3, 5, 1.2, 0
        )

        magnitude, angle = cv2.cartToPolar(flow[..., 0], flow[..., 1])
        mask = magnitude > 1  # Only modify areas with noticeable motion

        # Embed bits sparsely (e.g. every 50th valid pixel)
        coords = list(zip(*np.where(mask)))
        coords = coords[::50]  # skip to reduce load

        for (y, x) in coords:
            if bit_index >= total_bits:
                break
            mag = magnitude[y, x]
            magnitude[y, x] = mag * (1.10 if bits[bit_index] else 0.90)  # Stronger perturbation
            bit_index += 1

        fx, fy = cv2.polarToCart(magnitude, angle)
        flow_embedded = np.dstack((fx, fy))

        # Warp frame
        h_grid, w_grid = np.meshgrid(np.arange(w), np.arange(h))
        map_x = (w_grid + flow_embedded[..., 0]).astype(np.float32)
        map_y = (h_grid + flow_embedded[..., 1]).astype(np.float32)
        warped = cv2.remap(curr_frame, map_x, map_y, cv2.INTER_LINEAR)

        out.write(warped)
        prev_gray = curr_gray

        # Print progress
        if frame_num % 10 == 0:
            print(f"Processed frame {frame_num}/{frame_count} ({bit_index}/{total_bits} bits)")

        # Stop early if message is done
        if bit_index >= total_bits:
            print("All bits embedded. Finishing video...")
            break

    cap.release()
    out.release()
    print(f"✅ Finished embedding ({bit_index}/{total_bits} bits). Saved to {output_path}")


def extract_message_from_motion_vectors(stego_path, original_path, message_length):
    cap_s = cv2.VideoCapture(stego_path)
    cap_o = cv2.VideoCapture(original_path)
    bits = []

    ret_s, prev_s = cap_s.read()
    ret_o, prev_o = cap_o.read()
    prev_gray_s = cv2.cvtColor(prev_s, cv2.COLOR_BGR2GRAY)
    prev_gray_o = cv2.cvtColor(prev_o, cv2.COLOR_BGR2GRAY)

    frame_count = int(cap_s.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_num = 1
    total_bits = message_length * 8

    print("Extracting message...")

    while True:
        ret_s, curr_s = cap_s.read()
        ret_o, curr_o = cap_o.read()
        if not (ret_s and ret_o):
            break

        frame_num += 1
        curr_gray_s = cv2.cvtColor(curr_s, cv2.COLOR_BGR2GRAY)
        curr_gray_o = cv2.cvtColor(curr_o, cv2.COLOR_BGR2GRAY)

        flow_s = cv2.calcOpticalFlowFarneback(prev_gray_s, curr_gray_s, None, 0.5, 3, 15, 3, 5, 1.2, 0)
        flow_o = cv2.calcOpticalFlowFarneback(prev_gray_o, curr_gray_o, None, 0.5, 3, 15, 3, 5, 1.2, 0)

        mag_s, _ = cv2.cartToPolar(flow_s[..., 0], flow_s[..., 1])
        mag_o, _ = cv2.cartToPolar(flow_o[..., 0], flow_o[..., 1])

        mask = mag_o > 1
        diff = (mag_s - mag_o) / (mag_o + 1e-6)
        bits += list((diff[mask] > 0.05).astype(np.uint8))  # threshold for clear distinction

        if len(bits) >= total_bits:
            break

        prev_gray_s, prev_gray_o = curr_gray_s, curr_gray_o

        if frame_num % 10 == 0:
            print(f"Extracted bits: {len(bits)}/{total_bits}")

    cap_s.release()
    cap_o.release()

        # Rebuild message
    bits = np.array(bits[:message_length * 8], dtype=np.uint8)
    bytes_arr = np.packbits(bits)
    return bytes_arr.tobytes().decode('utf-8', errors='ignore')

def main():
    input_path = "data/data1.mp4"
    output_path = "Results/motion_vector_output.avi"
    secret_message = "Hidden message!"
    message_length = len(secret_message)

    print("Embedding message...")
    start = time.time()
    embed_message_in_motion_vectors(input_path, output_path, secret_message)
    print(f"Embedding done in {time.time() - start:.2f} sec")

    recovered = extract_message_from_motion_vectors(output_path, input_path, message_length)
    print(f"Recovered message: {recovered}")


main()