import cv2
import numpy as np

def embed_motion_vector(video_path, data_bits, output_path):
    cap = cv2.VideoCapture(video_path)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = None
    bit_idx = 0
    ret, prev = cap.read()
    if not ret: return

    if out is None:
        h, w, _ = prev.shape
        out = cv2.VideoWriter(output_path, fourcc, cap.get(cv2.CAP_PROP_FPS), (w, h))

    out.write(prev)

    prev_gray = cv2.cvtColor(prev, cv2.COLOR_BGR2GRAY)
    while True:
        ret, frame = cap.read()
        if not ret: break
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        flow = cv2.calcOpticalFlowFarneback(prev_gray, gray, None, 0.5, 3, 15, 3, 5, 1.2, 0)
        mag, ang = cv2.cartToPolar(flow[...,0], flow[...,1])
        motion_mask = mag > 1.0
        indices = np.where(motion_mask)
        for idx in zip(*indices):
            if bit_idx >= len(data_bits): break
            frame[idx] = (frame[idx] & ~1) | int(data_bits[bit_idx])
            bit_idx += 1
        out.write(frame)
        prev_gray = gray.copy()

    cap.release()
    out.release()

def extract_motion_vector(video_path, num_bits):
    cap = cv2.VideoCapture(video_path)
    bits = []
    ret, prev = cap.read()
    if not ret: return bits
    prev_gray = cv2.cvtColor(prev, cv2.COLOR_BGR2GRAY)
    while len(bits) < num_bits:
        ret, frame = cap.read()
        if not ret: break
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        flow = cv2.calcOpticalFlowFarneback(prev_gray, gray, None, 0.5, 3, 15, 3, 5, 1.2, 0)
        mag, _ = cv2.cartToPolar(flow[...,0], flow[...,1])
        motion_mask = mag > 1.0
        indices = np.where(motion_mask)
        for idx in zip(*indices):
            if len(bits) >= num_bits: break
            bits.append(frame[idx][0] & 1)
        prev_gray = gray.copy()
    cap.release()
    return bits[:num_bits]
