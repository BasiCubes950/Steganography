import cv2
import numpy as np

def embed_temporal(video_path, data_bits, output_path, frame_gap=2):
    cap = cv2.VideoCapture(video_path)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = None
    bit_idx = 0
    frame_id = 0

    prev_frame = None
    while True:
        ret, frame = cap.read()
        if not ret: break
        if out is None:
            h, w, _ = frame.shape
            out = cv2.VideoWriter(output_path, fourcc, cap.get(cv2.CAP_PROP_FPS), (w, h))

        if frame_id % frame_gap == 0 and prev_frame is not None and bit_idx < len(data_bits):
            diff = cv2.absdiff(frame, prev_frame)
            mask = diff > 5
            indices = np.where(mask[...,0])
            for idx in zip(*indices):
                if bit_idx >= len(data_bits): break
                frame[idx] = (frame[idx] & ~1) | int(data_bits[bit_idx])
                bit_idx += 1
        out.write(frame)
        prev_frame = frame.copy()
        frame_id += 1

    cap.release()
    out.release()

def extract_temporal(video_path, num_bits, frame_gap=2):
    cap = cv2.VideoCapture(video_path)
    bits = []
    prev_frame = None
    frame_id = 0

    while len(bits) < num_bits:
        ret, frame = cap.read()
        if not ret: break
        if frame_id % frame_gap == 0 and prev_frame is not None:
            diff = cv2.absdiff(frame, prev_frame)
            mask = diff > 5
            indices = np.where(mask[...,0])
            for idx in zip(*indices):
                if len(bits) >= num_bits: break
                bits.append(frame[idx][0] & 1)
        prev_frame = frame.copy()
        frame_id += 1

    cap.release()
    return bits[:num_bits]
