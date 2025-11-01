import cv2
import numpy as np


def embed_spatial_lsb(video_path, data_bits, output_path):
    cap = cv2.VideoCapture(video_path)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = None
    bit_idx = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if out is None:
            h, w, _ = frame.shape
            out = cv2.VideoWriter(output_path, fourcc, cap.get(cv2.CAP_PROP_FPS), (w, h))

        # Work with a flat view of the frame buffer. Ensure we operate on integers
        # and use an unsigned 8-bit mask (0xFE) instead of ~1 which yields -2 as
        # a Python int and can cause overflow errors when assigning back to uint8.
        flat = frame.flatten()
        for i in range(len(flat)):
            if bit_idx >= len(data_bits):
                break
            bit = int(data_bits[bit_idx]) & 1
            val = (int(flat[i]) & 0xFE) | bit
            flat[i] = np.uint8(val)
            bit_idx += 1

        frame = flat.reshape(frame.shape)
        out.write(frame)

    cap.release()
    out.release()


def extract_spatial_lsb(video_path, num_bits):
    cap = cv2.VideoCapture(video_path)
    bits = []
    while len(bits) < num_bits:
        ret, frame = cap.read()
        if not ret:
            break
        # Convert to list of ints then take LSBs to avoid numpy broadcasting quirks
        bits.extend(list((frame.flatten() & 1).tolist()))
    cap.release()
    return bits[:num_bits]