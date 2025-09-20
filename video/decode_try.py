import cv2
import numpy as np
import os
from PIL import Image
import random
import argparse

def extract_hidden_image_from_frame(frame):
    """Extracts the hidden image from a single encoded video frame."""
    if frame is None:
        return None
    if frame.ndim == 3 and frame.shape[2] == 4:  # Handle BGRA frames
        frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Recover hidden 4 bits and shift back up
    r_hidden = (rgb[:, :, 0] & 0x0F) << 4
    g_hidden = (rgb[:, :, 1] & 0x0F) << 4
    b_hidden = (rgb[:, :, 2] & 0x0F) << 4

    return np.stack([r_hidden, g_hidden, b_hidden], axis=2).astype(np.uint8)


def recover_with_key(reconstructed, key):
    """Reverses the shuffle mapping using the given key to restore the secret image."""
    height, width = reconstructed.shape[:2]
    coords = [(x, y) for y in range(height) for x in range(width)]
    shuffled = coords.copy()
    random.seed(key)
    random.shuffle(shuffled)

    recovered = np.zeros_like(reconstructed)
    for dst, src in zip(coords, shuffled):
        dx, dy = dst
        sx, sy = src
        recovered[sy, sx, :] = reconstructed[dy, dx, :]
    return recovered


def extract_images_from_video(input_path, frame_indices, output_folder, extract_all=False, stego_keys=None):
    """Extracts hidden images from either a video file or a folder of PNGs."""
    os.makedirs(output_folder, exist_ok=True)

    # build a map of frame -> key
    key_map = {}
    if stego_keys:
        for idx, key in zip(frame_indices, stego_keys):
            key_map[idx] = key

    # Case 1: Extract from image sequence
    if os.path.isdir(input_path):
        files = sorted([f for f in os.listdir(input_path) if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
        if not files:
            print(f"[ERROR] No images found in directory: {input_path}")
            return
        saved = 0
        for i, fname in enumerate(files):
            frame = cv2.imread(os.path.join(input_path, fname), cv2.IMREAD_UNCHANGED)
            if frame is None:
                continue
            if extract_all or i in set(frame_indices):
                reconstructed = extract_hidden_image_from_frame(frame)
                if reconstructed is None:
                    continue
                if i in key_map:  # reverse shuffle if key is available
                    reconstructed = recover_with_key(reconstructed, key_map[i])
                    print(f"[+] Extracted (used key) from {fname}")
                else:
                    print(f"[+] Extracted (no key) from {fname}")

                out_name = f"extracted_image_{i:06d}.png"
                Image.fromarray(reconstructed).save(os.path.join(output_folder, out_name), format="PNG")
                saved += 1
        print(f"[INFO] Done. Extracted {saved} images from directory.")
        return

    # Case 2: Extract from video
    if not os.path.exists(input_path):
        print(f"[ERROR] Input video not found: {input_path}")
        return

    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        print(f"[ERROR] Unable to open video: {input_path}")
        return

    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"[INFO] Opened video: {input_path} — fps={fps}, size={width}x{height}, frames={total_frames}")

    frame_set = set(frame_indices)
    frame_num, saved = 0, 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if extract_all or frame_num in frame_set:
            reconstructed = extract_hidden_image_from_frame(frame)
            if reconstructed is None:
                frame_num += 1
                continue

            if frame_num in key_map:
                reconstructed = recover_with_key(reconstructed, key_map[frame_num])
                print(f"[+] Extracted frame {frame_num} (used key)")
            else:
                print(f"[+] Extracted frame {frame_num} (no key)")

            out_name = f"extracted_frame_{frame_num:06d}.png"
            Image.fromarray(reconstructed).save(os.path.join(output_folder, out_name), format="PNG")
            saved += 1

        frame_num += 1

    cap.release()
    print(f"[INFO] Done. Extracted {saved} images.")



input_video = "video/Output/output_lossless.avi"
frame_indices = [10, 25, 50, 75, 100]
output_folder = "video/Extracted"
stego_keys = ["key1", "key2", "key3", "key4", "key5"]

os.makedirs(output_folder, exist_ok=True)
extract_images_from_video(input_video, frame_indices, output_folder, extract_all=False, stego_keys=stego_keys)