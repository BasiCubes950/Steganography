import cv2
import numpy as np
import os
from PIL import Image
import argparse
import random

# extracts single hidden image from a video frame
def extract_hidden_image_from_frame(frame):
    # frame is BGR (OpenCV). Convert to RGB for consistent channel order.
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # get hidden data
    """
    BUG might be here
    """
    """
    r_hidden = ((rgb[:, :, 0] % 16) * 16)
    g_hidden = ((rgb[:, :, 1] % 16) * 16)
    b_hidden = ((rgb[:, :, 2] % 16) * 16)
    """
    r_hidden = (rgb[:, :, 0] & 0x0F) << 4
    g_hidden = (rgb[:, :, 1] & 0x0F) << 4
    b_hidden = (rgb[:, :, 2] & 0x0F) << 4

    reconstructed = np.stack([r_hidden, g_hidden, b_hidden], axis=2).astype(np.uint8) # combine the 3 channels into image array
    return reconstructed

# gets all secret data
def extract_images_from_video(input_video_path, frame_indices, output_folder, extract_all=False, stego_keys=None):
    # verify input video path
    if not os.path.exists(input_video_path):
        # try some likely alternate locations
        alt = os.path.join(os.path.dirname(__file__), input_video_path)
        if os.path.exists(alt):
            input_video_path = alt
        else:
            print(f"[ERROR] input video not found: {input_video_path}")
            print(f"[INFO] tried alternate: {alt}")
            return

    # open video file
    cap = cv2.VideoCapture(input_video_path)
    if not cap.isOpened():
        print(f"[ERROR] Unable to open video: {input_video_path}")
        print(f"[INFO] File exists: {os.path.exists(input_video_path)}")
        return

    # get video properties
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"[INFO] Opened video: {input_video_path} — fps={fps}, size={width}x{height}, frames={total_frames}")

    # Ensure output directory exists
    os.makedirs(output_folder, exist_ok=True)

    # build a map of frame -> key if provided (keys list should align with frame_indices)
    key_map = {}
    if stego_keys:
        for idx, key in zip(frame_indices, stego_keys):
            key_map[idx] = key

    """
    ------------------------------------
    Extract images
    """
    frame_set = set(frame_indices)
    frame_num = 0
    saved = 0

    # read through video frames
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # check if we should extract from this frame
        if extract_all or frame_num in frame_set:
            reconstructed = extract_hidden_image_from_frame(frame)

            # If a key exists for this frame, reverse the shuffle mapping to recover the original secret image
            if frame_num in key_map:
                key = key_map[frame_num]
                # coords are (x,y) pairs
                coords = [(x, y) for y in range(height) for x in range(width)]
                shuffled = coords.copy()
                random.seed(key)
                random.shuffle(shuffled)

                # reconstructed currently has secret pixels at destination positions (dst).
                # We need to move them back to their source positions (src) used during encoding.
                recovered = np.zeros_like(reconstructed)
                for dst, src in zip(coords, shuffled):
                    dx, dy = dst
                    sx, sy = src
                    # numpy arrays use [row=y, col=x]
                    recovered[sy, sx, :] = reconstructed[dy, dx, :]

                img = Image.fromarray(recovered)
                out_name = f"extracted_frame_{frame_num:06d}.png"
                out_path = os.path.join(output_folder, out_name)
                img.save(out_path, format="PNG")
                print(f"[+] Saved extracted image for frame {frame_num} -> {out_path} (used key)")
                saved += 1
            else:
                # No key available: save the straightforward reconstruction (may be scrambled)
                img = Image.fromarray(reconstructed)
                out_name = f"extracted_frame_{frame_num:06d}.png"
                out_path = os.path.join(output_folder, out_name)
                img.save(out_path, format="PNG")
                print(f"[+] Saved extracted image for frame {frame_num} -> {out_path} (no key)")
                saved += 1

        frame_num += 1

    # cleanup
    cap.release()
    print(f"[INFO] Done. Extracted {saved} images.")


input = "video/Output/output.mp4"
frame_indices = [10, 25, 50, 75, 100]
output = os.path.join(os.path.dirname(__file__), "Output")

extract_images_from_video(input, frame_indices, output, False, stego_keys=["key1", "key2", "key3", "key4", "key5"])