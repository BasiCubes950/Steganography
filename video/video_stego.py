import cv2
import numpy as np
from PIL import Image
import random
import os

def encode_image_in_frame(frame, secret_img_path, stego_key):
    # Convert frame to PIL Image
    base_img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)).convert("RGB")
    secret_img = Image.open(secret_img_path).convert("RGB")
    secret_img = secret_img.resize(base_img.size)
    width, height = base_img.size
    base_pixels = base_img.load()
    secret_pixels = secret_img.load()
    random.seed(stego_key)
    pixel_indices = [(x, y) for y in range(height) for x in range(width)]
    random.shuffle(pixel_indices)
    for x, y in pixel_indices:
        br, bg, bb = base_pixels[x, y]
        sr, sg, sb = secret_pixels[x, y]
        br = (br & 0b11110000) | (sr >> 4)
        bg = (bg & 0b11110000) | (sg >> 4)
        bb = (bb & 0b11110000) | (sb >> 4)
        base_pixels[x, y] = (br, bg, bb)
    # Convert back to numpy array
    return cv2.cvtColor(np.array(base_img), cv2.COLOR_RGB2BGR)

def hide_images_in_video(input_video_path, image_paths, frame_indices, output_video_path, stego_keys):
    # Try to open the video file and provide diagnostics if it fails
    cap = cv2.VideoCapture(input_video_path)
    if not cap.isOpened():
        # Helpful diagnostics
        print(f"[ERROR] Unable to open video: {input_video_path}")
        print(f"[INFO] File exists: {os.path.exists(input_video_path)}")
        return
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"[INFO] Opened video: {input_video_path} — fps={fps}, size={width}x{height}, frames={total_frames}")
    # Ensure output directory exists
    os.makedirs(os.path.dirname(output_video_path), exist_ok=True)
    out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))
    frame_map = {idx: (img, key) for idx, img, key in zip(frame_indices, image_paths, stego_keys)}
    frame_num = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if frame_num in frame_map:
            img_path, key = frame_map[frame_num]
            frame = encode_image_in_frame(frame, img_path, key)
            print(f"[+] Hid image {img_path} in frame {frame_num} with key '{key}'")
        out.write(frame)
        frame_num += 1
    cap.release()
    out.release()
    print(f"[+] Output video saved to {output_video_path}")

def main():
    # Default to the video folder where input.mp4 is stored in this repo
    input_video = "video/input.mp4"
    image_paths = [f"video/Images/image{i+1}.png" for i in range(5)]
    frame_indices = [10, 50, 100, 150, 200]  # User can change these
    output_video = "video/Output/output.mp4"
    stego_keys = [f"key{i+1}" for i in range(5)]  # User can change these
    hide_images_in_video(input_video, image_paths, frame_indices, output_video, stego_keys)

main()