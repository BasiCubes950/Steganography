import cv2
import numpy as np
from PIL import Image
import random
import os

# embeds a secret image into a single video frame using a key to shuffle pixel order
def encode_image_in_frame(frame, secret_img_path, stego_key):
    # Setting + cleaning up images
    base_img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)).convert("RGB")
    secret_img = Image.open(secret_img_path).convert("RGB")
    secret_img = secret_img.resize(base_img.size)
    width, height = base_img.size

    # get the pixel data
    base_pixels = base_img.load()
    secret_pixels = secret_img.load()

    random.seed(stego_key) # Seed RNG for reproducibility


    # embed secret into base image
    """
    BUG MIGHT BE HERE
    """
    coords = [(x, y) for y in range(height) for x in range(width)]
    shuffled = coords.copy()
    random.shuffle(shuffled)
    # Map secret source coords -> destination coords using shuffled order so the key matters
    for dst, src in zip(coords, shuffled):
        dx, dy = dst
        sx, sy = src
        br, bg, bb = base_pixels[dx, dy]
        sr, sg, sb = secret_pixels[sx, sy]
        """
        br = ((br // 16)*16) + (sr >> 4)
        bg = ((bg // 16)*16) + (sg >> 4)
        bb = ((bb // 16)*16) + (sb >> 4)
        """

        br = (br & 0b11110000) | (sr >> 4)
        bg = (bg & 0b11110000) | (sg >> 4)
        bb = (bb & 0b11110000) | (sb >> 4)
        base_pixels[dx, dy] = (br, bg, bb)
    
    # Convert back to numpy array
    # BUG MAY BE HERE
    return cv2.cvtColor(np.array(base_img), cv2.COLOR_RGB2BGR)

# embed multiple images into set of frames in a video
def hide_images_in_video(input_video_path, image_paths, frame_indices, output_video_path, stego_keys):
    # Try to open the video file
    cap = cv2.VideoCapture(input_video_path)
    if not cap.isOpened():
        # Helpful diagnostics
        print(f"[ERROR] Unable to open video: {input_video_path}")
        print(f"[INFO] File exists: {os.path.exists(input_video_path)}")
        return
    
    # set up video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"[INFO] Opened video: {input_video_path} — fps={fps}, size={width}x{height}, frames={total_frames}")

    # Ensure output directory exists
    os.makedirs(os.path.dirname(output_video_path), exist_ok=True)
    
    # create video writer
    out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))
    frame_map = {idx: (img, key) for idx, img, key in zip(frame_indices, image_paths, stego_keys)}
    frame_num = 0

    # process each frame
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
    
    # cleanup
    cap.release()
    out.release()
    print(f"[+] Output video saved to {output_video_path}")

input_video = "video/input.mp4"
image_paths = [f"video/Images/image{i+1}.png" for i in range(5)]
frame_indices = [10, 25, 50, 75, 100]
output_video = "video/Output/output.mp4"
stego_keys = [f"key{i+1}" for i in range(5)]  # User can change these
hide_images_in_video(input_video, image_paths, frame_indices, output_video, stego_keys)