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
        br = (br & 0b11110000) | (sr >> 4)
        bg = (bg & 0b11110000) | (sg >> 4)
        bb = (bb & 0b11110000) | (sb >> 4)
        base_pixels[dx, dy] = (br, bg, bb)
    
    # Convert back to numpy array
    # BUG MAY BE HERE
    # debug only: save single encoded frame losslessly
    base_img.save("video/Output/debug_encoded_frame.png", format="PNG")
    return cv2.cvtColor(np.array(base_img), cv2.COLOR_RGB2BGR)

def assemble_pngs_to_video(png_dir, output_video_path, fps):
    # Collect all PNGs in order
    png_files = sorted([f for f in os.listdir(png_dir) if f.endswith(".png")])
    if not png_files:
        print("[ERROR] No PNGs found to assemble!")
        return
    
    # Get frame size from first image
    first_frame = cv2.imread(os.path.join(png_dir, png_files[0]))
    height, width, _ = first_frame.shape
    
    # Use FFV1 codec for truly lossless video
    fourcc = cv2.VideoWriter_fourcc(*'FFV1')  
    out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))
    
    for filename in png_files:
        frame = cv2.imread(os.path.join(png_dir, filename))
        out.write(frame)
    
    out.release()
    print(f"[+] Assembled {len(png_files)} PNGs into lossless video: {output_video_path}")

def count_bit_planes(frame):
    """Return number of non-empty bit planes (0-7) present across RGB channels in the frame.

    A bit plane is considered present if any pixel in any channel has that bit set.
    Returns an integer in 0..8 representing how many bit planes contain any 1-bits.
    """
    if frame is None:
        return 0
    arr = frame.astype(np.uint8)
    # Convert to RGB for clarity
    rgb = cv2.cvtColor(arr, cv2.COLOR_BGR2RGB)
    present_planes = 0
    for bit in range(8):
        mask = 1 << bit
        if np.any((rgb & mask) != 0):
            present_planes += 1
    return present_planes

# embed multiple images into set of frames in a video
def hide_images_in_video(input_video_path, image_paths, frame_indices, output_video_path, stego_keys, save_png_sequence=False):
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
    
    # create video writer (only if not saving PNG sequence)
    out = None
    if not save_png_sequence:
        out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))
    frame_map = {idx: (img, key) for idx, img, key in zip(frame_indices, image_paths, stego_keys)}
    frame_num = 0
    bit_plane_counts = []

    # process each frame
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # count bit planes on input frame (before embedding)
        #planes = count_bit_planes(frame)
        #bit_plane_counts.append(planes)
        #print(f"[INFO] Frame {frame_num}: {planes} non-empty bit planes (0-7)")

        if frame_num in frame_map:
            img_path, key = frame_map[frame_num]
            frame = encode_image_in_frame(frame, img_path, key)
            print(f"[+] Hid image {img_path} in frame {frame_num} with key '{key}'")

        if save_png_sequence:
            # save lossless PNG per frame
            os.makedirs(os.path.dirname(output_video_path), exist_ok=True)
            png_path = os.path.join(os.path.dirname(output_video_path), f"frame_{frame_num:06d}.png")
            Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)).save(png_path, format="PNG")
        else:
            out.write(frame)
        frame_num += 1
    
    # cleanup
    cap.release()
    if out is not None:
        out.release()
    elif save_png_sequence:
        # auto assemble lossless video from PNGs
        assemble_pngs_to_video(os.path.dirname(output_video_path), output_video_path.replace(".mp4", "_lossless.avi"), fps)

    print(f"[+] Output video saved to {output_video_path}")
    return bit_plane_counts


input_video = "video/input.mp4"
image_paths = [f"video/Images/image{i+1}.png" for i in range(5)]
frame_indices = [10, 25, 50, 75, 100]
output_video = "video/Output/output.mp4"
stego_keys = [f"key{i+1}" for i in range(5)]  # User can change these
hide_images_in_video(input_video, image_paths, frame_indices, output_video, stego_keys, save_png_sequence=True)