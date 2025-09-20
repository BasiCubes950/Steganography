import cv2
import numpy as np
import os
from PIL import Image
import argparse


def extract_hidden_image_from_frame(frame):
    # frame is BGR (OpenCV). Convert to RGB for consistent channel order.
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Get lower 4 bits and shift left to reconstruct the hidden (upper 4 bits) image
    r_hidden = (rgb[:, :, 0] & 0x0F) << 4
    g_hidden = (rgb[:, :, 1] & 0x0F) << 4
    b_hidden = (rgb[:, :, 2] & 0x0F) << 4

    reconstructed = np.stack([r_hidden, g_hidden, b_hidden], axis=2).astype(np.uint8)
    return reconstructed


def extract_images_from_video(input_video_path, frame_indices, output_folder, extract_all=False):
    if not os.path.exists(input_video_path):
        # try some likely alternate locations
        alt = os.path.join(os.path.dirname(__file__), input_video_path)
        if os.path.exists(alt):
            input_video_path = alt
        else:
            print(f"[ERROR] input video not found: {input_video_path}")
            print(f"[INFO] tried alternate: {alt}")
            return

    cap = cv2.VideoCapture(input_video_path)
    if not cap.isOpened():
        print(f"[ERROR] Unable to open video: {input_video_path}")
        print(f"[INFO] File exists: {os.path.exists(input_video_path)}")
        return

    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"[INFO] Opened video: {input_video_path} — fps={fps}, size={width}x{height}, frames={total_frames}")

    os.makedirs(output_folder, exist_ok=True)

    frame_set = set(frame_indices)
    frame_num = 0
    saved = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if extract_all or frame_num in frame_set:
            reconstructed = extract_hidden_image_from_frame(frame)
            img = Image.fromarray(reconstructed)
            out_name = f"extracted_frame_{frame_num:06d}.png"
            out_path = os.path.join(output_folder, out_name)
            img.save(out_path, format="PNG")
            print(f"[+] Saved extracted image for frame {frame_num} -> {out_path}")
            saved += 1

        frame_num += 1

    cap.release()
    print(f"[INFO] Done. Extracted {saved} images.")


def main():
    parser = argparse.ArgumentParser(description="Extract hidden images embedded in video frames (lower 4 bits per channel).")
    parser.add_argument("--input", help="Path to input video", default="video/Output/output.mp4")
    parser.add_argument("--frames", help="Comma-separated frame indices to extract (e.g. 10,50,100)", default="10,50,100,150,200")
    parser.add_argument("--output", help="Output folder", default=os.path.join(os.path.dirname(__file__), "Output"))
    parser.add_argument("--all", action="store_true", help="Extract from all frames")

    print("arguments in...")

    args = parser.parse_args()
    if args.all:
        print("--all specified, extracting from all frames")
        frame_indices = []
    else:
        print("extracting from specified frames only")
        frame_indices = []
        if args.frames.strip():
            print("parsing frame indices...")
            for part in args.frames.split(','):
                part = part.strip()
                if not part:
                    continue
                try:
                    frame_indices.append(int(part))
                except ValueError:
                    print(f"[WARNING] ignoring invalid frame index: {part}")

    print(f"[INFO] Extracting from frames: {frame_indices if frame_indices else 'ALL'}")
    extract_images_from_video(args.input, frame_indices, args.output, extract_all=args.all)

main()