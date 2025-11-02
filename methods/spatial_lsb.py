import cv2
import numpy as np
import os

def embed_message_in_frame(frame, message_bits, bit_index=0):
    """Embed binary message bits into the frame’s LSBs (per pixel)."""
    h, w, c = frame.shape
    total_pixels = h * w * c
    frame_flat = frame.flatten().astype(np.uint8)

    for i in range(len(message_bits)):
        if bit_index + i >= total_pixels:
            break
        pixel_val = frame_flat[bit_index + i]
        # Mask out LSB and insert bit (ensure uint8 type)
        frame_flat[bit_index + i] = np.uint8((int(pixel_val) & 0xFE) | int(message_bits[i]))

    frame_stego = frame_flat.reshape((h, w, c))
    return frame_stego, bit_index + len(message_bits)

def extract_message_from_frame(frame, message_length, bit_index=0):
    """Extract hidden message bits from the frame’s LSBs."""
    h, w, c = frame.shape
    frame_flat = frame.flatten()

    bits = []
    for i in range(message_length):
        if bit_index + i >= len(frame_flat):
            break
        bits.append(frame_flat[bit_index + i] & 1)
    return bits, bit_index + message_length

def embed_message_in_video(input_video, output_video, message):
    """Hide a binary string message in all frames of a video."""
    cap = cv2.VideoCapture(input_video)
    if not cap.isOpened():
        raise ValueError(f"Cannot open video file {input_video}")

    # Convert message to binary
    message_bytes = message.encode('utf-8')
    message_bits = np.unpackbits(np.frombuffer(message_bytes, dtype=np.uint8))

    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_video, fourcc, fps, (width, height))

    bit_index = 0
    total_bits = len(message_bits)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if bit_index < total_bits:
            frame, bit_index = embed_message_in_frame(frame, message_bits, bit_index)
        out.write(frame)

    cap.release()
    out.release()

def extract_message_from_video(stego_video, message_length_bytes):
    """Extract a hidden message from a stego video."""
    cap = cv2.VideoCapture(stego_video)
    if not cap.isOpened():
        raise ValueError(f"Cannot open video file {stego_video}")

    total_bits = message_length_bytes * 8
    extracted_bits = []
    bit_index = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if bit_index < total_bits:
            bits, bit_index = extract_message_from_frame(frame, total_bits - bit_index)
            extracted_bits.extend(bits)
        else:
            break

    cap.release()

    # Convert bits back to bytes
    message_bytes = np.packbits(np.array(extracted_bits[:total_bits], dtype=np.uint8))
    return message_bytes.tobytes().decode('utf-8', errors='ignore')

def main():
    # Example run
    input_path = "data/data1.mp4"
    output_path = "results/lsb_output.mp4"
    os.makedirs("output_videos", exist_ok=True)

    secret_message = "Hidden data using Spatial LSB steganography!"
    print("Embedding message...")
    embed_message_in_video(input_path, output_path, secret_message)

    print("Extracting message...")
    recovered = extract_message_from_video(output_path, len(secret_message))
    print("Recovered:", recovered)

main()