import cv2
import numpy as np
import os
from tqdm import tqdm


def image_to_bits(image_path):
    """Convert an image to a bit array."""
    from .utils.convert import image_to_bits as convert_image_to_bits
    return convert_image_to_bits(image_path)


def bits_to_image(bits, shape, output_path=None):
    """Convert bit array back to an image and save."""
    from .utils.convert import bits_to_image as convert_bits_to_image
    return convert_bits_to_image(bits, shape, output_path)


def embed_image_in_video(video_path, secret_image_path, output_path):
    """Embed a secret image bitstream into a video using LSB."""
    # Load secret image as bitstream
    message_bits, img_shape = image_to_bits(secret_image_path)
    total_bits = len(message_bits)
    bit_index = 0

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise FileNotFoundError(f"Cannot open video: {video_path}")

    fps = int(cap.get(cv2.CAP_PROP_FPS))
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # Setup output video
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter(output_path, fourcc, fps, (w, h))

    total_capacity = frame_count * w * h * 3
    if total_bits > total_capacity:
        raise ValueError(f"Image too large for this video. Capacity: {total_capacity} bits, Required: {total_bits}")

    print(f"Embedding {total_bits} bits into {frame_count} frames...")
    progress = tqdm(total=frame_count)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        flat = frame.flatten()
        for i in range(len(flat)):
            if bit_index >= total_bits:
                break
            flat[i] = (flat[i] & ~1) | message_bits[bit_index]
            bit_index += 1

        frame = flat.reshape(frame.shape)
        out.write(frame)
        progress.update(1)

        if bit_index >= total_bits:
            break

    cap.release()
    out.release()
    progress.close()

    print(f"✅ Embedded {bit_index}/{total_bits} bits into {os.path.basename(output_path)}")
    return img_shape, total_bits


def extract_image_from_video(stego_video_path, output_path, bit_count, shape):
    """Extract embedded bits from stego video and reconstruct image."""
    cap = cv2.VideoCapture(stego_video_path)
    if not cap.isOpened():
        raise FileNotFoundError(f"Cannot open stego video: {stego_video_path}")

    bits = []
    print("Extracting bits from video...")
    progress = tqdm(total=int(cap.get(cv2.CAP_PROP_FRAME_COUNT)))

    while len(bits) < bit_count:
        ret, frame = cap.read()
        if not ret:
            break

        flat = frame.flatten()
        remaining = bit_count - len(bits)
        extracted = flat[:remaining] & 1
        bits.extend(extracted)
        progress.update(1)

    cap.release()
    progress.close()

    bits = np.array(bits[:bit_count], dtype=np.uint8)
    bits_to_image(bits, shape, output_path)


def main():
    video_path = "data/data1.mp4"  # Input video
    secret_image_path = "data/secret.png"  # Your secret image
    
    # Ensure output directories exist
    os.makedirs("results/Videos", exist_ok=True)
    os.makedirs("results/images", exist_ok=True)
    
    output_video_path = "results/Videos/lsb_stego_output.avi"
    recovered_image_path = "results/images/lsb_extracted_secret.png"

    try:
        # Step 1: Embed image into video
        print(f"\nEmbedding image {secret_image_path} into video...")
        img_shape, bit_count = embed_image_in_video(video_path, secret_image_path, output_video_path)

        # Step 2: Extract the image back
        print(f"\nExtracting hidden image from {output_video_path}...")
        extract_image_from_video(output_video_path, recovered_image_path, bit_count, img_shape)
        
        print(f"\n✅ Process complete!")
        print(f"- Stego video saved to: {output_video_path}")
        print(f"- Extracted image saved to: {recovered_image_path}")
    except FileNotFoundError as e:
        print(f"\n❌ Error: {str(e)}")
        print("Please ensure input files exist in the correct locations:")


if __name__ == "__main__":
    main()
