import os
import cv2
import numpy as np
from skimage.metrics import peak_signal_noise_ratio as psnr, structural_similarity as ssim
from methods import spatial_lsb
from methods import temporal_embed
from methods import motion_vector_embed

def simulate_recompression(input_path, output_path):
    # simulate social-media compression (re-encode at low bitrate)
    os.system(f"ffmpeg -y -i {input_path} -b:v 500k -vf scale=640:360 -c:a copy {output_path} -loglevel quiet")

def bit_error_rate(original_bits, recovered_bits):
    errors = sum(o != r for o, r in zip(original_bits, recovered_bits))
    return errors / len(original_bits)

def analyze_quality(original, recompressed):
    psnr_val = psnr(original, recompressed)
    ssim_val = ssim(cv2.cvtColor(original, cv2.COLOR_BGR2GRAY),
                    cv2.cvtColor(recompressed, cv2.COLOR_BGR2GRAY))
    return psnr_val, ssim_val

def main(input_folder):
    methods = {
        "Spatial_LSB": (spatial_lsb.embed_spatial_lsb, spatial_lsb.extract_spatial_lsb),
        "Temporal": (temporal_embed.embed_temporal, temporal_embed.extract_temporal),
        "MotionVector": (motion_vector_embed.embed_motion_vector, motion_vector_embed.extract_motion_vector)
    }

    results = []
    data_bits = np.random.randint(0, 2, 8000).tolist()  # fixed payload

    for file in os.listdir(input_folder):
        if not file.endswith((".mp4", ".avi")):
            continue
        path = os.path.join(input_folder, file)
        original = cv2.VideoCapture(path)
        ret, orig_frame = original.read()
        original.release()

        for name, (embed_func, extract_func) in methods.items():
            embedded = f"{input_folder}/{name}_embedded.mp4"
            recompressed = f"{input_folder}/{name}_recompressed.mp4"

            embed_func(path, data_bits, embedded)
            simulate_recompression(embedded, recompressed)
            recovered_bits = extract_func(recompressed, len(data_bits))

            ber = bit_error_rate(data_bits, recovered_bits)
            cap = cv2.VideoCapture(recompressed)
            ret, rec_frame = cap.read()
            cap.release()

            p, s = analyze_quality(orig_frame, rec_frame)
            results.append((file, name, ber, p, s))

    # Statistical analysis
    import pandas as pd
    df = pd.DataFrame(results, columns=["Video", "Method", "BER", "PSNR", "SSIM"])
    print("\n=== Statistical Analysis ===")
    print(df.groupby("Method")[["BER", "PSNR", "SSIM"]].mean())
    print("\nPairwise correlation:\n", df.corr(numeric_only=True))
    print("\nStandard deviation across methods:\n", df.groupby("Method")[["BER", "PSNR", "SSIM"]].std())

main("data")  # folder of video files
