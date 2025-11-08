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
    if original is None or recompressed is None:
        print("[ERROR] Cannot analyze quality: one or both frames are None")
        return 0.0, 0.0  # return dummy values when analysis impossible
    
    try:
        psnr_val = psnr(original, recompressed)
        ssim_val = ssim(cv2.cvtColor(original, cv2.COLOR_BGR2GRAY),
                        cv2.cvtColor(recompressed, cv2.COLOR_BGR2GRAY))
        return psnr_val, ssim_val
    except Exception as e:
        print(f"[ERROR] Quality analysis failed: {str(e)}")
        return 0.0, 0.0

def main(input_folder):
    if not os.path.exists(input_folder):
        print(f"[ERROR] Input folder not found: {input_folder}")
        return

    methods = {
        "Spatial_LSB": (spatial_lsb.embed_spatial_lsb, spatial_lsb.extract_spatial_lsb),
        "Temporal": (temporal_embed.embed_temporal, temporal_embed.extract_temporal),
        "MotionVector": (motion_vector_embed.embed_motion_vector, motion_vector_embed.extract_motion_vector)
    }

    results = []
    data_bits = np.random.randint(0, 2, 8000).tolist()  # fixed payload

    video_files = [f for f in os.listdir(input_folder) if f.endswith((".mp4", ".avi"))]
    if not video_files:
        print(f"[ERROR] No video files found in {input_folder}")
        return

    for file in video_files:
        path = os.path.join(input_folder, file)
        print(f"\nProcessing video: {path}")
        
        # Read original frame
        original = cv2.VideoCapture(path)
        ret, orig_frame = original.read()
        original.release()
        
        if not ret or orig_frame is None:
            print(f"[ERROR] Could not read original video: {path}")
            continue

        for name, (embed_func, extract_func) in methods.items():
            print(f"Applying method: {name}")
            results_dir = os.path.join("results", "Videos")
            os.makedirs(results_dir, exist_ok=True)
            embedded = os.path.join(results_dir, f"{os.path.splitext(file)[0]}_{name}_embedded.mp4")
            recompressed = os.path.join(results_dir, f"{os.path.splitext(file)[0]}_{name}_recompressed.mp4")

            try:
                embed_func(path, data_bits, embedded)
            except Exception as e:
                print(f"[ERROR] Embedding failed for {name}: {str(e)}")
                continue

            simulate_recompression(embedded, recompressed)
            
            try:
                recovered_bits = extract_func(recompressed, len(data_bits))
            except Exception as e:
                print(f"[ERROR] Extraction failed for {name}: {str(e)}")
                continue

            # Read recompressed frame
            cap = cv2.VideoCapture(recompressed)
            ret, rec_frame = cap.read()
            cap.release()
            
            if not ret or rec_frame is None:
                print(f"[ERROR] Could not read recompressed video: {recompressed}")
                continue

            # Compute metrics
            ber = bit_error_rate(data_bits, recovered_bits)
            p, s = analyze_quality(orig_frame, rec_frame)
            results.append((file, name, ber, p, s))
            print(f"Results - BER: {ber:.4f}, PSNR: {p:.2f}, SSIM: {s:.4f}")

    # Statistical analysis
    if not results:
        print("\n[ERROR] No results collected - all video operations failed")
        return
        
    print("\n=== Statistical Analysis ===")
    import pandas as pd
    df = pd.DataFrame(results, columns=["Video", "Method", "BER", "PSNR", "SSIM"])
    print("\nMean values across methods:")
    print(df.groupby("Method")[["BER", "PSNR", "SSIM"]].mean())
    print("\nStandard deviation across methods:")
    print(df.groupby("Method")[["BER", "PSNR", "SSIM"]].std())
    print("\nPairwise correlation:")
    print(df.corr(numeric_only=True))

main("data")  # folder of video files
