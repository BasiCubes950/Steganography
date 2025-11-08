import os
from pathlib import Path
# CORRECTED IMPORT: Import the function directly, not the module object
from extract.spatial_lsb_extract import extract_spatial_lsb


def run_single_extraction_test():
    """
    A standalone script to test a single extraction method and save the result.
    """
    # --- Configuration ---
    # ⚠️ 1. DEFINE YOUR INPUT FILE PATH HERE 
    INPUT_VIDEO_PATH = "output/stego/spatial_lsb_stego.mp4" 
    
    # ⚠️ 2. DEFINE THE OUTPUT FOLDER AND FILENAME
    OUTPUT_FOLDER = Path("output/extracted")
    OUTPUT_IMAGE_NAME = "extracted_test_image_lsb.png"
    
    # --- Setup ---
    print("Setting up output directory...")
    OUTPUT_FOLDER.mkdir(exist_ok=True)
    
    output_image_path = OUTPUT_FOLDER / OUTPUT_IMAGE_NAME
    
    if not Path(INPUT_VIDEO_PATH).exists():
        print(f"ERROR: Input video not found at '{INPUT_VIDEO_PATH}'.")
        print("Please update INPUT_VIDEO_PATH in this script to a valid file.")
        return

    print(f"Starting extraction from: {INPUT_VIDEO_PATH}")
    print(f"Saving extracted image to: {output_image_path}")

    # --- Run Extraction ---
    try:
        # CORRECTED CALL: Call the function directly
        extract_spatial_lsb(str(INPUT_VIDEO_PATH), str(output_image_path))
        
        print("\n✅ Extraction Test COMPLETE.")
        print(f"Image saved successfully to: {output_image_path}")

    except Exception as e:
        print(f"\n❌ Extraction Test FAILED. Error: {e}")
        print("Check if your extraction logic (in spatial_lsb_extract.py) is correct.")

if __name__ == "__main__":
    run_single_extraction_test()