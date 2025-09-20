from PIL import Image
import random

def encode(base_img_path, secret_img_path, output_path, stego_key):
    # Load images
    base_img = Image.open(base_img_path).convert("RGB")
    secret_img = Image.open(secret_img_path).convert("RGB")

    # Resize secret to match base
    secret_img = secret_img.resize(base_img.size)
    width, height = base_img.size

    base_pixels = base_img.load()
    secret_pixels = secret_img.load()

    # Keep a copy of the original base image for change maps
    original_base_img = base_img.copy()

    # Seed random generator with stego key
    random.seed(stego_key)
    pixel_indices = [(x, y) for y in range(height) for x in range(width)]
    random.shuffle(pixel_indices)

    # Hide top 4 bits of secret in lower 4 bits of base
    for x, y in pixel_indices:
        br, bg, bb = base_pixels[x, y]
        sr, sg, sb = secret_pixels[x, y]

        br = (br & 0b11110000) | (sr >> 4)
        bg = (bg & 0b11110000) | (sg >> 4)
        bb = (bb & 0b11110000) | (sb >> 4)

        base_pixels[x, y] = (br, bg, bb)

    base_img.save(output_path, "PNG")
    print(f"[+] Encoded into {output_path} using key '{stego_key}'")

    # Create 8 change maps, one for each bit level
    create_bitwise_change_maps(original_base_img, base_img, "Changes/change_map")


def create_bitwise_change_maps(original_img, encoded_img, output_prefix):
    """Create 8 change maps, one for each bit level (0=LSB, 7=MSB), showing which pixels changed at each bit."""
    width, height = original_img.size
    orig_pixels = original_img.load()
    enc_pixels = encoded_img.load()
    for bit in range(8):
        change_map = Image.new("RGB", (width, height), (0, 0, 0))
        change_pixels = change_map.load()
        for x in range(width):
            for y in range(height):
                # Only check the red channel (index 0)
                orig_bit = (orig_pixels[x, y][0] >> bit) & 1
                enc_bit = (enc_pixels[x, y][0] >> bit) & 1
                if orig_bit != enc_bit:
                    change_pixels[x, y] = (255, 255, 255)
                else:
                    change_pixels[x, y] = (0, 0, 0)
        out_path = f"{output_prefix}_bit{bit}.png"
        change_map.save(out_path)
        print(f"[+] Bit {bit} change map saved to {out_path}")


def decode(stego_img_path, output_path, stego_key, secret_size):
    # Load stego image
    stego_img = Image.open(stego_img_path).convert("RGB")
    width, height = stego_img.size
    stego_pixels = stego_img.load()

    secret_img = Image.new("RGB", (width, height))
    secret_pixels = secret_img.load()

    # Seed random generator with same key
    random.seed(stego_key)
    pixel_indices = [(x, y) for y in range(height) for x in range(width)]
    random.shuffle(pixel_indices)

    # Extract secret from lower 4 bits
    for x, y in pixel_indices:
        br, bg, bb = stego_pixels[x, y]

        sr = (br & 0b00001111) << 4
        sg = (bg & 0b00001111) << 4
        sb = (bb & 0b00001111) << 4

        secret_pixels[x, y] = (sr, sg, sb)

    # Resize back to original secret size
    secret_img = secret_img.resize(secret_size)
    secret_img.save(output_path, "PNG")
    print(f"[+] Decoded to {output_path} using key '{stego_key}'")


if __name__ == "__main__":
    base_img_path = "Image Based/base.png"
    secret_img_path = "Image Based/message.png"
    encoded_img_path = "Results/encoded.png"
    decoded_img_path = "Results/decoded.png"
    stego_key = "supersecret123"

    # Encode
    encode(base_img_path, secret_img_path, encoded_img_path, stego_key)

    # Decode
    secret_size = Image.open(secret_img_path).size
    decode(encoded_img_path, decoded_img_path, stego_key, secret_size)
