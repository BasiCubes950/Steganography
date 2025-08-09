from PIL import Image
import os

# put message in base image
def encode(base_path, secret_path, output_path):
    # set to RGB format
    base_img = Image.open(base_path).convert("RGB")
    secret_img = Image.open(secret_path).convert("RGB")
    
    # get the pixel info
    base_pixels = base_img.load()
    secret_pixels = secret_img.load()
    
    # reformat to match base image
    width, height = base_img.size
    secret_img = secret_img.resize((width, height))  # Ensure same size
    
    # check every pixel
    for y in range(height):
        for x in range(width):
            # get RGB values for base and secret image
            r_base, g_base, b_base = base_pixels[x, y]
            r_secret, g_secret, b_secret = secret_pixels[x, y]
            
            """
            Use 4 bits of secret data per color channel

            1. save the upper 4 bits of the base color
            2. get the upper 4 bits of secret color
            3. combine the two
            4. repeat for a 4 values

            *** May be a problem if 4 bits, thats a lot of info
            and lose so much data (half) ***
            """
            r_new = (r_base & 0xF0) | (r_secret >> 4)
            g_new = (g_base & 0xF0) | (g_secret >> 4)
            b_new = (b_base & 0xF0) | (b_secret >> 4)
            
            # update!
            base_pixels[x, y] = (r_new, g_new, b_new)
    
    base_img.save(output_path)
    print(f"[+] Encoded image saved to {output_path}")

# give my image back!
def decode(encoded_path, output_folder):
    # set up
    encoded_img = Image.open(encoded_path).convert("RGB")
    width, height = encoded_img.size
    decoded_img = Image.new("RGB", (width, height)) # the cool looking map
    
    # get access to pixels
    encoded_pixels = encoded_img.load()
    decoded_pixels = decoded_img.load()
    
    # go through each pixel (once again)
    for y in range(height):
        for x in range(width):
            r, g, b = encoded_pixels[x, y]
            
            # get the hidden 4 values back for RGB
            r_hidden = (r & 0x0F) << 4
            g_hidden = (g & 0x0F) << 4
            b_hidden = (b & 0x0F) << 4
            
            # update
            decoded_pixels[x, y] = (r_hidden, g_hidden, b_hidden)
    
    output_path = os.path.join(output_folder, "extracted_secret.png")
    decoded_img.save(output_path)
    print(f"[+] Secret image extracted and saved to {output_path}")

# cool looking greyscale map of the changes
def generate_change_map(original_path, encoded_path, output_path):
    # get the base and encoded
    original = Image.open(original_path).convert("RGB")
    encoded = Image.open(encoded_path).convert("RGB")

    # load the pixels
    change_map = Image.new("L", original.size)
    orig_pixels = original.load()
    enc_pixels = encoded.load()
    change_pixels = change_map.load()

    # check every pixel (maybe we can make shortcut for this somehow)
    for x in range(original.width):
        for y in range(original.height):
            r1, g1, b1 = orig_pixels[x, y]
            r2, g2, b2 = enc_pixels[x, y]

            # Check for changes in lower 4 bits in any channel
            changes = int((r1 & 0x0F) != (r2 & 0x0F)) + \
                      int((g1 & 0x0F) != (g2 & 0x0F)) + \
                      int((b1 & 0x0F) != (b2 & 0x0F))

            # Scale to 0, 85, 170, 255 for visualization
            change_pixels[x, y] = changes * 85

    change_map.save(output_path)
    print(f"[+] Change map saved to {output_path}")


if __name__ == "__main__":
    base_path = 'Image Based/base.png'
    secret_path = 'Image Based/message.png'
    output_path = 'Results/output_image2.png'
    output_folder = 'Results/'

    print("Encoding...")
    encode(base_path, secret_path, output_path)

    print("Generating change map...")
    generate_change_map(base_path, output_path, 'Results/change_map.png')
    
    print("Decoding...")
    decode(output_path, output_folder)
