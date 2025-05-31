from PIL import Image
import os

def hide_text(image_path, secret_text, output_path, change_map_path='Results/change_map.png'):
    image = Image.open(image_path).convert("RGB")
    original_image = image.copy()  # Store original for comparison later

    # add marker
    secret_text += "|||END|||"

    #THIS IS THE CONVERTING PART
    binary_secret_text = ''.join(format(ord(char), '08b') for char in secret_text)

    # Check if info is too big
    image_capacity = image.width * image.height * 3 # 3 bits per pixel
    if len(binary_secret_text) > image_capacity:
        raise ValueError("Image does not have sufficient capacity to hide the secret text.")

    # Hide the secret text in the image
    pixels = image.load()
    index = 0
    for i in range(image.width):
        for j in range(image.height):
            r, g, b = pixels[i, j]

            if index < len(binary_secret_text):
                r = (r & 0xFE) | int(binary_secret_text[index])
                index += 1
            if index < len(binary_secret_text):
                g = (g & 0xFE) | int(binary_secret_text[index])
                index += 1
            if index < len(binary_secret_text):
                b = (b & 0xFE) | int(binary_secret_text[index])
                index += 1

            pixels[i, j] = (r, g, b)

    # Save as PNG (lossless to avoid corruption of data)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    image.save(output_path, format='PNG')

    # Generate pixel change map
    generate_change_map(original_image, image, change_map_path)

def generate_change_map(original, modified, output_path):
    """Creates a grayscale image showing how many RGB channels changed per pixel (0–3)."""
    change_map = Image.new("L", original.size) # Set up the map
    original_pixels = original.load() # RGB values or orginal image
    modified_pixels = modified.load() # RGB values of modified image
    change_pixels = change_map.load() # load the new map (black and white)

    # go through each point and check if differences in RGB exist
    for i in range(original.width):
        for j in range(original.height):
            # "Unpack" the RGB values
            r1, g1, b1 = original_pixels[i, j] # RGB tuple from the original image at (i, j)
            r2, g2, b2 = modified_pixels[i, j] # RGB tuple from the modified image

            # Check LSB
            changes = int((r1 & 1) != (r2 & 1)) + \
                      int((g1 & 1) != (g2 & 1)) + \
                      int((b1 & 1) != (b2 & 1))

            # Scale: 0 = 0, 1 = 85, 2 = 170, 3 = 255
            change_pixels[i, j] = changes * 85 # show the greyscale image
    # save the image
    change_map.save(output_path, format='PNG')

image_path = 'Images/image.png'
output_path = 'Results/output_image.png'
secret_text = 'This is a secret message. AND I AM ADDING EXTRA WORDSAFASFASF  as230 9qd q90d 129d dq 0d9 9 d932d 29390r293121290e12edi209jd23f2fdj2 9dj230d j'

print("1. Hiding text...")
hide_text(image_path, secret_text, output_path)