from PIL import Image
import os

def extract_text(image_path):
    image = Image.open(image_path)
    pixels = image.load()
    binary_secret_text = ""

    # get last bit of each color channel
    for i in range(image.width):
        for j in range(image.height):
            r, g, b = pixels[i, j]
            binary_secret_text += str(r & 1)
            binary_secret_text += str(g & 1)
            binary_secret_text += str(b & 1)

    # Converter
    secret_text = ""
    """
    every 8 bits becomes an
    American Standard Code for Information Interchange
    (ASCII) character
    """
    for i in range(0, len(binary_secret_text), 8):
        byte = binary_secret_text[i:i+8]
        if len(byte) < 8:
            break
        char = chr(int(byte, 2))
        secret_text += char

        # When the end marker is found stop
        if "|||END|||" in secret_text:
            secret_text = secret_text.split("|||END|||")[0]
            break

    return secret_text

output_path = 'Results/output_image.png'

print("2. Extracting text...")
extracted_text = extract_text(output_path)
print("Extracted text:", extracted_text)