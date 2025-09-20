# save as debug_compare.py and run: python3 debug_compare.py
import cv2, numpy as np
from PIL import Image
# paths (adjust if needed)
secret = "video/Images/image1.png"         # the secret image you embedded
encoded = "video/Output/debug_encoded_frame.png"  # the PNG you saved inside encoder
# load secret with PIL (RGB)
s = Image.open(secret).convert("RGB")
s_px = s.load()
# load encoded with OpenCV (BGR or BGRA)
e = cv2.imread(encoded, cv2.IMREAD_UNCHANGED)
print("encoded shape:", e.shape)
# convert to RGB explicitly
if e.shape[2] == 4:
    e_rgb = cv2.cvtColor(e, cv2.COLOR_BGRA2RGB)
else:
    e_rgb = cv2.cvtColor(e, cv2.COLOR_BGR2RGB)
# run extractor math locally for a test coord
x,y = 0,0
sr,sg,sb = s_px[x,y]
# what encoder wrote into base LSBs:
written_r = (sr >> 4) & 0x0F
encoded_pixel = e_rgb[y, x]  # numpy uses [row=y,col=x]
print("SECRET (R,G,B):", (sr,sg,sb))
print("SECRET top-nibble (0..15):", (sr>>4, sg>>4, sb>>4))
print("ENCODED pixel (R,G,B):", tuple(int(v) for v in encoded_pixel))
# extracted from encoded (inverse op)
ex_r = (int(encoded_pixel[0]) & 0x0F) << 4
ex_g = (int(encoded_pixel[1]) & 0x0F) << 4
ex_b = (int(encoded_pixel[2]) & 0x0F) << 4
print("EXTRACTED (R,G,B) from LSBs:", (ex_r, ex_g, ex_b))

# Lossy issues i think