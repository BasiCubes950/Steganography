import os
from pathlib import Path

def load_binary_payload(file_path):
    """Load binary data from a file."""
    with open(file_path, "rb") as f:
        return list(map(int, ''.join(format(byte, '08b') for byte in f.read())))

def save_binary_payload(bits, file_path):
    """Save binary data to a file."""
    Path(file_path).parent.mkdir(parents=True, exist_ok=True)
    byte_str = ''.join(str(b) for b in bits)
    byte_data = int(byte_str, 2).to_bytes((len(byte_str) + 7) // 8, byteorder='big')
    with open(file_path, "wb") as f:
        f.write(byte_data)

def ensure_folder(path):
    """Create folder if it doesn't exist."""
    os.makedirs(path, exist_ok=True)
