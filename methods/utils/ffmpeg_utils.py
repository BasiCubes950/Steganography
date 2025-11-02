import subprocess
from pathlib import Path

def recompress_video(input_path, output_path, bitrate=None, resolution=None, codec="libx264"):
    """
    Recompress a video using ffmpeg.
    - bitrate: e.g., '800k', '1500k', '2000k'
    - resolution: tuple (width, height), e.g., (1280, 720)
    - codec: 'libx264' (H.264) or 'libx265' (H.265)
    """
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    cmd = ["ffmpeg", "-y", "-i", str(input_path), "-c:v", codec]

    if bitrate:
        cmd += ["-b:v", bitrate]
    if resolution:
        w, h = resolution
        cmd += ["-vf", f"scale={w}:{h}"]

    cmd += [str(output_path)]

    subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

def batch_recompress(input_folder, output_folder, bitrates, resolutions, codecs):
    """
    Recompress all videos in a folder under multiple settings.
    """
    input_folder = Path(input_folder)
    output_folder = Path(output_folder)
    output_folder.mkdir(parents=True, exist_ok=True)

    for video in input_folder.glob("*.*"):
        for bitrate in bitrates:
            for resolution in resolutions:
                for codec in codecs:
                    suffix = f"_{bitrate}_{resolution[1]}p_{codec}.mp4"
                    out_path = output_folder / f"{video.stem}{suffix}"
                    recompress_video(video, out_path, bitrate, resolution, codec)