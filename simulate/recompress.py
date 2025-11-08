import ffmpeg
import logging

def recompress_video(input_path: str, output_path: str, bitrate: str, resolution: str = None, codec: str = 'libx264'):
    """
    Recompresses a video using ffmpeg-python.
    
    Args:
        input_path: Path to the source video.
        output_path: Path to save the recompressed video.
        bitrate: Target video bitrate (e.g., '500k', '1M').
        resolution: Target resolution (e.g., '640x480'). If None, keeps original.
        codec: Video codec to use (e.g., 'libx264', 'libvpx-vp9', 'hevc_videotoolbox').
    """
    try:
        stream = ffmpeg.input(input_path)
        
        # Prepare arguments for ffmpeg.output()
        output_args = {
            'vcodec': codec,
            'video_bitrate': bitrate,
            'an': None  # Disable audio
        }
        
        if resolution:
            output_args['vf'] = f'scale={resolution}'
            
        stream = ffmpeg.output(stream, output_path, **output_args)
        
        # Run ffmpeg, overwriting output and suppressing stdout/stderr
        ffmpeg.run(stream, overwrite_output=True, quiet=True)
        
    except ffmpeg.Error as e:
        logging.error(f"ffmpeg error on {input_path}: {e.stderr.decode()}")
        raise e
    except Exception as e:
        logging.error(f"Failed to recompress video {input_path}: {e}")
        raise e