import ffmpeg
import logging
import os

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
        # First verify input file exists and is readable
        if not os.path.exists(input_path):
            raise FileNotFoundError(f"Input video file not found: {input_path}")
            
        probe = None
        try:
            probe = ffmpeg.probe(input_path)
            logging.info(f"Input video probe successful: {probe['streams'][0]['codec_name']}")
        except ffmpeg.Error as e:
            logging.warning(f"Could not probe input file {input_path}: {e.stderr.decode()}")
            
        # Build the ffmpeg command with more robust settings
        stream = ffmpeg.input(input_path)
        
        # Prepare arguments for ffmpeg.output()
        output_args = {
            'vcodec': codec,
            'video_bitrate': bitrate,
            'an': None,  # Disable audio
            'movflags': '+faststart',  # Optimize for web playback
            'strict': 'experimental'  # Allow experimental codecs if needed
        }
        
        # Add pixel format for x264
        if codec == 'libx264':
            output_args['pix_fmt'] = 'yuv420p'
            
        if resolution:
            output_args['vf'] = f'scale={resolution}'
            
        stream = ffmpeg.output(stream, output_path, **output_args)
        
        # Log the ffmpeg command for debugging
        cmd = ffmpeg.get_args(stream)
        logging.info(f"FFmpeg command: ffmpeg {' '.join(cmd)}")
        
        # Run ffmpeg with more verbose output
        try:
            out, err = ffmpeg.run(stream, capture_stdout=True, capture_stderr=True, overwrite_output=True)
            if err:
                logging.info(f"FFmpeg stderr: {err.decode()}")
        except ffmpeg.Error as e:
            print(f"stdout: {e.stdout.decode()}")
            print(f"stderr: {e.stderr.decode()}")
            raise
        
    except ffmpeg.Error as e:
        logging.error(f"ffmpeg error on {input_path}: {e.stderr.decode()}")
        raise e
    except Exception as e:
        logging.error(f"Failed to recompress video {input_path}: {e}")
        raise e