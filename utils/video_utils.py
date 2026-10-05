import os
import sys
import math
from pathlib import Path

# Setup PATH for ffmpeg dependencies on Windows if needed
def setup_windows_paths():
    if os.name != 'nt':
        return
    paths_to_add = []
    
    sys_prefix = sys.prefix
    env_lib_bin = os.path.join(sys_prefix, "Library", "bin")
    if os.path.isdir(env_lib_bin):
        paths_to_add.append(env_lib_bin)
        
    env_scripts = os.path.join(sys_prefix, "Scripts")
    if os.path.isdir(env_scripts):
        paths_to_add.append(env_scripts)
        
    try:
        import imageio_ffmpeg
        imgio_bin = os.path.dirname(imageio_ffmpeg.get_ffmpeg_exe())
        if os.path.isdir(imgio_bin):
            paths_to_add.append(imgio_bin)
    except Exception:
        pass

    octave_bin = r"C:\Program Files\GNU Octave\Octave-10.2.0\mingw64\bin"
    if os.path.isdir(octave_bin):
        paths_to_add.append(octave_bin)

    current_path = os.environ.get("PATH", "")
    for p in paths_to_add:
        if p not in current_path:
            current_path = p + os.pathsep + current_path
    os.environ["PATH"] = current_path

setup_windows_paths()


def check_ffmpeg_available():
    """Verify if ffmpeg and ffprobe executables are accessible."""
    import shutil
    ffmpeg_exe = shutil.which("ffmpeg")
    ffprobe_exe = shutil.which("ffprobe")
    
    if not ffmpeg_exe or not ffprobe_exe:
        # Check imageio_ffmpeg fallback
        try:
            import imageio_ffmpeg
            ffmpeg_exe = imageio_ffmpeg.get_ffmpeg_exe()
        except Exception:
            ffmpeg_exe = None
            
    if not ffmpeg_exe:
        return False, "Error: FFmpeg/FFprobe executable is required but was not found in system PATH. Please install FFmpeg and ensure it is available in your PATH."
    return True, "OK"


def convert_to_float(frac_str):
    try:
        return float(frac_str)
    except ValueError:
        try:
            num, denom = frac_str.split('/')
            return float(num) / float(denom)
        except Exception:
            return None


def validate_video_file(video_path, max_duration=150.0):
    """
    Validates an input video file:
    1. Checks if path exists and is a file.
    2. Checks if file is non-empty.
    3. Verifies FFmpeg/FFprobe accessibility.
    4. Probes file format and video stream readability.
    5. Checks if duration exceeds max_duration (default: 150s).

    Returns:
        (is_valid: bool, error_message_or_status: str, info_dict: dict)
    """
    if not video_path:
        return False, "Error: No video file path was provided.", {}

    video_path = os.path.abspath(video_path)
    filename = os.path.basename(video_path)

    if not os.path.exists(video_path):
        return False, f"Error: Video file not found at: {video_path}", {}

    if not os.path.isfile(video_path):
        return False, f"Error: The path '{video_path}' is not a valid file.", {}

    if os.path.getsize(video_path) == 0:
        return False, f"Error: The video file '{filename}' is empty (0 bytes).", {}

    ffmpeg_ok, ffmpeg_msg = check_ffmpeg_available()
    if not ffmpeg_ok:
        return False, ffmpeg_msg, {}

    try:
        import ffmpeg
        probe = ffmpeg.probe(video_path)
    except Exception as e:
        return False, f"Error: The supplied file '{filename}' is not a valid or readable video file. (ffprobe error: {e})", {}

    if not probe or 'streams' not in probe:
        return False, f"Error: Failed to probe streams from video file '{filename}'.", {}

    video_stream = next((stream for stream in probe['streams'] if stream.get('codec_type') == 'video'), None)
    if not video_stream:
        return False, f"Error: No valid video stream found in '{filename}'.", {}

    # Extract dimensions
    try:
        width = int(video_stream.get('width', 0))
        height = int(video_stream.get('height', 0))
    except (ValueError, TypeError):
        width, height = 0, 0

    # Extract duration
    duration = None
    if 'duration' in video_stream:
        try:
            duration = float(video_stream['duration'])
        except (ValueError, TypeError):
            pass
    if duration is None and 'format' in probe and 'duration' in probe['format']:
        try:
            duration = float(probe['format']['duration'])
        except (ValueError, TypeError):
            pass

    # Fallback duration estimation from nb_frames and avg_frame_rate
    if duration is None or duration <= 0:
        try:
            nb_frames = float(video_stream.get('nb_frames', 0))
            fps_str = video_stream.get('avg_frame_rate', '25/1')
            fps_val = convert_to_float(fps_str)
            if nb_frames > 0 and fps_val and fps_val > 0:
                duration = nb_frames / fps_val
        except Exception:
            pass

    if duration is not None and max_duration is not None and max_duration > 0:
        if duration > max_duration + 0.5:  # Allow 0.5s margin for rounding
            return False, (
                f"Error: Video duration ({duration:.1f}s) exceeds the maximum supported limit of "
                f"{max_duration:.0f} seconds (75 two-second clips). Please provide a shorter video."
            ), {"duration": duration, "width": width, "height": height}

    info = {
        "duration": duration if duration is not None else -1,
        "width": width,
        "height": height
    }
    return True, "OK", info
