import os
from typing import List

import numpy as np
import ffmpeg
from PIL import Image

OUTPUT_DIR = "tests/outputs"
if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

SD_CPP_CLI = "C:\\Users\\Willi\\Documents\\GitHub\\stable-diffusion.cpp\\build\\bin\\sd"


# ===========================================
# Video Saving
# ===========================================


def save_video_ffmpeg(frames: List[Image.Image], fps: int, out_path: str) -> None:
    if not frames:
        raise ValueError("No frames provided")

    width, height = frames[0].size

    # Concatenate frames into raw RGB bytes
    raw_bytes = b"".join(np.array(frame.convert("RGB"), dtype=np.uint8).tobytes() for frame in frames)
    (
        ffmpeg.input(
            "pipe:",
            format="rawvideo",
            pix_fmt="rgb24",
            s=f"{width}x{height}",
            r=fps,
        )
        .output(
            out_path,
            vcodec="libx264",
            pix_fmt="yuv420p",
            r=fps,
            movflags="+faststart",
        )
        .overwrite_output()
        .run(input=raw_bytes)
    )
