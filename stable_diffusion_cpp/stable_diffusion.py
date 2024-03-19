from typing import List, Literal, Any, Optional
import os

import stable_diffusion_cpp.stable_diffusion_cpp as sd_cpp

from ._logger import set_verbose

import ctypes
from PIL import Image


# TODO add enum logic for cpp enums
# from enum import Enum

# class Color(Enum):
#     RED = 1
#     GREEN = 2
#     BLUE = 3

# print(Color.RED)          # Output: Color.RED
# print(Color.RED.value)    # Output: 1


class StableDiffusion:
    """High-level Python wrapper for a stable-diffusion.cpp model."""

    def __init__(
        self,
        model_path: str,
        vae_path: str = "",
        taesd_path: str = "",
        control_net_path: str = "",
        lora_model_dir: str = "",
        embed_dir: str = "",
        stacked_id_embed_dir: str = "",
        vae_decode_only: bool = False,
        vae_tiling: bool = False,
        free_params_immediately: bool = False,
        n_threads: int = -1,
        wtype: int = 8,
        rng_type: int = 0,
        schedule: int = 0,
        keep_clip_on_cpu: bool = False,
        keep_control_net_cpu: bool = False,
        keep_vae_on_cpu: bool = False,
        # Misc
        verbose: bool = True,
    ):
        # Logging
        self.verbose = verbose
        set_verbose(verbose)

        # Params
        self.model_path = model_path
        self.vae_path = vae_path
        self.taesd_path = taesd_path
        self.control_net_path = control_net_path
        self.lora_model_dir = lora_model_dir
        self.embed_dir = embed_dir
        self.stacked_id_embed_dir = stacked_id_embed_dir
        self.vae_decode_only = vae_decode_only
        self.vae_tiling = vae_tiling
        self.free_params_immediately = free_params_immediately
        self.n_threads = n_threads
        self.wtype = wtype
        self.rng_type = rng_type
        self.schedule = schedule
        self.keep_clip_on_cpu = keep_clip_on_cpu
        self.keep_control_net_cpu = keep_control_net_cpu
        self.keep_vae_on_cpu = keep_vae_on_cpu

        # Model loading
        self.model = None

        if not os.path.exists(model_path):
            raise ValueError(f"Model path does not exist: {model_path}")

        # Initialize the model
        self.model = sd_cpp.new_sd_ctx(
            self.model_path.encode("utf-8"),
            self.vae_path.encode("utf-8"),
            self.taesd_path.encode("utf-8"),
            self.control_net_path.encode("utf-8"),
            self.lora_model_dir.encode("utf-8"),
            self.embed_dir.encode("utf-8"),
            self.stacked_id_embed_dir.encode("utf-8"),
            self.vae_decode_only,
            self.vae_tiling,
            self.free_params_immediately,
            self.n_threads,
            self.wtype,
            self.rng_type,
            self.schedule,
            self.keep_clip_on_cpu,
            self.keep_control_net_cpu,
            self.keep_vae_on_cpu,
        )

    def txt_to_img(
        self,
        prompt: str,
        negative_prompt: str = "",
        clip_skip: int = -1,
        cfg_scale: float = 7.0,
        width: int = 512,
        height: int = 512,
        sample_method: int = 0,
        sample_steps: int = 1,
        seed: int = 42,
        batch_count: int = 1,
        control_cond=None,
        control_strength: float = 0.9,
        style_strength: float = 20.0,
        normalize_input: bool = False,
        input_id_images_path: str = "",
        batch_size: int = 1,
    ) -> List[Image.Image]:
        c_images = sd_cpp.txt2img(
            self.model,
            prompt.encode("utf-8"),
            negative_prompt.encode("utf-8"),
            clip_skip,
            cfg_scale,
            width,
            height,
            sample_method,
            sample_steps,
            seed,
            batch_count,
            control_cond,
            control_strength,
            style_strength,
            normalize_input,
            input_id_images_path.encode("utf-8"),
        )

        images = image_slice(c_images, 1)

        # Convert images to Image
        for img in images:
            output_image = bytes_to_image(img["data"], img["width"], img["height"])

    def img_to_img(
        self,
        image: Image.Image,
        prompt: bytes,
        negative_prompt: bytes,
        clip_skip: int,
        cfg_scale: float,
        width: int,
        height: int,
        sample_method: int,
        sample_steps: int,
        strength: float,
        seed: int,
        batch_count: int,
    ) -> List[Image.Image]:
        pass

    def img_to_vid(
        self,
        image: Image.Image,
        width: int,
        height: int,
        video_frames: int,
        motion_bucket_id: int,
        fps: int,
        augmentation_level: float,
        min_cfg: float,
        cfg_scale: float,
        sample_method: int,
        sample_steps: int,
        strength: float,
        seed: int,
    ) -> List[Image.Image]:
        pass

    def upscale(
        self,
        image: Image.Image,
        upscale_factor: int = 4,
    ) -> List[Image.Image]:
        pass


# Convert C array to Python list of bytes
def c_array_to_bytes(c_array, buffer_size: int):
    return bytearray(
        ctypes.cast(c_array, ctypes.POINTER(ctypes.c_byte * buffer_size)).contents
    )


# Define a function to retrieve images from C array
def image_slice(c_images: sd_cpp.sd_image_t_p, count: int):
    # Convert the C array to a Python list
    img_array = ctypes.cast(
        c_images, ctypes.POINTER(sd_cpp.sd_image_t * count)
    ).contents

    images = []
    for i in range(count):
        c_img = img_array[i]

        # Calculate the size of the data buffer
        buffer_size = c_img.channel * c_img.width * c_img.height

        img = {
            "width": c_img.width,
            "height": c_img.height,
            "channel": c_img.channel,
            "data": c_array_to_bytes(c_img.data, buffer_size),
        }
        images.append(img)
    return images


def bytes_to_image(byte_data: bytes, width: int, height: int):
    img = Image.new("RGBA", (width, height))

    for y in range(height):
        for x in range(width):
            idx = (y * width + x) * 3
            img.putpixel(
                (x, y), (byte_data[idx], byte_data[idx + 1], byte_data[idx + 2], 255)
            )

    return img
