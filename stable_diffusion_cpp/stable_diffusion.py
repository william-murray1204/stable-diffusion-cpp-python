from typing import List, Literal, Any, Optional
import os

import stable_diffusion_cpp.stable_diffusion_cpp as sd_cpp

from ._logger import set_verbose

import ctypes
from PIL import Image


class StableDiffusion:
    """High-level Python wrapper for a stable-diffusion.cpp model."""

    def __init__(
        self,
        model_path: str,
        vae_path: str = "",
        taesd_path: str = "",
        control_net_path: str = "",
        upscaler_path: str = "",
        lora_model_dir: str = "",
        embed_dir: str = "",
        stacked_id_embed_dir: str = "",
        vae_decode_only: bool = False,
        vae_tiling: bool = False,
        free_params_immediately: bool = False,
        n_threads: int = -1,
        ggml_type: int = sd_cpp.GGMLType.SD_TYPE_COUNT,
        rng_type: int = sd_cpp.RNGType.STD_DEFAULT_RNG,
        schedule: int = sd_cpp.Schedule.DISCRETE,
        keep_clip_on_cpu: bool = False,
        keep_control_net_cpu: bool = False,
        keep_vae_on_cpu: bool = False,
        # Misc
        verbose: bool = True,
        # Extra Params
        **kwargs,  # type: ignore
    ):
        """Load a stable-diffusion.cpp model from `model_path`.

        Examples:
            Basic usage

            >>> import stable_diffusion_cpp
            >>> model = stable_diffusion_cpp.StableDiffusion(
            ...     model_path="path/to/model",
            ... )
            >>> images = stable_diffusion.txt_to_img(prompt="a lovely cat")

        Args:
            model_path: Path to the model.
            vae_path: Path to the vae.
            taesd_path: Path to the taesd.
            control_net_path: Path to the control net.
            upscaler_path: Path to esrgan model (Upscale images after generation).
            lora_model_dir: Lora model directory.
            embed_dir: Path to embeddings.
            stacked_id_embed_dir: Path to PHOTOMAKER stacked id embeddings.
            vae_decode_only: Process vae in decode only mode.
            vae_tiling: Process vae in tiles to reduce memory usage.
            free_params_immediately: Free parameters immediately after use.
            n_threads: Number of threads to use for generation.
            ggml_type: GGML type (default: sd_type_count).
            rng_type: RNG (default: cuda).
            schedule: Denoiser sigma schedule (default: discrete).
            keep_clip_on_cpu: Keep clip in CPU (for low vram).
            keep_control_net_cpu: Keep controlnet in CPU (for low vram).
            keep_vae_on_cpu: Keep vae in CPU (for low vram).
            verbose: Print verbose output to stderr.

        Raises:
            ValueError: If a model path does not exist.

        Returns:
            A Stable Diffusion instance.
        """
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
        self.ggml_type = ggml_type
        self.rng_type = rng_type
        self.schedule = schedule
        self.keep_clip_on_cpu = keep_clip_on_cpu
        self.keep_control_net_cpu = keep_control_net_cpu
        self.keep_vae_on_cpu = keep_vae_on_cpu

        # =========== Logging ===========

        self.verbose = verbose
        set_verbose(verbose)

        # =========== Model loading ===========
        self.model = None

        if not os.path.exists(model_path):
            raise ValueError(f"Model path does not exist: {model_path}")

        # Initialize the Stable Diffusion model
        self._model = sd_cpp.new_sd_ctx(
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
            self.ggml_type,
            self.rng_type,
            self.schedule,
            self.keep_clip_on_cpu,
            self.keep_control_net_cpu,
            self.keep_vae_on_cpu,
        )

        if self._model is None:
            raise ValueError("Failed to initialize Stable Diffusion model")

        # =========== Model loading ===========

        self._upscaler = None

        # Initialize the upscaler model
        if upscaler_path:
            if not os.path.exists(upscaler_path):
                raise ValueError(f"Upscaler path does not exist: {upscaler_path}")

            self._upscaler = sd_cpp.new_upscaler_ctx(
                upscaler_path.encode("utf-8"), self.n_threads, self.ggml_type
            )

            if self._upscaler is None:
                raise ValueError("Failed to initialize upscaler model")

    # ============================================
    # Text to Image
    # ============================================

    def txt_to_img(
        self,
        prompt: str,
        negative_prompt: str = "",
        clip_skip: int = -1,
        cfg_scale: float = 7.0,
        width: int = 512,
        height: int = 512,
        sample_method: int = 0,
        sample_steps: int = 20,
        seed: int = 42,
        batch_count: int = 1,
        control_cond=None,
        control_strength: float = 0.9,
        style_strength: float = 20.0,
        normalize_input: bool = False,
        input_id_images_path: str = "",
    ) -> List[Image.Image]:

        # Run the txt2img to generate images
        c_images = sd_cpp.txt2img(
            self._model,
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

        # Convert the C array of images to a Python list of images
        return self._convert_to_images(c_images)

    # ============================================
    # Image to Image
    # ============================================

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
        """Generate images from an input image."""

        # Convert the image to a byte array
        image_bytes = self._image_to_bytes(image)

        c_images = sd_cpp.img2img(
            self._model,
            image_bytes,
            prompt,
            negative_prompt,
            clip_skip,
            cfg_scale,
            width,
            height,
            sample_method,
            sample_steps,
            strength,
            seed,
            batch_count,
        )
        return self._convert_to_images(c_images)

    # ============================================
    # Image to Video
    # ============================================

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
        """Generate a video from an input image."""
        image_bytes = self._image_to_bytes(image)

        c_video = sd_cpp.img2vid(
            self._model,
            image_bytes,
            width,
            height,
            video_frames,
            motion_bucket_id,
            fps,
            augmentation_level,
            min_cfg,
            cfg_scale,
            sample_method,
            sample_steps,
            strength,
            seed,
        )
        # TODO - Convert to video
        return

    # ============================================
    # Image Upscaling
    # ============================================

    def upscale(
        self,
        images: List[Image.Image],
        upscale_factor: int = 4,
    ) -> List[Image.Image]:
        """Upscale a list of images using the upscaler model."""

        c_images = []

        for image in images:
            # Convert the image to a byte array
            image_bytes = self._image_to_bytes(image)

            # Upscale the image
            c_images.append(
                sd_cpp.upscale(
                    self._upscaler,
                    image_bytes,
                    upscale_factor,
                )
            )

        return self._convert_to_images(c_images)

    # ============================================
    # Utility functions
    # ============================================

    def _image_to_bytes(self, img: Image.Image):
        """Convert a PIL Image to a byte array."""
        return img.tobytes()

    def _convert_to_images(self, c_images):
        # Convert C array to Python list of images
        images = self._image_slice(c_images, 1)

        # Convert each image to PIL Image
        for i in range(len(images)):
            img = images[i]
            images[i] = self._bytes_to_image(img["data"], img["width"], img["height"])

        return images

    def _image_slice(self, c_images: sd_cpp.sd_image_t_p, count: int):
        """Convert a C array of images to a Python list of images."""

        def _c_array_to_bytes(self, c_array, buffer_size: int):
            return bytearray(
                ctypes.cast(
                    c_array, ctypes.POINTER(ctypes.c_byte * buffer_size)
                ).contents
            )

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
                "data": _c_array_to_bytes(c_img.data, buffer_size),
            }
            images.append(img)

        # Return the list of images
        return images

    def _bytes_to_image(self, byte_data: bytes, width: int, height: int):
        """Convert a byte array to a PIL Image."""
        img = Image.new("RGBA", (width, height))

        for y in range(height):
            for x in range(width):
                idx = (y * width + x) * 3
                img.putpixel(
                    (x, y),
                    (byte_data[idx], byte_data[idx + 1], byte_data[idx + 2], 255),
                )

        return img
