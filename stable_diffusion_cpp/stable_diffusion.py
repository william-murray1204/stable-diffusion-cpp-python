import os
import ctypes
import random
import contextlib
import multiprocessing
from typing import Dict, List, Union, Callable, Optional

from PIL import Image

import stable_diffusion_cpp as sd_cpp
from ._utils import suppress_stdout_stderr
from ._logger import log_event, set_verbose
from ._internals import _UpscalerModel, _StableDiffusionModel
from stable_diffusion_cpp import RNGType, GGMLType, Scheduler, SampleMethod


class StableDiffusion:
    """High-level Python wrapper for a stable-diffusion.cpp model."""

    def __init__(
        self,
        model_path: str = "",
        clip_l_path: str = "",
        clip_g_path: str = "",
        clip_vision_path: str = "",
        t5xxl_path: str = "",
        diffusion_model_path: str = "",
        high_noise_diffusion_model_path: str = "",
        vae_path: str = "",
        taesd_path: str = "",
        control_net_path: str = "",
        upscaler_path: str = "",
        lora_model_dir: str = "",
        embedding_dir: str = "",
        stacked_id_embed_dir: str = "",
        vae_decode_only: bool = False,
        vae_tiling: bool = False,
        n_threads: int = -1,
        wtype: Union[str, GGMLType, int, float] = "default",
        rng_type: Union[str, RNGType, int, float] = "cuda",
        offload_params_to_cpu: bool = False,
        keep_clip_on_cpu: bool = False,
        keep_control_net_on_cpu: bool = False,
        keep_vae_on_cpu: bool = False,
        diffusion_flash_attn: bool = False,
        diffusion_conv_direct: bool = False,
        vae_conv_direct: bool = False,
        chroma_use_dit_mask: bool = True,
        chroma_use_t5_mask: bool = False,
        chroma_t5_mask_pad: int = 1,
        flow_shift: float = float("inf"),
        verbose: bool = True,
    ):
        """Load a stable-diffusion.cpp model from `model_path`.

        Examples:
            Basic usage

            >>> import stable_diffusion_cpp
            >>> model = stable_diffusion_cpp.StableDiffusion(
            ...     model_path="path/to/model",
            ... )
            >>> images = stable_diffusion.generate_image(prompt="a lovely cat")
            >>> images[0].save("output.png")

        Args:
            model_path: Path to the full model.
            clip_l_path: Path to the clip-l text encoder.
            clip_g_path: Path to the clip-g text encoder.
            clip_vision_path: Path to the clip-vision encoder.
            t5xxl_path: Path to the t5xxl text encoder.
            diffusion_model_path: Path to the standalone diffusion model.
            high_noise_diffusion_model_path: Path to the standalone high noise diffusion model.
            vae_path: Path to the vae.
            taesd_path: Path to the taesd. Using Tiny AutoEncoder for fast decoding (low quality).
            control_net_path: Path to the Control Net model.
            upscaler_path: Path to ESRGAN model (Upscale images after generation).
            lora_model_dir: Lora model directory.
            embedding_dir: Path to embeddings.
            stacked_id_embed_dir: Path to PHOTOMAKER stacked id embeddings.
            vae_decode_only: Process vae in decode only mode.
            vae_tiling: Process vae in tiles to reduce memory usage.
            n_threads: Number of threads to use for generation (default: half the number of CPUs).
            wtype: The weight type (default: automatically determines the weight type of the model file).
            rng_type: Random number generator.
            offload_params_to_cpu: Place the weights in RAM to save VRAM, and automatically load them into VRAM when needed.
            keep_clip_on_cpu: Keep clip in CPU (for low vram).
            keep_control_net_on_cpu: Keep Control Net in CPU (for low vram).
            keep_vae_on_cpu: Keep vae in CPU (for low vram).
            diffusion_flash_attn: Use flash attention in diffusion model (can reduce memory usage significantly). May lower quality or crash if backend not supported.
            diffusion_conv_direct: Use Conv2d direct in the diffusion model. May crash if backend not supported.
            vae_conv_direct: Use Conv2d direct in the vae model (should improve performance). May crash if backend not supported.
            chroma_use_dit_mask: Use DiT mask for Chroma.
            chroma_use_t5_mask: Use T5 mask for Chroma.
            chroma_t5_mask_pad: T5 mask padding size of Chroma.
            flow_shift: Shift value for Flow models like SD3.x or WAN (default: auto).
            verbose: Print verbose output.

        Raises:
            ValueError: If arguments are invalid or mutually incompatible.
            RuntimeError: If the model is not loaded when required.
            NotImplementedError: If a feature is not implemented.

        Returns:
            A Stable Diffusion instance.
        """
        # Params
        self.model_path = self._clean_path(model_path)
        self.clip_l_path = self._clean_path(clip_l_path)
        self.clip_g_path = self._clean_path(clip_g_path)
        self.clip_vision_path = self._clean_path(clip_vision_path)
        self.t5xxl_path = self._clean_path(t5xxl_path)
        self.diffusion_model_path = self._clean_path(diffusion_model_path)
        self.high_noise_diffusion_model_path = self._clean_path(high_noise_diffusion_model_path)
        self.vae_path = self._clean_path(vae_path)
        self.taesd_path = self._clean_path(taesd_path)
        self.control_net_path = self._clean_path(control_net_path)
        self.upscaler_path = self._clean_path(upscaler_path)
        self.lora_model_dir = self._clean_path(lora_model_dir)
        self.embedding_dir = self._clean_path(embedding_dir)
        self.stacked_id_embed_dir = self._clean_path(stacked_id_embed_dir)
        self.vae_decode_only = vae_decode_only
        self.vae_tiling = vae_tiling
        self.n_threads = n_threads
        self.wtype = wtype
        self.rng_type = rng_type
        self.offload_params_to_cpu = offload_params_to_cpu
        self.keep_clip_on_cpu = keep_clip_on_cpu
        self.keep_control_net_on_cpu = keep_control_net_on_cpu
        self.keep_vae_on_cpu = keep_vae_on_cpu
        self.diffusion_flash_attn = diffusion_flash_attn
        self.diffusion_conv_direct = diffusion_conv_direct
        self.vae_conv_direct = vae_conv_direct
        self.chroma_use_dit_mask = chroma_use_dit_mask
        self.chroma_use_t5_mask = chroma_use_t5_mask
        self.chroma_t5_mask_pad = chroma_t5_mask_pad
        self.flow_shift = flow_shift
        self._stack = contextlib.ExitStack()

        # Default to half the number of CPUs
        if n_threads <= 0:
            self.n_threads = max(multiprocessing.cpu_count() // 2, 1)

        # -------------------------------------------
        # Logging
        # -------------------------------------------

        self.verbose = verbose
        set_verbose(verbose)

        # -------------------------------------------
        # Validate Inputs
        # -------------------------------------------

        self.wtype = self._validate_and_set_input(self.wtype, GGML_TYPE_MAP, "wtype")
        self.rng_type = self._validate_and_set_input(self.rng_type, RNG_TYPE_MAP, "rng_type")

        # -------------------------------------------
        # SD Model Loading
        # -------------------------------------------

        self._model = self._stack.enter_context(
            contextlib.closing(
                _StableDiffusionModel(
                    model_path=self.model_path,
                    clip_l_path=self.clip_l_path,
                    clip_g_path=self.clip_g_path,
                    clip_vision_path=self.clip_vision_path,
                    t5xxl_path=self.t5xxl_path,
                    diffusion_model_path=self.diffusion_model_path,
                    high_noise_diffusion_model_path=self.high_noise_diffusion_model_path,
                    vae_path=self.vae_path,
                    taesd_path=self.taesd_path,
                    control_net_path=self.control_net_path,
                    lora_model_dir=self.lora_model_dir,
                    embedding_dir=self.embedding_dir,
                    stacked_id_embed_dir=self.stacked_id_embed_dir,
                    vae_decode_only=self.vae_decode_only,
                    vae_tiling=self.vae_tiling,
                    n_threads=self.n_threads,
                    wtype=self.wtype,
                    rng_type=self.rng_type,
                    offload_params_to_cpu=self.offload_params_to_cpu,
                    keep_clip_on_cpu=self.keep_clip_on_cpu,
                    keep_control_net_on_cpu=self.keep_control_net_on_cpu,
                    keep_vae_on_cpu=self.keep_vae_on_cpu,
                    diffusion_flash_attn=self.diffusion_flash_attn,
                    diffusion_conv_direct=self.diffusion_conv_direct,
                    vae_conv_direct=self.vae_conv_direct,
                    chroma_use_dit_mask=self.chroma_use_dit_mask,
                    chroma_use_t5_mask=self.chroma_use_t5_mask,
                    chroma_t5_mask_pad=self.chroma_t5_mask_pad,
                    flow_shift=self.flow_shift,
                    verbose=self.verbose,
                )
            )
        )

        # -------------------------------------------
        # Upscaler Model Loading
        # -------------------------------------------

        self._upscaler = self._stack.enter_context(
            contextlib.closing(
                _UpscalerModel(
                    upscaler_path=upscaler_path,
                    offload_params_to_cpu=self.offload_params_to_cpu,
                    direct=self.diffusion_conv_direct,  # Use diffusion_conv_direct
                    n_threads=self.n_threads,
                    verbose=self.verbose,
                )
            )
        )

    @property
    def model(self) -> sd_cpp.sd_ctx_t_p:
        assert self._model.model is not None
        return self._model.model

    @property
    def upscaler(self) -> sd_cpp.upscaler_ctx_t_p:
        if self._upscaler is None or self._upscaler.upscaler is None:
            raise RuntimeError("Upscaler not initialized, did you pass `upscaler_path`")
        return self._upscaler.upscaler

    # ===========================================
    # Generate Image
    # ===========================================

    def generate_image(
        self,
        prompt: str,
        negative_prompt: str = "",
        clip_skip: int = -1,
        init_image: Optional[Union[Image.Image, str]] = None,
        ref_images: Optional[List[Union[Image.Image, str]]] = None,
        increase_ref_index: bool = False,
        mask_image: Optional[Union[Image.Image, str]] = None,
        width: int = 512,
        height: int = 512,
        # ---
        # guidance_params
        cfg_scale: float = 7.0,
        image_cfg_scale: Optional[float] = None,
        guidance: float = 3.5,
        # sample_params
        scheduler: Union[str, Scheduler, int, float] = "default",
        sample_method: Union[str, SampleMethod, int, float] = "euler_a",
        sample_steps: int = 20,
        eta: float = 0.0,
        # slg_params
        skip_layers: List[int] = [7, 8, 9],
        skip_layer_start: float = 0.01,
        skip_layer_end: float = 0.2,
        slg_scale: float = 0.0,
        # ---
        strength: float = 0.75,
        seed: int = 42,
        batch_count: int = 1,
        control_image: Optional[Union[Image.Image, str]] = None,
        control_strength: float = 0.9,
        style_strength: float = 20.0,
        normalize_input: bool = False,
        input_id_images_path: str = "",
        canny: bool = False,
        upscale_factor: int = 1,
        progress_callback: Optional[Callable] = None,
    ) -> List[Image.Image]:
        """Generate images from a text prompt and or input images.

        Args:
            prompt: The prompt to render.
            negative_prompt: The negative prompt.
            clip_skip: Ignore last layers of CLIP network (1 ignores none, 2 ignores one layer, <= 0 represents unspecified, will be 1 for SD1.x, 2 for SD2.x).
            init_image: An input image path or Pillow Image to direct the generation.
            ref_images: A list of input image paths or Pillow Images for Flux Kontext models (can be used multiple times).
            increase_ref_index: Automatically increase the indices of reference images based on the order they are listed (starting with 1).
            mask_image: The inpainting mask image path or Pillow Image.
            width: Image width, in pixel space.
            height: Image height, in pixel space.
            cfg_scale: Unconditional guidance scale.
            image_cfg_scale: Image guidance scale for inpaint or instruct-pix2pix models.
            guidance: Distilled guidance scale for models with guidance input.
            scheduler: Denoiser sigma scheduler.
            sample_method: Sampling method.
            sample_steps: Number of sample steps.
            eta: Eta in DDIM, only for DDIM and TCD.
            skip_layers: Layers to skip for SLG steps (SLG will be enabled at step int([STEPS]x[START]) and disabled at int([STEPS]x[END])).
            skip_layer_start: SLG enabling point.
            skip_layer_end: SLG disabling point.
            slg_scale: Skip layer guidance (SLG) scale, only for DiT models.
            strength: Strength for noising/unnoising.
            seed: RNG seed (uses random seed for < 0).
            batch_count: Number of images to generate.
            control_image: A control condition image path or Pillow Image (Control Net).
            control_strength: Strength to apply Control Net.
            style_strength: Strength for keeping input identity.
            normalize_input: Normalize PHOTOMAKER input id images.
            input_id_images_path: Path to PHOTOMAKER input id images dir.
            canny: Apply canny edge detection preprocessor to the control_image.
            upscale_factor: Run the ESRGAN upscaler this many times.
            progress_callback: Callback function to call on each step end.

        Returns:
            A list of Pillow Images.
        """

        if self.model is None:
            raise RuntimeError("Stable Diffusion model not loaded")

        if self.vae_decode_only == True and (init_image or ref_images):
            raise ValueError("`vae_decode_only` cannot be True when an `init_image` or `ref_images` are provided")

        # -------------------------------------------
        # Validation
        # -------------------------------------------

        width = self._validate_dimensions(width, "width")
        height = self._validate_dimensions(height, "height")

        if batch_count < 1:
            raise ValueError("`batch_count` must be at least 1")
        if upscale_factor < 1:
            raise ValueError("`upscale_factor` must at least 1")
        if sample_steps < 1:
            raise ValueError("`sample_steps` must be at least 1")
        if strength < 0.0 or strength > 1.0:
            raise ValueError("`strength` must be in the range [0.0, 1.0]")

        # -------------------------------------------
        # CFG Scale
        # -------------------------------------------

        image_cfg_scale = cfg_scale if image_cfg_scale is None else image_cfg_scale

        # -------------------------------------------
        # Set Seed
        # -------------------------------------------

        # Set a random seed if seed is negative
        if seed < 0:
            seed = random.randint(0, 10000)

        # -------------------------------------------
        # Set the Callback Function
        # -------------------------------------------

        if progress_callback is not None:

            @sd_cpp.sd_progress_callback
            def sd_progress_callback(
                step: int,
                steps: int,
                time: float,
                data: ctypes.c_void_p,
            ):
                progress_callback(step, steps, time)

            sd_cpp.sd_set_progress_callback(sd_progress_callback, ctypes.c_void_p(0))

        # -------------------------------------------
        # Reference Images
        # -------------------------------------------

        ref_images_pointer, ref_images_count = self._format_reference_images(ref_images, width, height)

        # -------------------------------------------
        # Parameters
        # -------------------------------------------

        guidance_params = sd_cpp.sd_guidance_params_t(
            txt_cfg=cfg_scale,
            img_cfg=image_cfg_scale,
            distilled_guidance=guidance,
            slg=sd_cpp.sd_slg_params_t(
                layers=(ctypes.c_int * len(skip_layers))(*skip_layers),  # Convert to ctypes array
                layer_count=len(skip_layers),
                layer_start=skip_layer_start,
                layer_end=skip_layer_end,
                scale=slg_scale,
            ),
        )

        sample_params = sd_cpp.sd_sample_params_t(
            guidance=guidance_params,
            scheduler=self._validate_and_set_input(scheduler, SCHEDULER_MAP, "scheduler"),
            sample_method=self._validate_and_set_input(sample_method, SAMPLE_METHOD_MAP, "sample_method"),
            sample_steps=sample_steps,
            eta=eta,
        )

        params = sd_cpp.sd_img_gen_params_t(
            prompt=prompt.encode("utf-8"),
            negative_prompt=negative_prompt.encode("utf-8"),
            clip_skip=clip_skip,
            init_image=self._format_init_image(init_image, width, height),
            ref_images=ref_images_pointer,
            ref_images_count=ref_images_count,
            increase_ref_index=increase_ref_index,
            mask_image=self._format_mask_image(mask_image, width, height),
            width=width,
            height=height,
            sample_params=sample_params,
            strength=strength,
            seed=seed,
            batch_count=batch_count,
            control_image=self._format_control_image(control_image, canny, width, height),
            control_strength=control_strength,
            style_strength=style_strength,
            normalize_input=normalize_input,
            input_id_images_path=input_id_images_path.encode("utf-8"),
        )

        with suppress_stdout_stderr(disable=self.verbose):
            # Generate images
            c_images = sd_cpp.generate_image(
                self.model,
                ctypes.byref(params),
            )

        # Convert C array to Python list of images
        images = self._sd_image_t_p_to_images(c_images, batch_count, upscale_factor)

        # -------------------------------------------
        # Attach Image Metadata
        # -------------------------------------------

        func_args = locals()
        gen_args = {
            k: v
            for k, v in func_args.items()
            if k
            not in {
                "self",
                "images",
                "c_images",
                "ref_images_pointer",
                "progress_callback",
                "sd_progress_callback",
                "guidance_params",
                "sample_params",
                "params",
            }
        }
        model_args = {k: v for k, v in self.__dict__.items() if not k.startswith("_")}  # Skip internals

        for i, image in enumerate(images):
            image.info.update({**model_args, **gen_args, "seed": seed + i if batch_count > 1 else seed})

        return images

    # ===========================================
    # Generate Video
    # ===========================================

    def generate_video(
        self,
        prompt: str = "",
        negative_prompt: str = "",
        clip_skip: int = -1,
        init_image: Optional[Union[Image.Image, str]] = None,
        end_image: Optional[Union[Image.Image, str]] = None,
        width: int = 512,
        height: int = 512,
        # ---
        # guidance_params
        cfg_scale: float = 7.0,
        image_cfg_scale: Optional[float] = None,
        guidance: float = 3.5,
        # sample_params
        scheduler: Union[str, Scheduler, int, float] = "default",
        sample_method: Optional[Union[str, SampleMethod, int, float]] = "euler_a",
        sample_steps: int = 20,
        eta: float = 0.0,
        # slg_params
        skip_layers: List[int] = [7, 8, 9],
        skip_layer_start: float = 0.01,
        skip_layer_end: float = 0.2,
        slg_scale: float = 0.0,
        # ---
        # high_noise_guidance_params
        high_noise_cfg_scale: float = 7.0,
        high_noise_image_cfg_scale: Optional[float] = None,
        high_noise_guidance: float = 3.5,
        # high_noise_sample_params
        high_noise_scheduler: Union[str, Scheduler, int, float] = "default",
        high_noise_sample_method: Union[str, SampleMethod, int, float] = "euler_a",
        high_noise_sample_steps: int = -1,
        high_noise_eta: float = 0.0,
        # high_noise_slg_params
        high_noise_skip_layers: List[int] = [7, 8, 9],
        high_noise_skip_layer_start: float = 0.01,
        high_noise_skip_layer_end: float = 0.2,
        high_noise_slg_scale: float = 0.0,
        # ---
        moe_boundary: float = 0.875,
        strength: float = 0.75,
        seed: int = 42,
        video_frames: int = 1,
        upscale_factor: int = 1,
        progress_callback: Optional[Callable] = None,
    ) -> List[Image.Image]:
        """Generate a video from an image input.

        Args:
            prompt: The prompt to render.
            negative_prompt: The negative prompt.
            clip_skip: Ignore last layers of CLIP network (1 ignores none, 2 ignores one layer, <= 0 represents unspecified, will be 1 for SD1.x, 2 for SD2.x).
            init_image: An input image path or Pillow Image to start the generation.
            end_image: An input image path or Pillow Image to end the generation (required by flf2v).
            width: Video width, in pixel space.
            height: Video height, in pixel space.
            cfg_scale: Unconditional guidance scale.
            image_cfg_scale: Image guidance scale for inpaint or instruct-pix2pix models (default: same as `cfg_scale`).
            guidance: Distilled guidance scale for models with guidance input.
            scheduler: Denoiser sigma scheduler.
            sample_method: Sampling method.
            sample_steps: Number of sample steps.
            eta: Eta in DDIM, only for DDIM and TCD.
            skip_layers: Layers to skip for SLG steps (SLG will be enabled at step int([STEPS]x[START]) and disabled at int([STEPS]x[END])).
            skip_layer_start: SLG enabling point.
            skip_layer_end: SLG disabling point.
            slg_scale: Skip layer guidance (SLG) scale, only for DiT models.
            high_noise_cfg_scale: High noise diffusion model equivalent of `cfg_scale`.
            high_noise_image_cfg_scale: High noise diffusion model equivalent of `image_cfg_scale`.
            high_noise_guidance: High noise diffusion model equivalent of `guidance`.
            high_noise_scheduler: High noise diffusion model equivalent of `scheduler`.
            high_noise_sample_method: High noise diffusion model equivalent of `sample_method`.
            high_noise_sample_steps: High noise diffusion model equivalent of `sample_steps` (default: -1 = auto).
            high_noise_eta: High noise diffusion model equivalent of `eta`.
            high_noise_skip_layers: High noise diffusion model equivalent of `skip_layers`.
            high_noise_skip_layer_start: High noise diffusion model equivalent of `skip_layer_start`.
            high_noise_skip_layer_end: High noise diffusion model equivalent of `skip_layer_end`.
            high_noise_slg_scale: High noise diffusion model equivalent of `slg_scale`.
            moe_boundary: Timestep boundary for Wan2.2 MoE model. Only enabled if `high_noise_sample_steps` is set to -1.
            strength: Strength for noising/unnoising.
            seed: RNG seed (uses random seed for < 0).
            video_frames: Number of video frames to generate.
            upscale_factor: Run the ESRGAN upscaler this many times.
            progress_callback: Callback function to call on each step end.

        Returns:
            A list of Pillow Images (video frames)."""

        if self.model is None:
            raise RuntimeError("Stable Diffusion model not loaded")

        if self.vae_decode_only == True:
            raise ValueError("`vae_decode_only` cannot be True when generating videos")

        # -------------------------------------------
        # Validation
        # -------------------------------------------

        width = self._validate_dimensions(width, "width")
        height = self._validate_dimensions(height, "height")

        if upscale_factor < 1:
            raise ValueError("`upscale_factor` must at least 1")
        if sample_steps < 1:
            raise ValueError("`sample_steps` must be at least 1")
        if strength < 0.0 or strength > 1.0:
            raise ValueError("`strength` must be in the range [0.0, 1.0]")
        if video_frames < 1:
            raise ValueError("`video_frames` must be at least 1")

        # -------------------------------------------
        # CFG Scale
        # -------------------------------------------

        image_cfg_scale = cfg_scale if image_cfg_scale is None else image_cfg_scale
        high_noise_image_cfg_scale = high_noise_cfg_scale if high_noise_image_cfg_scale is None else high_noise_image_cfg_scale

        # -------------------------------------------
        # Set Seed
        # -------------------------------------------

        # Set a random seed if seed is negative
        if seed < 0:
            seed = random.randint(0, 10000)

        # -------------------------------------------
        # Set the Callback Function
        # -------------------------------------------

        if progress_callback is not None:

            @sd_cpp.sd_progress_callback
            def sd_progress_callback(
                step: int,
                steps: int,
                time: float,
                data: ctypes.c_void_p,
            ):
                progress_callback(step, steps, time)

            sd_cpp.sd_set_progress_callback(sd_progress_callback, ctypes.c_void_p(0))

        # -------------------------------------------
        #  High Noise Parameters
        # -------------------------------------------

        high_noise_guidance_params = sd_cpp.sd_guidance_params_t(
            txt_cfg=high_noise_cfg_scale,
            img_cfg=high_noise_image_cfg_scale,
            distilled_guidance=high_noise_guidance,
            slg=sd_cpp.sd_slg_params_t(
                layers=(ctypes.c_int * len(high_noise_skip_layers))(*high_noise_skip_layers),  # Convert to ctypes array
                layer_count=len(high_noise_skip_layers),
                layer_start=high_noise_skip_layer_start,
                layer_end=high_noise_skip_layer_end,
                scale=high_noise_slg_scale,
            ),
        )

        high_noise_sample_params = sd_cpp.sd_sample_params_t(
            guidance=high_noise_guidance_params,
            scheduler=self._validate_and_set_input(high_noise_scheduler, SCHEDULER_MAP, "high_noise_scheduler"),
            sample_method=self._validate_and_set_input(high_noise_sample_method, SAMPLE_METHOD_MAP, "high_noise_sample_method"),
            sample_steps=high_noise_sample_steps,
            eta=high_noise_eta,
        )

        # -------------------------------------------
        # Parameters
        # -------------------------------------------

        guidance_params = sd_cpp.sd_guidance_params_t(
            txt_cfg=cfg_scale,
            img_cfg=image_cfg_scale,
            distilled_guidance=guidance,
            slg=sd_cpp.sd_slg_params_t(
                layers=(ctypes.c_int * len(skip_layers))(*skip_layers),  # Convert to ctypes array
                layer_count=len(skip_layers),
                layer_start=skip_layer_start,
                layer_end=skip_layer_end,
                scale=slg_scale,
            ),
        )

        sample_params = sd_cpp.sd_sample_params_t(
            guidance=guidance_params,
            scheduler=self._validate_and_set_input(scheduler, SCHEDULER_MAP, "scheduler"),
            sample_method=self._validate_and_set_input(sample_method, SAMPLE_METHOD_MAP, "sample_method"),
            sample_steps=sample_steps,
            eta=eta,
        )

        params = sd_cpp.sd_vid_gen_params_t(
            prompt=prompt.encode("utf-8"),
            negative_prompt=negative_prompt.encode("utf-8"),
            clip_skip=clip_skip,
            init_image=self._format_init_image(init_image, width, height),
            end_image=self._format_init_image(end_image, width, height),
            width=width,
            height=height,
            sample_params=sample_params,
            high_noise_sample_params=high_noise_sample_params,
            moe_boundary=moe_boundary,
            strength=strength,
            seed=seed,
            video_frames=video_frames,
        )

        num_results = ctypes.c_int()
        with suppress_stdout_stderr(disable=self.verbose):
            # Generate the video
            c_images = sd_cpp.generate_video(
                self.model,
                ctypes.byref(params),
                ctypes.byref(num_results),
            )

        # Convert C array to Python list of images
        images = self._sd_image_t_p_to_images(c_images, int(num_results.value), upscale_factor)

        # -------------------------------------------
        # Attach Image Metadata
        # -------------------------------------------

        func_args = locals()
        gen_args = {
            k: v
            for k, v in func_args.items()
            if k
            not in {
                "self",
                "images",
                "c_images",
                "progress_callback",
                "sd_progress_callback",
                "num_results",
                "guidance_params",
                "sample_params",
                "high_noise_guidance_params",
                "high_noise_sample_params",
                "params",
            }
        }
        model_args = {k: v for k, v in self.__dict__.items() if not k.startswith("_")}  # Skip internals

        for image in images:
            image.info.update({**model_args, **gen_args})

        return images

    # ===========================================
    # Preprocess Canny
    # ===========================================

    def preprocess_canny(
        self,
        image: Union[Image.Image, str],
        high_threshold: float = 0.08,
        low_threshold: float = 0.08,
        weak: float = 0.8,
        strong: float = 1.0,
        inverse: bool = False,
        output_as_c_uint8: bool = False,
    ) -> Image.Image:
        """Apply canny edge detection to an input image.
        Width and height determined automatically.

        Args:
            image: An input image path or Pillow Image.
            high_threshold: High edge detection threshold.
            low_threshold: Low edge detection threshold.
            weak: Weak edge thickness.
            strong: Strong edge thickness.
            inverse: Invert the edge detection.
            output_as_c_uint8: Return the output as a c_types uint8 pointer.

        Returns:
            A Pillow Image."""

        # Convert the image to a C uint8 pointer
        data, width, height = self._cast_image(image)

        with suppress_stdout_stderr(disable=self.verbose):
            # Run the preprocess canny
            c_image = sd_cpp.preprocess_canny(
                data,
                int(width),
                int(height),
                high_threshold,
                low_threshold,
                weak,
                strong,
                inverse,
            )

        # Return the c_image if output_as_c_uint8 (for running inside txt2img/img2img pipeline)
        if output_as_c_uint8:
            return c_image

        # Calculate the size of the data buffer (channels * width * height)
        buffer_size = 3 * width * height

        # Convert c_image to a Pillow Image
        image = self._c_array_to_bytes(c_image, buffer_size)
        image = self._bytes_to_image(image, width, height)
        return image

    # ===========================================
    # Upscale
    # ===========================================

    def upscale(
        self,
        images: Union[List[Union[Image.Image, str]], Union[Image.Image, str]],
        upscale_factor: int = 4,
        progress_callback: Optional[Callable] = None,
    ) -> List[Image.Image]:
        """Upscale a list of images using the upscaler model.

        Args:
            images: A list of image paths or Pillow Images to upscale.
            upscale_factor: The image upscaling factor.
            progress_callback: Callback function to call on each step end.

        Returns:
            A list of Pillow Images."""

        if self.upscaler is None:
            raise RuntimeError("Upscaling model not loaded")

        # ==================== Set the callback function ====================

        if progress_callback is not None:

            @sd_cpp.sd_progress_callback
            def sd_progress_callback(
                step: int,
                steps: int,
                time: float,
                data: ctypes.c_void_p,
            ):
                progress_callback(step, steps, time)

            sd_cpp.sd_set_progress_callback(sd_progress_callback, ctypes.c_void_p(0))

        if not isinstance(images, list):
            images = [images]  # Wrap single image in a list

        # ==================== Upscale images ====================

        upscaled_images = []
        for image in images:

            # Convert the image to a byte array
            image_bytes = self._image_to_sd_image_t_p(image)

            with suppress_stdout_stderr(disable=self.verbose):
                # Upscale the image
                image = sd_cpp.upscale(
                    self.upscaler,
                    image_bytes,
                    upscale_factor,
                )

            # Load the image from the C sd_image_t and convert it to a PIL Image
            image = self._dereference_sd_image_t_p(image)
            image = self._bytes_to_image(image["data"], image["width"], image["height"])
            upscaled_images.append(image)

        return upscaled_images

    # ===========================================
    # Input Formatting and Validation
    # ===========================================

    # -------------------------------------------
    # Format Control Image
    # -------------------------------------------

    def _format_control_image(
        self, control_image: Optional[Union[Image.Image, str]], canny: bool, width: int, height: int
    ) -> sd_cpp.sd_image_t:
        """Convert an image path or Pillow Image to an C sd_image_t image."""

        if not isinstance(control_image, (str, Image.Image)) or not self.control_net_path:
            if not self.control_net_path:
                log_event(1, "`control_net_path` not set. Skipping control image")

            # Return an empty sd_image_t
            return self._c_uint8_to_sd_image_t_p(
                image=None,
                width=width,
                height=height,
                channel=3,
            )

        if canny:
            # Convert Pillow Image to canny edge detection image then format into C sd_image_t
            image, width, height = self._format_image(control_image)
            image = self.preprocess_canny(image, output_as_c_uint8=True)
            image = self._c_uint8_to_sd_image_t_p(image, width, height)
        else:
            # Convert Pillow Image to C sd_image_t
            image = self._image_to_sd_image_t_p(control_image)
        return image

    # -------------------------------------------
    # Format Init Image
    # -------------------------------------------

    def _format_init_image(self, init_image: Optional[Union[Image.Image, str]], width: int, height: int) -> sd_cpp.sd_image_t:
        if isinstance(init_image, (str, Image.Image)):
            # Input image and generated image must have the same size
            init_image = self._resize_image(init_image, width, height)
            return self._image_to_sd_image_t_p(init_image)  # Convert to byte array
        else:
            # Return an empty sd_image_t
            return self._c_uint8_to_sd_image_t_p(
                image=None,
                width=width,
                height=height,
                channel=3,
            )

    # -------------------------------------------
    # Format Mask Image
    # -------------------------------------------

    def _format_mask_image(self, mask_image: Optional[Union[Image.Image, str]], width: int, height: int) -> sd_cpp.sd_image_t:
        if isinstance(mask_image, (str, Image.Image)):
            # Resize the mask image (ideally it should already match the input image size)
            mask_image = self._resize_image(mask_image, width, height)
            return self._image_to_sd_image_t_p(mask_image, channel=1)  # Convert to byte array
        else:
            # Return a blank white mask image in c_unit8 format
            return self._c_uint8_to_sd_image_t_p(
                image=(ctypes.c_uint8 * (width * height))(*[255] * (width * height)),
                width=width,
                height=height,
                channel=1,
            )

    # -------------------------------------------
    # Format Reference Images
    # -------------------------------------------

    def _format_reference_images(
        self, ref_images: List[Union[Image.Image, str]], width: int, height: int
    ) -> List[sd_cpp.sd_image_t]:
        if not isinstance(ref_images, list):
            ref_images = [ref_images]

        reference_images = []
        for img in ref_images:
            if not isinstance(img, (str, Image.Image)):
                # Skip invalid images
                continue

            # Convert the image to a byte array
            img_ptr = self._image_to_sd_image_t_p(img)
            reference_images.append(img_ptr)

        # Create a contiguous array of sd_image_t
        RefImageArrayType = sd_cpp.sd_image_t * len(reference_images)
        return RefImageArrayType(*reference_images), len(reference_images)

    # -------------------------------------------
    # Validate Dimensions
    # -------------------------------------------

    def _validate_dimensions(self, dimension: Union[int, float], attribute_name: str) -> int:
        dimension = int(dimension)
        if dimension <= 0:
            raise ValueError(f"`{attribute_name}` must be greater than 0")
        return dimension

    # -------------------------------------------
    # Validate and Set Input
    # -------------------------------------------

    def _validate_and_set_input(self, user_input: Union[str, int, float], type_map: Dict, attribute_name: str):
        """Validate an input strinbg or int from a map of strings to integers."""
        if isinstance(user_input, float):
            user_input = int(user_input)  # Convert float to int

        # Handle string input
        if isinstance(user_input, str):
            user_input = user_input.strip().lower()
            if user_input in type_map:
                return int(type_map[user_input])
            else:
                raise ValueError(f"Invalid `{attribute_name}` type '{user_input}'. Must be one of {list(type_map.keys())}.")
        elif isinstance(user_input, int) and user_input in type_map.values():
            return int(user_input)
        else:
            raise ValueError(f"`{attribute_name}` must be a string or an integer matching one of {list(type_map.keys())}")

    # ===========================================
    # Utility Functions
    # ===========================================

    # -------------------------------------------
    # Resize Image
    # -------------------------------------------

    def _resize_image(self, image: Union[Image.Image, str], width: int, height: int) -> Image.Image:
        """Resize an image to a new width and height."""
        image, _, _ = self._format_image(image)

        # Resize the image if the width and height are different
        if image.width != width or image.height != height:
            image = image.resize((width, height), Image.Resampling.BILINEAR)
        return image

    # -------------------------------------------
    # Format Image
    # -------------------------------------------

    def _format_image(
        self,
        image: Union[Image.Image, str],
        channel: int = 3,
    ) -> Image.Image:
        """Convert an image path or Pillow Image to a Pillow Image of RGBA or grayscale (inpainting masks) format."""
        # Convert image path to image if str
        if isinstance(image, str):
            image = Image.open(image)

        if channel == 1:
            # Grayscale the image if channel is 1
            image = image.convert("L")
        else:
            # Convert any non RGBA to RGBA
            if image.format != "PNG":
                image = image.convert("RGBA")

            # Ensure the image is in RGB mode
            if image.mode != "RGB":
                image = image.convert("RGB")

        return image, image.width, image.height

    # -------------------------------------------
    # Image to C uint8 pointer
    # -------------------------------------------

    def _cast_image(self, image: Union[Image.Image, str], channel: int = 3):
        """Cast a PIL Image to a C uint8 pointer."""
        image, width, height = self._format_image(image, channel)

        # Convert the PIL Image to a byte array
        image_bytes = image.tobytes()
        data = ctypes.cast(
            (ctypes.c_byte * len(image_bytes))(*image_bytes),
            ctypes.POINTER(ctypes.c_uint8),
        )
        return data, width, height

    # -------------------------------------------
    # C uint8 pointer to C sd_image_t
    # -------------------------------------------

    def _c_uint8_to_sd_image_t_p(
        self, image: Union[ctypes.c_uint8, None], width: int, height: int, channel: int = 3
    ) -> sd_cpp.sd_image_t:
        """Convert a C uint8 pointer to a C sd_image_t."""
        c_image = sd_cpp.sd_image_t(
            width=width,
            height=height,
            channel=channel,
            data=image,
        )
        return c_image

    # -------------------------------------------
    # Image to C sd_image_t
    # -------------------------------------------

    def _image_to_sd_image_t_p(self, image: Union[Image.Image, str], channel: int = 3) -> sd_cpp.sd_image_t:
        """Convert a PIL Image or image path to a C sd_image_t."""
        data, width, height = self._cast_image(image, channel)
        c_image = self._c_uint8_to_sd_image_t_p(data, width, height, channel)
        return c_image

    # -------------------------------------------
    # C sd_image_t to Image
    # -------------------------------------------

    def _c_array_to_bytes(self, c_array, buffer_size: int) -> bytes:
        return bytearray(ctypes.cast(c_array, ctypes.POINTER(ctypes.c_byte * buffer_size)).contents)

    # -------------------------------------------
    # Dereference C sd_image_t pointer
    # -------------------------------------------

    def _dereference_sd_image_t_p(self, c_image: sd_cpp.sd_image_t) -> Dict:
        """Dereference a C sd_image_t pointer to a Python dictionary with height, width, channel and data (bytes)."""

        # Calculate the size of the data buffer
        buffer_size = c_image.channel * c_image.width * c_image.height

        image = {
            "width": c_image.width,
            "height": c_image.height,
            "channel": c_image.channel,
            "data": self._c_array_to_bytes(c_image.data, buffer_size),
        }
        return image

    # -------------------------------------------
    # Image Slice
    # -------------------------------------------

    def _image_slice(self, c_images: sd_cpp.sd_image_t, count: int, upscale_factor: int) -> List[Dict]:
        """Slice a C array of images."""
        image_array = ctypes.cast(c_images, ctypes.POINTER(sd_cpp.sd_image_t * count)).contents

        images = []

        for i in range(count):
            c_image = image_array[i]

            # Upscale the image
            if upscale_factor > 1:
                c_image = sd_cpp.upscale(
                    self.upscaler,
                    c_image,
                    upscale_factor,
                )

            image = self._dereference_sd_image_t_p(c_image)
            images.append(image)

        # Return the list of images
        return images

    # -------------------------------------------
    # sd_image_t_p to Images
    # -------------------------------------------

    def _sd_image_t_p_to_images(self, c_images: sd_cpp.sd_image_t, count: int, upscale_factor: int) -> List[Image.Image]:
        """Convert C sd_image_t_p images to a Python list of images."""

        # Convert C array to Python list of images
        images = self._image_slice(c_images, count, upscale_factor)

        # Convert each image to PIL Image
        for i in range(len(images)):
            image = images[i]
            images[i] = self._bytes_to_image(image["data"], image["width"], image["height"])

        return images

    # -------------------------------------------
    # Bytes to Image
    # -------------------------------------------

    def _bytes_to_image(self, byte_data: bytes, width: int, height: int, channel: int = 3) -> Image.Image:
        """Convert a byte array to a PIL Image."""
        # Initialize the image with RGBA mode
        image = Image.new("RGBA", (width, height))

        for y in range(height):
            for x in range(width):
                idx = (y * width + x) * channel
                # Dynamically create the color tuple
                color = tuple(byte_data[idx + i] if idx + i < len(byte_data) else 0 for i in range(channel))
                if channel == 1:  # Grayscale
                    color = (color[0],) * 3 + (255,)  # Convert to (R, G, B, A)
                elif channel == 3:  # RGB
                    color = color + (255,)  # Add alpha channel
                elif channel == 4:  # RGBA
                    pass  # Use color as is
                else:
                    raise ValueError(f"Unsupported channel value: '{channel}'")
                # Set the pixel
                image.putpixel((x, y), color)

        return image

    # -------------------------------------------
    # Clean Path
    # -------------------------------------------

    def _clean_path(self, path: str) -> str:
        return os.path.normpath(path.strip()) if path else ""

    # -------------------------------------------
    # State Management
    # -------------------------------------------

    def __setstate__(self, state) -> None:
        self.__init__(**state)

    def close(self) -> None:
        """Explicitly free the model from memory."""
        self._stack.close()

    def __del__(self) -> None:
        self.close()


RNG_TYPE_MAP = {
    "default": RNGType.STD_DEFAULT_RNG,
    "cuda": RNGType.CUDA_RNG,
    "type_count": RNGType.RNG_TYPE_COUNT,
}

SAMPLE_METHOD_MAP = {
    "euler_a": SampleMethod.EULER_A,
    "euler": SampleMethod.EULER,
    "heun": SampleMethod.HEUN,
    "dpm2": SampleMethod.DPM2,
    "dpmpp2s_a": SampleMethod.DPMPP2S_A,
    "dpmpp2m": SampleMethod.DPMPP2M,
    "dpmpp2mv2": SampleMethod.DPMPP2Mv2,
    "ipndm": SampleMethod.IPNDM,
    "ipndm_v": SampleMethod.IPNDM_V,
    "lcm": SampleMethod.LCM,
    "ddim_trailing": SampleMethod.DDIM_TRAILING,
    "tcd": SampleMethod.TCD,
    "sample_method_count": SampleMethod.SAMPLE_METHOD_COUNT,
}

SCHEDULER_MAP = {
    "default": Scheduler.DEFAULT,
    "discrete": Scheduler.DISCRETE,
    "karras": Scheduler.KARRAS,
    "exponential": Scheduler.EXPONENTIAL,
    "ays": Scheduler.AYS,
    "gits": Scheduler.GITS,
    "schedule_count": Scheduler.SCHEDULE_COUNT,
}

GGML_TYPE_MAP = {
    "f32": GGMLType.SD_TYPE_F32,
    "f16": GGMLType.SD_TYPE_F16,
    "q4_0": GGMLType.SD_TYPE_Q4_0,
    "q4_1": GGMLType.SD_TYPE_Q4_1,
    "q5_0": GGMLType.SD_TYPE_Q5_0,
    "q5_1": GGMLType.SD_TYPE_Q5_1,
    "q8_0": GGMLType.SD_TYPE_Q8_0,
    "q8_1": GGMLType.SD_TYPE_Q8_1,
    # k-quantizations
    "q2_k": GGMLType.SD_TYPE_Q2_K,
    "q3_k": GGMLType.SD_TYPE_Q3_K,
    "q4_k": GGMLType.SD_TYPE_Q4_K,
    "q5_k": GGMLType.SD_TYPE_Q5_K,
    "q6_k": GGMLType.SD_TYPE_Q6_K,
    "q8_k": GGMLType.SD_TYPE_Q8_K,
    "iq2_xxs": GGMLType.SD_TYPE_IQ2_XXS,
    "iq2_xs": GGMLType.SD_TYPE_IQ2_XS,
    "iq3_xxs": GGMLType.SD_TYPE_IQ3_XXS,
    "iq1_s": GGMLType.SD_TYPE_IQ1_S,
    "iq4_nl": GGMLType.SD_TYPE_IQ4_NL,
    "iq3_s": GGMLType.SD_TYPE_IQ3_S,
    "iq2_s": GGMLType.SD_TYPE_IQ2_S,
    "iq4_xs": GGMLType.SD_TYPE_IQ4_XS,
    "i8": GGMLType.SD_TYPE_I8,
    "i16": GGMLType.SD_TYPE_I16,
    "i32": GGMLType.SD_TYPE_I32,
    "i64": GGMLType.SD_TYPE_I64,
    "f64": GGMLType.SD_TYPE_F64,
    "iq1_m": GGMLType.SD_TYPE_IQ1_M,
    "bf16": GGMLType.SD_TYPE_BF16,
    # "q4_0_4_4": GGMLType.SD_TYPE_Q4_0_4_4,
    # "q4_0_4_8": GGMLType.SD_TYPE_Q4_0_4_8,
    # "q4_0_8_8": GGMLType.SD_TYPE_Q4_0_8_8,
    "tq1_0": GGMLType.SD_TYPE_TQ1_0,
    "tq2_0": GGMLType.SD_TYPE_TQ2_0,
    # "iq4_nl_4_4": GGMLType.SD_TYPE_IQ4_NL_4_4,
    # "iq4_nl_4_8": GGMLType.SD_TYPE_IQ4_NL_4_8,
    # "iq4_nl_8_8": GGMLType.SD_TYPE_IQ4_NL_8_8,
    "mxfp4": GGMLType.SD_TYPE_MXFP4,
    # Default
    "default": GGMLType.SD_TYPE_COUNT,
}
