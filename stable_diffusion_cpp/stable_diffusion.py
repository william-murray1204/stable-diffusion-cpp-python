import os
import re
import ctypes
import random
import contextlib
import multiprocessing
from ctypes import c_uint32
from typing import Dict, List, Union, Literal, Callable, Optional
from pathlib import Path

from PIL import Image

import stable_diffusion_cpp as sd_cpp
from ._utils import suppress_stdout_stderr
from ._logger import log_event, set_verbose
from ._internals import _UpscalerModel, _StableDiffusionModel
from stable_diffusion_cpp import (
    Preview,
    RNGType,
    GGMLType,
    Scheduler,
    Prediction,
    SDCacheMode,
    SampleMethod,
    LoraApplyMode,
)


class StableDiffusion:
    """High-level Python wrapper for a stable-diffusion.cpp model."""

    def __init__(
        self,
        model_path: str = "",
        clip_l_path: str = "",
        clip_g_path: str = "",
        clip_vision_path: str = "",
        t5xxl_path: str = "",
        llm_path: str = "",
        llm_vision_path: str = "",
        diffusion_model_path: str = "",
        high_noise_diffusion_model_path: str = "",
        vae_path: str = "",
        taesd_path: str = "",
        control_net_path: str = "",
        upscaler_path: str = "",
        upscale_tile_size: int = 128,
        lora_model_dir: str = "",
        embedding_paths: List[str] = [],
        photo_maker_path: str = "",
        tensor_type_rules: str = "",
        vae_decode_only: bool = False,
        n_threads: int = -1,
        wtype: Union[str, GGMLType, int, float] = "default",
        rng_type: Union[str, RNGType, int, float] = "cuda",
        sampler_rng_type: Union[str, RNGType, int, float] = "cuda",
        prediction: Union[str, Prediction, int, float] = "default",
        lora_apply_mode: Union[str, LoraApplyMode, int, float] = "auto",
        offload_params_to_cpu: bool = False,
        enable_mmap: bool = False,
        keep_clip_on_cpu: bool = False,
        keep_control_net_on_cpu: bool = False,
        keep_vae_on_cpu: bool = False,
        diffusion_flash_attn: bool = False,
        tae_preview_only: bool = False,
        diffusion_conv_direct: bool = False,
        vae_conv_direct: bool = False,
        circular_x: bool = False,
        circular_y: bool = False,
        force_sdxl_vae_conv_scale: bool = False,
        chroma_use_dit_mask: bool = True,
        chroma_use_t5_mask: bool = False,
        chroma_t5_mask_pad: int = 1,
        qwen_image_zero_cond_t: bool = False,
        flow_shift: float = float("inf"),
        image_resize_method: str = "crop",
        verbose: bool = True,
    ):
        """Load a stable-diffusion.cpp model from `model_path` or `diffusion_model_path`.

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
            llm_path: Path to the llm text encoder (example: qwenvl2.5 for qwen-image, mistral-small3.2 for flux2).
            llm_vision_path: Path to the llm vit.
            diffusion_model_path: Path to the standalone diffusion model.
            high_noise_diffusion_model_path: Path to the standalone high noise diffusion model.
            vae_path: Path to the standalone vae model.
            taesd_path: Path to the taesd. Using Tiny AutoEncoder for fast decoding (low quality).
            control_net_path: Path to the Control Net model.
            upscaler_path: Path to ESRGAN model (upscale images separately or after generation).
            upscale_tile_size: Tile size for upscaler model.
            lora_model_dir: Lora model directory.
            embedding_paths: List of paths to embedding files.
            photo_maker_path: Path to PhotoMaker model.
            tensor_type_rules: Weight type per tensor pattern (example: "^vae\\.=f16,model\\.=q8_0")
            vae_decode_only: Process vae in decode only mode.
            n_threads: Number of threads to use for generation (default: half the number of CPUs).
            wtype: The weight type (default: automatically determines the weight type of the model file).
            rng_type: Random number generator.
            sampler_rng_type: Random number generator for sampler.
            prediction: Prediction type override.
            lora_apply_mode: The way to apply LoRA, (default: "auto"). In auto mode, if the model weights contain any quantized parameters, the "at_runtime" mode will be used; otherwise, "immediately" will be used. The "immediately" mode may have precision and compatibility issues with quantized parameters, but it usually offers faster inference speed and, in some cases, lower memory usage. The "at_runtime" mode, on the other hand, is exactly the opposite.
            offload_params_to_cpu: Place the weights in RAM to save VRAM, and automatically load them into VRAM when needed.
            enable_mmap: Whether to memory-map model.
            keep_clip_on_cpu: Keep clip in CPU (for low vram).
            keep_control_net_on_cpu: Keep Control Net in CPU (for low vram).
            keep_vae_on_cpu: Keep vae in CPU (for low vram).
            diffusion_flash_attn: Use flash attention in diffusion model (can reduce memory usage significantly). May lower quality or crash if backend not supported.
            tae_preview_only: Prevents usage of taesd for decoding the final image (for use with preview="tae").
            diffusion_conv_direct: Use Conv2d direct in the diffusion model. May crash if backend not supported.
            vae_conv_direct: Use Conv2d direct in the vae model (should improve performance). May crash if backend not supported.
            circular_x: Enable circular RoPE wrapping on x-axis (width) only.
            circular_y: Enable circular RoPE wrapping on y-axis (height) only.
            force_sdxl_vae_conv_scale: Force use of conv scale on SDXL vae.
            chroma_use_dit_mask: Use DiT mask for Chroma.
            chroma_use_t5_mask: Use T5 mask for Chroma.
            chroma_t5_mask_pad: T5 mask padding size of Chroma.
            qwen_image_zero_cond_t: Enable zero_cond_t for Qwen image.
            flow_shift: Shift value for Flow models like SD3.x or WAN (default: auto).
            image_resize_method: Method to resize images for init, mask, control and reference images ("crop" or "resize").
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
        self.llm_path = self._clean_path(llm_path)
        self.llm_vision_path = self._clean_path(llm_vision_path)
        self.diffusion_model_path = self._clean_path(diffusion_model_path)
        self.high_noise_diffusion_model_path = self._clean_path(high_noise_diffusion_model_path)
        self.vae_path = self._clean_path(vae_path)
        self.taesd_path = self._clean_path(taesd_path)
        self.control_net_path = self._clean_path(control_net_path)
        self.upscaler_path = self._clean_path(upscaler_path)
        self.upscale_tile_size = upscale_tile_size
        self.lora_model_dir = self._clean_path(lora_model_dir)
        self.embedding_paths = [self._clean_path(p) for p in embedding_paths]
        self.photo_maker_path = self._clean_path(photo_maker_path)
        self.tensor_type_rules = tensor_type_rules
        self.vae_decode_only = vae_decode_only
        self.n_threads = n_threads
        self.wtype = wtype
        self.rng_type = rng_type
        self.sampler_rng_type = sampler_rng_type
        self.prediction = prediction
        self.lora_apply_mode = lora_apply_mode
        self.offload_params_to_cpu = offload_params_to_cpu
        self.enable_mmap = enable_mmap
        self.keep_clip_on_cpu = keep_clip_on_cpu
        self.keep_control_net_on_cpu = keep_control_net_on_cpu
        self.keep_vae_on_cpu = keep_vae_on_cpu
        self.diffusion_flash_attn = diffusion_flash_attn
        self.tae_preview_only = tae_preview_only
        self.diffusion_conv_direct = diffusion_conv_direct
        self.vae_conv_direct = vae_conv_direct
        self.circular_x = circular_x
        self.circular_y = circular_y
        self.force_sdxl_vae_conv_scale = force_sdxl_vae_conv_scale
        self.chroma_use_dit_mask = chroma_use_dit_mask
        self.chroma_use_t5_mask = chroma_use_t5_mask
        self.chroma_t5_mask_pad = chroma_t5_mask_pad
        self.qwen_image_zero_cond_t = qwen_image_zero_cond_t
        self.flow_shift = flow_shift
        self.image_resize_method = image_resize_method
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
        self.sampler_rng_type = self._validate_and_set_input(self.sampler_rng_type, RNG_TYPE_MAP, "sampler_rng_type")
        self.lora_apply_mode = self._validate_and_set_input(self.lora_apply_mode, LORA_APPLY_MODE_MAP, "lora_apply_mode")
        self.prediction = self._validate_and_set_input(self.prediction, PREDICTION_MAP, "prediction")

        # -------------------------------------------
        # Embeddings
        # -------------------------------------------

        _embedding_items = []
        for p in self.embedding_paths:
            path = Path(p)
            if not path.is_file():
                raise ValueError(f"Embedding not found: {p}")

            _embedding_items.append(
                sd_cpp.sd_embedding_t(
                    name=path.stem.encode("utf-8"),  # Filename minus extension
                    path=str(path).encode("utf-8"),
                )
            )

        if _embedding_items:
            EmbeddingArrayType = sd_cpp.sd_embedding_t * len(self._embedding_items)
            _embedding_array = EmbeddingArrayType(*self._embedding_items)
            _embedding_count = c_uint32(len(self._embedding_items))
        else:
            _embedding_array = None
            _embedding_count = c_uint32(0)

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
                    llm_path=self.llm_path,
                    llm_vision_path=self.llm_vision_path,
                    diffusion_model_path=self.diffusion_model_path,
                    high_noise_diffusion_model_path=self.high_noise_diffusion_model_path,
                    vae_path=self.vae_path,
                    taesd_path=self.taesd_path,
                    control_net_path=self.control_net_path,
                    embeddings=_embedding_array,
                    embedding_count=_embedding_count,
                    photo_maker_path=self.photo_maker_path,
                    tensor_type_rules=self.tensor_type_rules,
                    vae_decode_only=self.vae_decode_only,
                    n_threads=self.n_threads,
                    wtype=self.wtype,
                    rng_type=self.rng_type,
                    sampler_rng_type=self.sampler_rng_type,
                    prediction=self.prediction,
                    lora_apply_mode=self.lora_apply_mode,
                    offload_params_to_cpu=self.offload_params_to_cpu,
                    enable_mmap=self.enable_mmap,
                    keep_clip_on_cpu=self.keep_clip_on_cpu,
                    keep_control_net_on_cpu=self.keep_control_net_on_cpu,
                    keep_vae_on_cpu=self.keep_vae_on_cpu,
                    diffusion_flash_attn=self.diffusion_flash_attn,
                    tae_preview_only=self.tae_preview_only,
                    diffusion_conv_direct=self.diffusion_conv_direct,
                    vae_conv_direct=self.vae_conv_direct,
                    circular_x=self.circular_x,
                    circular_y=self.circular_y,
                    force_sdxl_vae_conv_scale=self.force_sdxl_vae_conv_scale,
                    chroma_use_dit_mask=self.chroma_use_dit_mask,
                    chroma_use_t5_mask=self.chroma_use_t5_mask,
                    chroma_t5_mask_pad=self.chroma_t5_mask_pad,
                    qwen_image_zero_cond_t=self.qwen_image_zero_cond_t,
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
                    tile_size=self.upscale_tile_size,
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
        auto_resize_ref_image: bool = True,
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
        scheduler: Union[str, Scheduler, int, float, None] = "default",
        sample_method: Union[str, SampleMethod, int, float, None] = "default",
        sample_steps: int = 20,
        eta: float = 0.0,
        timestep_shift: int = 0,
        sigmas: Optional[str] = None,
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
        pm_id_embed_path: str = "",
        pm_id_images: Optional[List[Union[Image.Image, str]]] = None,
        pm_style_strength: float = 20.0,
        vae_tiling: bool = False,
        vae_tile_overlap: float = 0.5,
        vae_tile_size: Optional[Union[int, str]] = "0x0",
        vae_relative_tile_size: Optional[Union[float, str]] = "0x0",
        # ---
        cache_mode: Union[str, SDCacheMode, int, float, None] = "disabled",
        cache_reuse_threshold: float = 1.0,
        cache_start_percent: float = 0.15,
        cache_end_percent: float = 0.95,
        cache_error_decay_rate: float = 1.0,
        cache_use_relative_threshold: bool = True,
        cache_reset_error_on_compute: bool = True,
        cache_Fn_compute_blocks: int = 8,
        cache_Bn_compute_blocks: int = 0,
        cache_residual_diff_threshold: float = 0.08,
        cache_max_warmup_steps: int = 8,
        cache_max_continuous_cached_steps: int = -1,
        cache_taylorseer_n_derivatives: int = 1,
        cache_taylorseer_skip_interval: int = 1,
        scm_mask: str = "",
        scm_policy: Literal["dynamic", "static"] = "dynamic",
        # ---
        canny: bool = False,
        upscale_factor: int = 1,
        preview_method: Union[str, Preview, int, float] = "none",
        preview_noisy: bool = False,
        preview_interval: int = 1,
        preview_callback: Optional[Callable] = None,
        progress_callback: Optional[Callable] = None,
    ) -> List[Image.Image]:
        """Generate images from a text prompt and or input images.

        Args:
            prompt: The prompt to render.
            negative_prompt: The negative prompt.
            clip_skip: Ignore last layers of CLIP network (1 ignores none, 2 ignores one layer, <= 0 represents unspecified, will be 1 for SD1.x, 2 for SD2.x).
            init_image: An input image path or Pillow Image to direct the generation.
            ref_images: A list of input image paths or Pillow Images for Flux Kontext models (can be used multiple times).
            auto_resize_ref_image: Automatically resize reference images.
            increase_ref_index: Automatically increase the indices of reference images based on the order they are listed (starting with 1).
            mask_image: The inpainting mask image path or Pillow Image.
            width: Image width, in pixel space.
            height: Image height, in pixel space.
            cfg_scale: Unconditional guidance scale.
            image_cfg_scale: Image guidance scale for inpaint or instruct-pix2pix models.
            guidance: Distilled guidance scale for models with guidance input.
            scheduler: Denoiser sigma scheduler (default: discrete).
            sample_method: Sampling method (default: euler for Flux/SD3/Wan, euler_a otherwise).
            sample_steps: Number of sample steps.
            eta: Eta in DDIM, only for DDIM and TCD.
            timestep_shift: Shift timestep for NitroFusion models, default: 0, recommended N for NitroSD-Realism around 250 and 500 for NitroSD-Vibrant.
            sigmas: Custom sigma values for the sampler, comma-separated (e.g. "14.61,7.8,3.5,0.0").
            skip_layers: Layers to skip for SLG steps (SLG will be enabled at step int([STEPS]x[START]) and disabled at int([STEPS]x[END])).
            skip_layer_start: SLG enabling point.
            skip_layer_end: SLG disabling point.
            slg_scale: Skip layer guidance (SLG) scale, only for DiT models.
            strength: Strength for noising/unnoising.
            seed: RNG seed (uses random seed for < 0).
            batch_count: Number of images to generate.
            control_image: A control condition image path or Pillow Image (Control Net).
            control_strength: Strength to apply Control Net.
            pm_id_embed_path: Path to PhotoMaker v2 id embed.
            pm_id_images: A list of input image paths or Pillow Images for PhotoMaker input identity.
            pm_style_strength: Strength for keeping PhotoMaker input identity.
            vae_tiling: Process vae in tiles to reduce memory usage.
            vae_tile_overlap: Tile overlap for vae tiling, in fraction of tile size.
            vae_tile_size: Tile size for vae tiling ([X]x[Y] format).
            vae_relative_tile_size: Relative tile size for vae tiling, in fraction of image size if < 1, in number of tiles per dim if >=1 ([X]x[Y] format) (overrides `vae_tile_size`).
            cache_mode: The caching method to use (default: disabled).
            scm_mask: SCM steps mask for cache-dit: comma-separated 0/1 (e.g., "1,1,1,0,0,1,0,0,1,0") - 1=compute, 0=can cache.
            scm_policy: SCM policy 'dynamic' or 'static'.
            canny: Apply canny edge detection preprocessor to the `control_image`.
            upscale_factor: Run the ESRGAN upscaler this many times.
            preview_method: The preview method to use (default: none).
            preview_noisy: Enables previewing noisy inputs of the models rather than the denoised outputs.
            preview_interval: Interval in denoising steps between consecutive updates of the image preview (default: 1, meaning update at every step)
            preview_callback: Callback function to call on each preview frame.
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
        if timestep_shift < 0 or timestep_shift > 1000:
            raise ValueError("`timestep_shift` must be in the range [0, 1000]")

        # -------------------------------------------
        # Set CFG Scale
        # -------------------------------------------

        image_cfg_scale = cfg_scale if image_cfg_scale is None else image_cfg_scale

        # -------------------------------------------
        # Set Seed
        # -------------------------------------------

        # Set a random seed if seed is negative
        if seed < 0:
            seed = random.randint(0, 10000)

        # -------------------------------------------
        # Set the Progress Callback Function
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
        # Set the Preview Callback Function
        # -------------------------------------------

        preview_method = self._validate_and_set_input(preview_method, PREVIEW_MAP, "preview_method")

        if preview_callback is not None:

            @sd_cpp.sd_preview_callback
            def sd_preview_callback(
                step: int,
                frame_count: int,
                frames: sd_cpp.sd_image_t,
                is_noisy: ctypes.c_bool,
                data: ctypes.c_void_p,
            ):
                pil_frames = self._sd_image_t_p_to_images(frames, frame_count, 1)
                preview_callback(step, pil_frames, is_noisy)

            sd_cpp.sd_set_preview_callback(
                sd_preview_callback,
                preview_method,
                preview_interval,
                not preview_noisy,
                preview_noisy,
                ctypes.c_void_p(0),
            )

        # -------------------------------------------
        # Extract Loras
        # -------------------------------------------

        _prompt_without_loras, _lora_array, _lora_count, _lora_string_buffers = self._extract_and_build_loras(
            prompt,
            self.lora_model_dir,
        )

        # -------------------------------------------
        # Reference Images
        # -------------------------------------------

        _ref_images_pointer, ref_images_count = self._create_image_array(
            ref_images, resize=False
        )  # Disable resize, sd.cpp handles it
        _id_images_pointer, id_images_count = self._create_image_array(pm_id_images)

        # -------------------------------------------
        # Vae Tiling
        # -------------------------------------------

        tile_size_x, tile_size_y = self._parse_tile_size(vae_tile_size, as_float=False)
        rel_size_x, rel_size_y = self._parse_tile_size(vae_relative_tile_size, as_float=True)

        # -------------------------------------------
        # Scheduler/Sample Method
        # -------------------------------------------

        scheduler = self._validate_and_set_input(scheduler, SCHEDULER_MAP, "scheduler", allow_none=True)
        if scheduler is None:
            scheduler = sd_cpp.sd_get_default_scheduler(self.model)

        sample_method = self._validate_and_set_input(sample_method, SAMPLE_METHOD_MAP, "sample_method", allow_none=True)
        if sample_method is None:
            sample_method = sd_cpp.sd_get_default_sample_method(self.model)

        # -------------------------------------------
        # Sigmas
        # -------------------------------------------

        _custom_sigmas = self._parse_sigmas(sigmas)
        _custom_sigmas_count = len(_custom_sigmas)

        SigmasArrayType = ctypes.c_float * _custom_sigmas_count
        _custom_sigmas = ctypes.cast(SigmasArrayType(*_custom_sigmas), ctypes.POINTER(ctypes.c_float))

        # -------------------------------------------
        # Cache
        # -------------------------------------------

        cache_mode = self._validate_and_set_input(cache_mode, SD_CACHE_MODE_MAP, "cache_mode")
        scm_policy = self._validate_and_set_input(scm_policy, {"dynamic": True, "static": False}, "scm_policy")

        # If default reuse threshold and mode is easycache, set to 0.2
        cache_reuse_threshold = (
            0.2 if cache_mode == SDCacheMode.SD_CACHE_EASYCACHE and cache_reuse_threshold == 1.0 else cache_reuse_threshold
        )

        # -------------------------------------------
        # Parameters
        # -------------------------------------------

        _cache_params = sd_cpp.sd_cache_params_t(
            mode=cache_mode,
            reuse_threshold=cache_reuse_threshold,
            start_percent=cache_start_percent,
            end_percent=cache_end_percent,
            error_decay_rate=cache_error_decay_rate,
            use_relative_threshold=cache_use_relative_threshold,
            reset_error_on_compute=cache_reset_error_on_compute,
            Fn_compute_blocks=cache_Fn_compute_blocks,
            Bn_compute_blocks=cache_Bn_compute_blocks,
            residual_diff_threshold=cache_residual_diff_threshold,
            max_warmup_steps=cache_max_warmup_steps,
            max_continuous_cached_steps=cache_max_continuous_cached_steps,
            taylorseer_n_derivatives=cache_taylorseer_n_derivatives,
            taylorseer_skip_interval=cache_taylorseer_skip_interval,
            scm_mask=scm_mask.encode("utf-8"),
            scm_policy_dynamic=scm_policy,
        )

        _pm_params = sd_cpp.sd_pm_params_t(
            id_images=_id_images_pointer,
            id_images_count=id_images_count,
            id_embed_path=pm_id_embed_path.encode("utf-8"),
            style_strength=pm_style_strength,
        )

        _vae_tiling_params = sd_cpp.sd_tiling_params_t(
            enabled=vae_tiling,
            tile_size_x=tile_size_x,
            tile_size_y=tile_size_y,
            target_overlap=vae_tile_overlap,
            rel_size_x=rel_size_x,
            rel_size_y=rel_size_y,
        )

        _guidance_params = sd_cpp.sd_guidance_params_t(
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

        _sample_params = sd_cpp.sd_sample_params_t(
            guidance=_guidance_params,
            scheduler=scheduler,
            sample_method=sample_method,
            sample_steps=sample_steps,
            eta=eta,
            shifted_timestep=timestep_shift,
            custom_sigmas=_custom_sigmas,
            custom_sigmas_count=_custom_sigmas_count,
        )

        _params = sd_cpp.sd_img_gen_params_t(
            loras=_lora_array,
            lora_count=_lora_count,
            prompt=_prompt_without_loras.encode("utf-8"),
            negative_prompt=negative_prompt.encode("utf-8"),
            clip_skip=clip_skip,
            init_image=self._format_init_image(init_image, width, height),
            ref_images=_ref_images_pointer,
            auto_resize_ref_image=auto_resize_ref_image,
            ref_images_count=ref_images_count,
            increase_ref_index=increase_ref_index,
            mask_image=self._format_mask_image(mask_image, width, height),
            width=width,
            height=height,
            sample_params=_sample_params,
            strength=strength,
            seed=seed,
            batch_count=batch_count,
            control_image=self._format_control_image(control_image, canny, width, height),
            control_strength=control_strength,
            pm_params=_pm_params,
            vae_tiling_params=_vae_tiling_params,
            cache=_cache_params,
        )

        # Log system info
        log_event(level=2, message=sd_cpp.sd_get_system_info().decode("utf-8"))

        with suppress_stdout_stderr(disable=self.verbose):
            # Generate images
            _c_images = sd_cpp.generate_image(
                self.model,
                ctypes.byref(_params),
            )

        # Convert C array to Python list of images
        images = self._sd_image_t_p_to_images(_c_images, batch_count, upscale_factor)

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
                "progress_callback",
                "sd_progress_callback",
                "preview_callback",
                "sd_preview_callback",
            }
            and not k.startswith("_")  # Skip internals
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
        control_frames: Optional[List[Union[Image.Image, str]]] = None,
        width: int = 512,
        height: int = 512,
        # ---
        # guidance_params
        cfg_scale: float = 7.0,
        image_cfg_scale: Optional[float] = None,
        guidance: float = 3.5,
        # sample_params
        scheduler: Union[str, Scheduler, int, float, None] = "default",
        sample_method: Optional[Union[str, SampleMethod, int, float, None]] = "default",
        sample_steps: int = 20,
        eta: float = 0.0,
        timestep_shift: int = 0,
        sigmas: Optional[str] = None,
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
        high_noise_scheduler: Union[str, Scheduler, int, float, None] = "default",
        high_noise_sample_method: Union[str, SampleMethod, int, float, None] = "default",
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
        vace_strength: int = 1,
        vae_tiling: bool = False,
        vae_tile_overlap: float = 0.5,
        vae_tile_size: Optional[Union[int, str]] = "0x0",
        vae_relative_tile_size: Optional[Union[float, str]] = "0x0",
        # ---
        cache_mode: Union[str, SDCacheMode, int, float, None] = "disabled",
        cache_reuse_threshold: float = 1.0,
        cache_start_percent: float = 0.15,
        cache_end_percent: float = 0.95,
        cache_error_decay_rate: float = 1.0,
        cache_use_relative_threshold: bool = True,
        cache_reset_error_on_compute: bool = True,
        cache_Fn_compute_blocks: int = 8,
        cache_Bn_compute_blocks: int = 0,
        cache_residual_diff_threshold: float = 0.08,
        cache_max_warmup_steps: int = 8,
        cache_max_continuous_cached_steps: int = -1,
        cache_taylorseer_n_derivatives: int = 1,
        cache_taylorseer_skip_interval: int = 1,
        scm_mask: str = "",
        scm_policy: Literal["dynamic", "static"] = "dynamic",
        # ---
        upscale_factor: int = 1,
        preview_method: Union[str, Preview, int, float] = "none",
        preview_noisy: bool = False,
        preview_interval: int = 1,
        preview_callback: Optional[Callable] = None,
        progress_callback: Optional[Callable] = None,
    ) -> List[Image.Image]:
        """Generate a video from input images and or a text prompt.

        Args:
            prompt: The prompt to render.
            negative_prompt: The negative prompt.
            clip_skip: Ignore last layers of CLIP network (1 ignores none, 2 ignores one layer, <= 0 represents unspecified, will be 1 for SD1.x, 2 for SD2.x).
            init_image: An input image path or Pillow Image to start the generation.
            end_image: An input image path or Pillow Image to end the generation (required by flf2v).
            control_frames: A list of control video frame image paths or Pillow Images in the correct order for the video.
            width: Video width, in pixel space.
            height: Video height, in pixel space.
            cfg_scale: Unconditional guidance scale.
            image_cfg_scale: Image guidance scale for inpaint or instruct-pix2pix models (default: same as `cfg_scale`).
            guidance: Distilled guidance scale for models with guidance input.
            scheduler: Denoiser sigma scheduler (default: discrete).
            sample_method: Sampling method (default: euler for Flux/SD3/Wan, euler_a otherwise).
            sample_steps: Number of sample steps.
            eta: Eta in DDIM, only for DDIM and TCD.
            timestep_shift: Shift timestep for NitroFusion models, default: 0, recommended N for NitroSD-Realism around 250 and 500 for NitroSD-Vibrant.
            sigmas: Custom sigma values for the sampler, comma-separated (e.g. "14.61,7.8,3.5,0.0").
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
            vace_strength: Wan VACE strength.
            vae_tiling: Process vae in tiles to reduce memory usage.
            vae_tile_overlap: Tile overlap for vae tiling, in fraction of tile size.
            vae_tile_size: Tile size for vae tiling ([X]x[Y] format).
            vae_relative_tile_size: Relative tile size for vae tiling, in fraction of image size if < 1, in number of tiles per dim if >=1 ([X]x[Y] format) (overrides `vae_tile_size`).
            cache_mode: The caching method to use (default: disabled).
            scm_mask: SCM steps mask for cache-dit: comma-separated 0/1 (e.g., "1,1,1,0,0,1,0,0,1,0") - 1=compute, 0=can cache.
            scm_policy: SCM policy 'dynamic' or 'static'.
            upscale_factor: Run the ESRGAN upscaler this many times.
            preview_method: The preview method to use (default: none).
            preview_noisy: Enables previewing noisy inputs of the models rather than the denoised outputs.
            preview_interval: Interval in denoising steps between consecutive updates of the image preview (default: 1, meaning update at every step)
            preview_callback: Callback function to call on each preview frame.
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
        if timestep_shift < 0 or timestep_shift > 1000:
            raise ValueError("`timestep_shift` must be in the range [0, 1000]")

        if high_noise_sample_steps <= 0:
            high_noise_sample_steps = -1  # Auto

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
        # Set the Progress Callback Function
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
        # Set the Preview Callback Function
        # -------------------------------------------

        preview_method = self._validate_and_set_input(preview_method, PREVIEW_MAP, "preview_method")

        if preview_callback is not None:

            @sd_cpp.sd_preview_callback
            def sd_preview_callback(
                step: int,
                frame_count: int,
                frames: sd_cpp.sd_image_t,
                is_noisy: ctypes.c_bool,
                data: ctypes.c_void_p,
            ):
                pil_frames = self._sd_image_t_p_to_images(frames, frame_count, 1)
                preview_callback(step, pil_frames, is_noisy)

            sd_cpp.sd_set_preview_callback(
                sd_preview_callback,
                preview_method,
                preview_interval,
                not preview_noisy,
                preview_noisy,
                ctypes.c_void_p(0),
            )

        # -------------------------------------------
        # Extract Loras
        # -------------------------------------------

        _prompt_without_loras, _lora_array, _lora_count, _lora_string_buffers = self._extract_and_build_loras(
            prompt,
            self.lora_model_dir,
        )

        # -------------------------------------------
        # Control Frames
        # -------------------------------------------

        _control_frames_pointer, control_frames_size = self._create_image_array(
            control_frames,
            width=width,
            height=height,
            max_images=video_frames,
        )

        # -------------------------------------------
        # Vae Tiling
        # -------------------------------------------

        tile_size_x, tile_size_y = self._parse_tile_size(vae_tile_size, as_float=False)
        rel_size_x, rel_size_y = self._parse_tile_size(vae_relative_tile_size, as_float=True)

        # -------------------------------------------
        # Scheduler/Sample Method
        # -------------------------------------------

        scheduler = self._validate_and_set_input(scheduler, SCHEDULER_MAP, "scheduler", allow_none=True)
        if scheduler is None:
            scheduler = sd_cpp.sd_get_default_scheduler(self.model)

        # "sample_method_count" is not valid here (it will crash)
        sample_method = self._validate_and_set_input(
            sample_method,
            {k: v for k, v in SAMPLE_METHOD_MAP.items() if k not in ["sample_method_count"]},
            "sample_method",
            allow_none=True,
        )
        if sample_method is None:
            sample_method = sd_cpp.sd_get_default_sample_method(self.model)

        # High Noise
        high_noise_scheduler = self._validate_and_set_input(
            high_noise_scheduler, SCHEDULER_MAP, "high_noise_scheduler", allow_none=True
        )
        if high_noise_scheduler is None:
            high_noise_scheduler = scheduler

        high_noise_sample_method = self._validate_and_set_input(
            high_noise_sample_method, SAMPLE_METHOD_MAP, "high_noise_sample_method", allow_none=True
        )
        if high_noise_sample_method is None:
            high_noise_sample_method = sample_method

        # -------------------------------------------
        # Sigmas
        # -------------------------------------------

        _custom_sigmas = self._parse_sigmas(sigmas)
        _custom_sigmas_count = len(_custom_sigmas)

        SigmasArrayType = ctypes.c_float * _custom_sigmas_count
        _custom_sigmas = ctypes.cast(SigmasArrayType(*_custom_sigmas), ctypes.POINTER(ctypes.c_float))

        # -------------------------------------------
        # Cache
        # -------------------------------------------

        cache_mode = self._validate_and_set_input(cache_mode, SD_CACHE_MODE_MAP, "cache_mode")
        scm_policy = self._validate_and_set_input(scm_policy, {"dynamic": True, "static": False}, "scm_policy")

        # If default reuse threshold and mode is easycache, set to 0.2
        cache_reuse_threshold = (
            0.2 if cache_mode == SDCacheMode.SD_CACHE_EASYCACHE and cache_reuse_threshold == 1.0 else cache_reuse_threshold
        )

        # -------------------------------------------
        #  High Noise Parameters
        # -------------------------------------------

        _high_noise_guidance_params = sd_cpp.sd_guidance_params_t(
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

        _high_noise_sample_params = sd_cpp.sd_sample_params_t(
            guidance=_high_noise_guidance_params,
            scheduler=high_noise_scheduler,
            sample_method=high_noise_sample_method,
            sample_steps=high_noise_sample_steps,
            eta=high_noise_eta,
            shifted_timestep=timestep_shift,
            custom_sigmas=_custom_sigmas,
            custom_sigmas_count=_custom_sigmas_count,
        )

        # -------------------------------------------
        # Parameters
        # -------------------------------------------

        _cache_params = sd_cpp.sd_cache_params_t(
            mode=cache_mode,
            reuse_threshold=cache_reuse_threshold,
            start_percent=cache_start_percent,
            end_percent=cache_end_percent,
            error_decay_rate=cache_error_decay_rate,
            use_relative_threshold=cache_use_relative_threshold,
            reset_error_on_compute=cache_reset_error_on_compute,
            Fn_compute_blocks=cache_Fn_compute_blocks,
            Bn_compute_blocks=cache_Bn_compute_blocks,
            residual_diff_threshold=cache_residual_diff_threshold,
            max_warmup_steps=cache_max_warmup_steps,
            max_continuous_cached_steps=cache_max_continuous_cached_steps,
            taylorseer_n_derivatives=cache_taylorseer_n_derivatives,
            taylorseer_skip_interval=cache_taylorseer_skip_interval,
            scm_mask=scm_mask.encode("utf-8"),
            scm_policy_dynamic=scm_policy,
        )

        _vae_tiling_params = sd_cpp.sd_tiling_params_t(
            enabled=vae_tiling,
            tile_size_x=tile_size_x,
            tile_size_y=tile_size_y,
            target_overlap=vae_tile_overlap,
            rel_size_x=rel_size_x,
            rel_size_y=rel_size_y,
        )

        _guidance_params = sd_cpp.sd_guidance_params_t(
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

        _sample_params = sd_cpp.sd_sample_params_t(
            guidance=_guidance_params,
            scheduler=scheduler,
            sample_method=sample_method,
            sample_steps=sample_steps,
            eta=eta,
            shifted_timestep=timestep_shift,
            custom_sigmas=_custom_sigmas,
            custom_sigmas_count=_custom_sigmas_count,
        )

        _params = sd_cpp.sd_vid_gen_params_t(
            loras=_lora_array,
            lora_count=_lora_count,
            prompt=_prompt_without_loras.encode("utf-8"),
            negative_prompt=negative_prompt.encode("utf-8"),
            clip_skip=clip_skip,
            init_image=self._format_init_image(init_image, width, height),
            end_image=self._format_init_image(end_image, width, height),
            control_frames=_control_frames_pointer,
            control_frames_size=control_frames_size,
            width=width,
            height=height,
            sample_params=_sample_params,
            high_noise_sample_params=_high_noise_sample_params,
            moe_boundary=moe_boundary,
            strength=strength,
            seed=seed,
            video_frames=video_frames,
            vace_strength=vace_strength,
            vae_tiling_params=_vae_tiling_params,
            cache=_cache_params,
        )

        # Log system info
        log_event(level=2, message=sd_cpp.sd_get_system_info().decode("utf-8"))

        _num_results = ctypes.c_int()
        with suppress_stdout_stderr(disable=self.verbose):
            # Generate the video
            _c_images = sd_cpp.generate_video(
                self.model,
                ctypes.byref(_params),
                ctypes.byref(_num_results),
            )

        # Convert C array to Python list of images
        images = self._sd_image_t_p_to_images(_c_images, int(_num_results.value), upscale_factor)

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
                "progress_callback",
                "sd_progress_callback",
                "preview_callback",
                "sd_preview_callback",
            }
            and not k.startswith("_")  # Skip internals
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
        output_as_sd_image_t: bool = False,
    ) -> Union[Image.Image, sd_cpp.sd_image_t]:
        """Apply canny edge detection to an input image.
        Width and height determined automatically.

        Args:
            image: An input image path or Pillow Image.
            high_threshold: High edge detection threshold.
            low_threshold: Low edge detection threshold.
            weak: Weak edge thickness.
            strong: Strong edge thickness.
            inverse: Invert the edge detection.
            output_as_sd_image_t: Return the output as a c_types sd_image_t pointer.

        Returns:
            A Pillow Image."""

        # Convert the image to a byte array
        image_bytes = self._image_to_sd_image_t_p(image)

        with suppress_stdout_stderr(disable=self.verbose):
            # Apply the preprocess canny
            sd_cpp.preprocess_canny(
                image_bytes,
                high_threshold,
                low_threshold,
                weak,
                strong,
                inverse,
            )

        # Return the sd_image_t if output_as_sd_image_t (for running inside txt2img/img2img pipeline)
        if output_as_sd_image_t:
            return image_bytes

        # Load the image from the C sd_image_t and convert it to a PIL Image
        image = self._dereference_sd_image_t_p(image_bytes)
        image = self._bytes_to_image(image["data"], image["width"], image["height"])
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

        # NOTE: Preview callback not supported for upscaling (nothing is called back from sd.cpp)

        # -------------------------------------------
        # Ensure List of Images
        # -------------------------------------------

        if not isinstance(images, list):
            images = [images]  # Wrap single image in a list

        # -------------------------------------------
        # Upscale Images
        # -------------------------------------------

        # Log system info
        log_event(level=2, message=sd_cpp.sd_get_system_info().decode("utf-8"))

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
    # Convert
    # ===========================================

    def convert(
        self,
        input_path: str,
        vae_path: str = "",
        output_path: str = "output.gguf",
        output_type: Union[str, GGMLType, int, float] = "default",
        tensor_type_rules: str = "",
        convert_name: bool = True,
    ) -> bool:
        """Convert a model to gguf format.

        Args:
            input_path: Path to the input model.
            vae_path: Path to the vae.
            output_path: Path to save the converted model.
            output_type: The weight type (default: auto).
            tensor_type_rules: Weight type per tensor pattern (example: "^vae\\\\.=f16,model\\\\.=q8_0")
            convert_name: Convert tensor name.

        Returns:
            A boolean indicating success."""

        # -------------------------------------------
        # Validation
        # -------------------------------------------

        output_type = self._validate_and_set_input(output_type, GGML_TYPE_MAP, "output_type")

        # -------------------------------------------
        # Convert the Model
        # -------------------------------------------

        # Log system info
        log_event(level=2, message=sd_cpp.sd_get_system_info().decode("utf-8"))

        with suppress_stdout_stderr(disable=self.verbose):
            model_converted = sd_cpp.convert(
                self._clean_path(input_path).encode("utf-8"),
                self._clean_path(vae_path).encode("utf-8"),
                self._clean_path(output_path).encode("utf-8"),
                output_type,
                tensor_type_rules.encode("utf-8"),
                convert_name,
            )

        return model_converted

    # ===========================================
    # Input Formatting and Validation
    # ===========================================

    # -------------------------------------------
    # Extract and Remove Lora
    # -------------------------------------------

    def _extract_and_build_loras(self, prompt: str, lora_model_dir: str):
        re_lora = re.compile(r"<lora:([^:>]+):([^>]+)>")
        valid_ext = [".pt", ".safetensors", ".gguf"]

        lora_map = {}
        high_noise_lora_map = {}

        tmp = prompt

        while True:
            m = re_lora.search(tmp)
            if not m:
                break

            raw_path = m.group(1)
            raw_mul = m.group(2)

            try:
                mul = float(raw_mul)
            except ValueError:
                prompt = re_lora.sub("", prompt, count=1)
                tmp = tmp[m.end() :]
                continue

            is_high_noise = False
            prefix = "|high_noise|"

            if raw_path.startswith(prefix):
                raw_path = raw_path[len(prefix) :]
                is_high_noise = True

            path = Path(raw_path)
            final_path = path if path.is_absolute() else Path(lora_model_dir) / path

            if not final_path.exists():
                found = False
                for ext in valid_ext:
                    try_path = final_path.with_suffix(final_path.suffix + ext)
                    if try_path.exists():
                        final_path = try_path
                        found = True
                        break
                if not found:
                    log_event(level=1, message=f"Can not find lora {final_path}")
                    prompt = re_lora.sub("", prompt, count=1)
                    tmp = tmp[m.end() :]
                    continue

            key = str(final_path.resolve())
            target = high_noise_lora_map if is_high_noise else lora_map
            target[key] = target.get(key, 0.0) + mul

            prompt = re_lora.sub("", prompt, count=1)
            tmp = tmp[m.end() :]

        # Build ctypes array
        all_items = []
        for path, mul in lora_map.items():
            all_items.append((False, mul, path))

        for path, mul in high_noise_lora_map.items():
            all_items.append((True, mul, path))

        count = len(all_items)
        LoraArray = sd_cpp.sd_lora_t * count
        lora_array = LoraArray()

        # IMPORTANT: keep string buffers alive
        string_buffers = []

        for i, (is_high_noise, mul, path) in enumerate(all_items):
            buf = ctypes.create_string_buffer(path.encode("utf-8"))
            string_buffers.append(buf)

            lora_array[i].is_high_noise = is_high_noise
            lora_array[i].multiplier = mul
            lora_array[i].path = ctypes.cast(buf, ctypes.c_char_p)

        return prompt, lora_array, count, string_buffers

    # -------------------------------------------
    # Parse Sigmas
    # -------------------------------------------

    def _parse_sigmas(self, sigmas: str) -> list[float]:
        if not sigmas:
            return []

        # Strip surrounding brackets
        sigmas = sigmas.strip()
        if sigmas.startswith("["):
            sigmas = sigmas[1:]
        if sigmas.endswith("]"):
            sigmas = sigmas[:-1]

        custom_sigmas: list[float] = []

        for item in sigmas.split(","):
            item = item.strip()
            if not item:
                continue

            try:
                custom_sigmas.append(float(item))
            except ValueError as e:
                raise ValueError(f"Invalid float value '{item}' in sigmas") from e

        if not custom_sigmas and sigmas:
            raise ValueError(f"Could not parse any sigma values from '{sigmas}'")

        return custom_sigmas

    # -------------------------------------------
    # Parse Tile Size
    # -------------------------------------------

    def _parse_tile_size(self, value: Optional[Union[str, float, int]], as_float: bool = False) -> tuple:
        if not value:
            return (0.0, 0.0) if as_float else (0, 0)

        try:
            if "x" in value:
                x_str, y_str = value.split("x", 1)
                x = float(x_str) if as_float else int(x_str)
                y = float(y_str) if as_float else int(y_str)
            else:
                v = float(value) if as_float else int(value)
                x = y = v
        except (ValueError, OverflowError):
            raise ValueError(f"Invalid tile size value: {value}")

        return (x, y)

    # -------------------------------------------
    # Format Control Image
    # -------------------------------------------

    def _format_control_image(
        self, control_image: Optional[Union[Image.Image, str]], canny: bool, width: int, height: int
    ) -> sd_cpp.sd_image_t:
        """Convert an image path or Pillow Image to an C sd_image_t image."""

        if not isinstance(control_image, (str, Image.Image)):
            # Return an empty sd_image_t
            return self._c_uint8_to_sd_image_t_p(
                image=None,
                width=width,
                height=height,
                channel=3,
            )

        if canny:
            # Apply canny edge detection preprocessor to Pillow Image
            image, width, height = self._format_image(control_image)
            image = self.preprocess_canny(image, output_as_sd_image_t=True)
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
    # Create Image Array
    # -------------------------------------------

    def _create_image_array(
        self,
        images: List[Union[Image.Image, str]],
        width: Optional[int] = None,
        height: Optional[int] = None,
        max_images: Optional[int] = None,
        resize: bool = True,
    ) -> List[sd_cpp.sd_image_t]:
        if not isinstance(images, list):
            images = [images]

        # Enforce max_images
        if max_images is not None and max_images > 0:
            images = images[:max_images]

        reference_images = []
        for img in images:
            if not isinstance(img, (str, Image.Image)):
                # Skip invalid images
                continue

            if width and height and resize == True:
                # Resize if width and height are provided
                img = self._resize_image(img, width=width, height=height)

            # Convert the image to a byte array
            img_ptr = self._image_to_sd_image_t_p(img)
            reference_images.append(img_ptr)

        # Create a contiguous array of sd_image_t
        ImageArrayType = sd_cpp.sd_image_t * len(reference_images)
        return ImageArrayType(*reference_images), len(reference_images)

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

    def _validate_and_set_input(
        self, user_input: Union[str, int, float, None], type_map: Dict, attribute_name: str, allow_none: bool = False
    ) -> Optional[int]:
        """Validate an input strinbg or int from a map of strings to integers."""
        if user_input is None and allow_none == True:
            return None

        if isinstance(user_input, float):
            user_input = int(user_input)  # Convert float to int

        # Handle string input
        if isinstance(user_input, str):
            user_input = user_input.strip().lower()
            if user_input in type_map:
                map_result = type_map[user_input]
                if map_result is None:
                    return None

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

    def _resize_image(
        self,
        image: Union[Image.Image, str],
        width: int,
        height: int,
    ) -> Image.Image:
        image, _, _ = self._format_image(image)

        if image.width == width and image.height == height:
            return image

        if self.image_resize_method == "resize":
            return image.resize((width, height), Image.Resampling.BILINEAR)

        elif self.image_resize_method == "crop":
            src_w, src_h = image.width, image.height
            src_aspect = src_w / src_h
            dst_aspect = width / height

            # Default crop box is full image
            crop_x, crop_y = 0, 0
            crop_w, crop_h = src_w, src_h

            if src_aspect > dst_aspect:
                # Source is wider than destination -> crop width
                crop_w = int(src_h * dst_aspect)
                crop_x = (src_w - crop_w) // 2
            elif src_aspect < dst_aspect:
                # Source is taller than destination -> crop height
                crop_h = int(src_w / dst_aspect)
                crop_y = (src_h - crop_h) // 2

            # Crop first, then resize
            image = image.crop((crop_x, crop_y, crop_x + crop_w, crop_y + crop_h))
            return image.resize((width, height), Image.Resampling.BILINEAR)

        else:
            raise ValueError(f"Invalid `image_resize_method` '{self.image_resize_method}', must be 'resize' or 'crop'")

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
            image = Image.open(self._clean_path(image))

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
    "cuda": RNGType.CUDA_RNG,  # Default
    "cpu": RNGType.CPU_RNG,
    "type_count": RNGType.RNG_TYPE_COUNT,
}

SAMPLE_METHOD_MAP = {
    "default": None,  # Default
    "euler": SampleMethod.EULER_SAMPLE_METHOD,
    "euler_a": SampleMethod.EULER_A_SAMPLE_METHOD,
    "heun": SampleMethod.HEUN_SAMPLE_METHOD,
    "dpm2": SampleMethod.DPM2_SAMPLE_METHOD,
    "dpm++2s_a": SampleMethod.DPMPP2S_A_SAMPLE_METHOD,
    "dpm++2m": SampleMethod.DPMPP2M_SAMPLE_METHOD,
    "dpm++2mv2": SampleMethod.DPMPP2Mv2_SAMPLE_METHOD,
    "ipndm": SampleMethod.IPNDM_SAMPLE_METHOD,
    "ipndm_v": SampleMethod.IPNDM_V_SAMPLE_METHOD,
    "lcm": SampleMethod.LCM_SAMPLE_METHOD,
    "ddim_trailing": SampleMethod.DDIM_TRAILING_SAMPLE_METHOD,
    "tcd": SampleMethod.TCD_SAMPLE_METHOD,
    "sample_method_count": SampleMethod.SAMPLE_METHOD_COUNT,
}

SCHEDULER_MAP = {
    "default": None,  # Default
    "discrete": Scheduler.DISCRETE_SCHEDULER,
    "karras": Scheduler.KARRAS_SCHEDULER,
    "exponential": Scheduler.EXPONENTIAL_SCHEDULER,
    "ays": Scheduler.AYS_SCHEDULER,
    "gits": Scheduler.GITS_SCHEDULER,
    "sgm_uniform": Scheduler.SGM_UNIFORM_SCHEDULER,
    "simple": Scheduler.SIMPLE_SCHEDULER,
    "smoothstep": Scheduler.SMOOTHSTEP_SCHEDULER,
    "kl_optimal": Scheduler.KL_OPTIMAL_SCHEDULER,
    "lcm": Scheduler.LCM_SCHEDULER,
    "scheduler_count": Scheduler.SCHEDULER_COUNT,
}

PREDICTION_MAP = {
    "eps": Prediction.EPS_PRED,
    "v": Prediction.V_PRED,
    "edm_v": Prediction.EDM_V_PRED,
    "flow": Prediction.FLOW_PRED,
    "flux_flow": Prediction.FLUX_FLOW_PRED,
    "flux2_flow": Prediction.FLUX2_FLOW_PRED,
    "default": Prediction.PREDICTION_COUNT,  # Default
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
    "default": GGMLType.SD_TYPE_COUNT,  # Default
}

PREVIEW_MAP = {
    "none": Preview.PREVIEW_NONE,  # Default
    "proj": Preview.PREVIEW_PROJ,
    "tae": Preview.PREVIEW_TAE,
    "vae": Preview.PREVIEW_VAE,
    "preview_count": Preview.PREVIEW_COUNT,
}

LORA_APPLY_MODE_MAP = {
    "auto": LoraApplyMode.LORA_APPLY_AUTO,  # Default
    "immediately": LoraApplyMode.LORA_APPLY_IMMEDIATELY,
    "at_runtime": LoraApplyMode.LORA_APPLY_AT_RUNTIME,
    "lora_apply_mode_count": LoraApplyMode.LORA_APPLY_MODE_COUNT,
}

SD_CACHE_MODE_MAP = {
    "disabled": SDCacheMode.SD_CACHE_DISABLED,  # Default
    "easycache": SDCacheMode.SD_CACHE_EASYCACHE,
    "ucache": SDCacheMode.SD_CACHE_UCACHE,
    "dbcache": SDCacheMode.SD_CACHE_DBCACHE,
    "taylorseer": SDCacheMode.SD_CACHE_TAYLORSEER,
    "cachedit": SDCacheMode.SD_CACHE_CACHE_DIT,
}
