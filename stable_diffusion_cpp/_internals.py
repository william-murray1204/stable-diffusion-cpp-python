import os
import ctypes
from contextlib import ExitStack

import stable_diffusion_cpp.stable_diffusion_cpp as sd_cpp
from ._utils import suppress_stdout_stderr

# ============================================
# Stable Diffusion Model
# ============================================


class _StableDiffusionModel:
    """Intermediate Python wrapper for a stable-diffusion.cpp stable_diffusion_model."""

    _free_sd_ctx = None
    # NOTE: this must be "saved" here to avoid exceptions when calling __del__

    def __init__(
        self,
        model_path: str,
        clip_l_path: str,
        clip_g_path: str,
        t5xxl_path: str,
        diffusion_model_path: str,
        vae_path: str,
        taesd_path: str,
        control_net_path: str,
        lora_model_dir: str,
        embedding_dir: str,
        stacked_id_embed_dir: str,
        vae_decode_only: bool,
        vae_tiling: bool,
        n_threads: int,
        wtype: int,
        rng_type: int,
        schedule: int,
        keep_clip_on_cpu: bool,
        keep_control_net_on_cpu: bool,
        keep_vae_on_cpu: bool,
        diffusion_flash_attn: bool,
        diffusion_conv_direct: bool,
        vae_conv_direct: bool,
        chroma_use_dit_mask: bool,
        chroma_use_t5_mask: bool,
        chroma_t5_mask_pad: int,
        verbose: bool,
    ):
        self._exit_stack = ExitStack()
        self.model = None
        self.params = sd_cpp.sd_ctx_params_t(
            model_path=model_path.encode("utf-8"),
            clip_l_path=clip_l_path.encode("utf-8"),
            clip_g_path=clip_g_path.encode("utf-8"),
            t5xxl_path=t5xxl_path.encode("utf-8"),
            diffusion_model_path=diffusion_model_path.encode("utf-8"),
            vae_path=vae_path.encode("utf-8"),
            taesd_path=taesd_path.encode("utf-8"),
            control_net_path=control_net_path.encode("utf-8"),
            lora_model_dir=lora_model_dir.encode("utf-8"),
            embedding_dir=embedding_dir.encode("utf-8"),
            stacked_id_embed_dir=stacked_id_embed_dir.encode("utf-8"),
            vae_decode_only=vae_decode_only,
            vae_tiling=vae_tiling,
            free_params_immediately=False,  # Don't unload model
            n_threads=n_threads,
            wtype=wtype,
            rng_type=rng_type,
            schedule=schedule,
            keep_clip_on_cpu=keep_clip_on_cpu,
            keep_control_net_on_cpu=keep_control_net_on_cpu,
            keep_vae_on_cpu=keep_vae_on_cpu,
            diffusion_flash_attn=diffusion_flash_attn,
            diffusion_conv_direct=diffusion_conv_direct,
            vae_conv_direct=vae_conv_direct,
            chroma_use_dit_mask=chroma_use_dit_mask,
            chroma_use_t5_mask=chroma_use_t5_mask,
            chroma_t5_mask_pad=chroma_t5_mask_pad,
        )

        # Load the free_sd_ctx function
        self._free_sd_ctx = sd_cpp._lib.free_sd_ctx

        # Load the model from the file if the path is provided
        if model_path:
            if not os.path.exists(model_path):
                raise ValueError(f"Model path does not exist: {model_path}")

        if diffusion_model_path:
            if not os.path.exists(diffusion_model_path):
                raise ValueError(f"Diffusion model path does not exist: {diffusion_model_path}")

        if model_path or diffusion_model_path:
            with suppress_stdout_stderr(disable=verbose):
                # Call function with a pointer to params
                self.model = sd_cpp.new_sd_ctx(ctypes.byref(self.params))

            # Check if the model was loaded successfully
            if self.model is None:
                raise ValueError(f"Failed to load model from file: {model_path}")

        def free_ctx():
            """Free the model from memory."""
            if self.model is not None and self._free_sd_ctx is not None:
                self._free_sd_ctx(self.model)
                self.model = None

        self._exit_stack.callback(free_ctx)

    def close(self):
        """Closes the exit stack, ensuring all context managers are exited."""
        self._exit_stack.close()

    def __del__(self):
        """Free memory when the object is deleted."""
        self.close()


# ============================================
# Upscaler Model
# ============================================


class _UpscalerModel:
    """Intermediate Python wrapper for an Esrgan image upscaling model."""

    _free_upscaler_ctx = None
    # NOTE: this must be "saved" here to avoid exceptions when calling __del__

    def __init__(
        self,
        upscaler_path: str,
        n_threads: int,
        diffusion_conv_direct: bool,
        verbose: bool,
    ):
        self.upscaler_path = upscaler_path
        self.n_threads = n_threads
        self.diffusion_conv_direct = diffusion_conv_direct
        self.verbose = verbose
        self._exit_stack = ExitStack()

        self.upscaler = None

        # Load the model from the file if the path is provided
        if upscaler_path:

            # Load the free_upscaler_ctx function
            self._free_upscaler_ctx = sd_cpp._lib.free_upscaler_ctx

            if not os.path.exists(upscaler_path):
                raise ValueError(f"Upscaler model path does not exist: {upscaler_path}")

            # Load the image upscaling model ctx
            self.upscaler = sd_cpp.new_upscaler_ctx(
                upscaler_path.encode("utf-8"),
                self.n_threads,
                self.diffusion_conv_direct,
            )

            # Check if the model was loaded successfully
            if self.upscaler is None:
                raise ValueError(f"Failed to load upscaler model from file: {upscaler_path}")

        def free_ctx():
            """Free the model from memory."""
            if self.upscaler is not None and self._free_upscaler_ctx is not None:
                self._free_upscaler_ctx(self.upscaler)
                self.upscaler = None

        self._exit_stack.callback(free_ctx)

    def close(self):
        """Closes the exit stack, ensuring all context managers are exited."""
        self._exit_stack.close()

    def __del__(self):
        """Free memory when the object is deleted."""
        self.close()
