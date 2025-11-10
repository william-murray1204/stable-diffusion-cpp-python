from PIL import PngImagePlugin
from conftest import OUTPUT_DIR

from stable_diffusion_cpp import StableDiffusion

DIFFUSION_MODEL_PATH = "F:\\stable-diffusion\\flux-chroma\\Chroma1-HD-Flash-Q4_0.gguf"
T5XXL_PATH = "F:\\stable-diffusion\\flux\\t5xxl_q8_0.gguf"
VAE_PATH = "F:\\stable-diffusion\\flux\\ae-f16.gguf"


PROMPT = "a lovely cat holding a sign says 'chroma.cpp'"
STEPS = 4
CFG_SCALE = 4.0


def test_chroma():

    stable_diffusion = StableDiffusion(
        diffusion_model_path=DIFFUSION_MODEL_PATH,
        t5xxl_path=T5XXL_PATH,
        vae_path=VAE_PATH,
        keep_clip_on_cpu=True,
        vae_decode_only=True,
        chroma_use_dit_mask=False,
    )

    def progress_callback(step: int, steps: int, time: float):
        print("Completed step: {} of {}".format(step, steps))

    # Generate image
    image = stable_diffusion.generate_image(
        prompt=PROMPT,
        sample_steps=STEPS,
        cfg_scale=CFG_SCALE,
        progress_callback=progress_callback,
    )[0]

    # Save image
    pnginfo = PngImagePlugin.PngInfo()
    pnginfo.add_text("Parameters", ", ".join([f"{k.replace('_', ' ').title()}: {v}" for k, v in image.info.items()]))
    image.save(f"{OUTPUT_DIR}/chroma.png", pnginfo=pnginfo)


# ===========================================
# C++ CLI
# ===========================================

# import subprocess

# from conftest import SD_CPP_CLI

# stable_diffusion = None  # Clear model

# cli_cmd = [
#     SD_CPP_CLI,
#     "--diffusion-model",
#     DIFFUSION_MODEL_PATH,
#     "--vae",
#     VAE_PATH,
#     "--t5xxl",
#     T5XXL_PATH,
#     "--prompt",
#     PROMPT,
#     "--steps",
#     str(STEPS),
#     "--cfg-scale",
#     str(CFG_SCALE),
#     "--clip-on-cpu",
#     "--chroma-disable-dit-mask",
#     "--output",
#     f"{OUTPUT_DIR}/chroma_cli.png",
#     "-v",
# ]
# print(" ".join(cli_cmd))
# subprocess.run(cli_cmd, check=True)
