from PIL import PngImagePlugin
from conftest import OUTPUT_DIR

from stable_diffusion_cpp import StableDiffusion

DIFFUSION_MODEL_PATH = "C:\\stable-diffusion\\flux-chroma\\Chroma1-HD-Flash-Q4_0.gguf"
T5XXL_PATH = "C:\\stable-diffusion\\flux\\t5xxl_q8_0.gguf"
VAE_PATH = "C:\\stable-diffusion\\flux\\ae-f16.gguf"


PROMPT = "a lovely cat holding a sign says 'chroma.cpp'"
STEPS = 4
CFG_SCALE = 4.0
SAMPLE_METHOD = "euler"


def test_chroma():

    stable_diffusion = StableDiffusion(
        diffusion_model_path=DIFFUSION_MODEL_PATH,
        t5xxl_path=T5XXL_PATH,
        vae_path=VAE_PATH,
        keep_clip_on_cpu=True,
        vae_decode_only=True,
        chroma_use_dit_mask=False,
    )

    def callback(step: int, steps: int, time: float):
        print("Completed step: {} of {}".format(step, steps))

    # Generate images
    images = stable_diffusion.generate_image(
        prompt=PROMPT,
        sample_steps=STEPS,
        cfg_scale=CFG_SCALE,
        sample_method=SAMPLE_METHOD,
        progress_callback=callback,
    )

    # Save images
    for i, image in enumerate(images):
        pnginfo = PngImagePlugin.PngInfo()
        pnginfo.add_text("Parameters", ", ".join([f"{k.replace('_', ' ').title()}: {v}" for k, v in image.info.items()]))
        image.save(f"{OUTPUT_DIR}/chroma_{i}.png", pnginfo=pnginfo)


# ===========================================
# C++ CLI
# ===========================================

# import subprocess

# stable_diffusion = None  # Clear model

# SD_CPP_CLI = "C:\\Users\\Willi\\Documents\\GitHub\\stable-diffusion.cpp\\build\\bin\\sd"

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
#     "--sampling-method",
#     SAMPLE_METHOD,
#     "--clip-on-cpu",
#     "--chroma-disable-dit-mask",
#     "--output",
#     f"{OUTPUT_DIR}/chroma_cli.png",
#     "-v",
# ]
# print(" ".join(cli_cmd))
# subprocess.run(cli_cmd, check=True)
