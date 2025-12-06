from PIL import PngImagePlugin
from conftest import OUTPUT_DIR

from stable_diffusion_cpp import StableDiffusion

DIFFUSION_MODEL_PATH = "F:\\stable-diffusion\\z-image\\z_image_turbo-Q3_K.gguf"
LLM_PATH = "F:\\stable-diffusion\\z-image\\Qwen3-4B-Instruct-2507-Q4_K_M.gguf"
VAE_PATH = "F:\\stable-diffusion\\z-image\\ae.safetensors"

PROMPT = "A cinematic, melancholic photograph of a solitary hooded figure walking through a sprawling, rain-slicked metropolis at night. The city lights are a chaotic blur of neon orange and cool blue, reflecting on the wet asphalt. The scene evokes a sense of being a single component in a vast machine. Superimposed over the image in a sleek, modern, slightly glitched font is the philosophical quote: 'THE CITY IS A CIRCUIT BOARD, AND I AM A BROKEN TRANSISTOR.' -- moody, atmospheric, profound, dark academic"
STEPS = 20
CFG_SCALE = 1.0
HEIGHT = 1024
WIDTH = 512


def test_z_image():

    stable_diffusion = StableDiffusion(
        diffusion_model_path=DIFFUSION_MODEL_PATH,
        llm_path=LLM_PATH,
        vae_path=VAE_PATH,
        diffusion_flash_attn=True,
    )

    def progress_callback(step: int, steps: int, time: float):
        print("Completed step: {} of {}".format(step, steps))

    # Generate image
    image = stable_diffusion.generate_image(
        prompt=PROMPT,
        height=HEIGHT,
        width=WIDTH,
        sample_steps=STEPS,
        cfg_scale=CFG_SCALE,
        progress_callback=progress_callback,
    )[0]

    # Save image
    pnginfo = PngImagePlugin.PngInfo()
    pnginfo.add_text("Parameters", ", ".join([f"{k.replace('_', ' ').title()}: {v}" for k, v in image.info.items()]))
    image.save(f"{OUTPUT_DIR}/z_image.png", pnginfo=pnginfo)


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
#     "--llm",
#     LLM_PATH,
#     "--vae",
#     VAE_PATH,
#     "--prompt",
#     PROMPT,
#     "--height",
#     str(HEIGHT),
#     "--width",
#     str(WIDTH),
#     "--steps",
#     str(STEPS),
#     "--cfg-scale",
#     str(CFG_SCALE),
#     "--diffusion-fa",
#     "--output",
#     f"{OUTPUT_DIR}/z_image_cli.png",
#     "-v",
# ]
# print(" ".join(cli_cmd))
# subprocess.run(cli_cmd, check=True)
