import os
import traceback

from stable_diffusion_cpp import StableDiffusion

MODEL_PATH = "C:\\stable-diffusion\\photomaker\\sdxlUnstableDiffusers_v11.safetensors"
STACKED_ID_EMBED_DIR = "C:\\stable-diffusion\\photomaker\\photomaker-v1.safetensors"
VAE_PATH = "C:\\stable-diffusion\\photomaker\\sdxl.vae.safetensors"

INPUT_ID_IMAGES_PATH = ".\\assets\\newton_man"

PROMPT = "a man img, retro futurism, retro game art style but extremely beautiful, intricate details, masterpiece, best quality, space-themed, cosmic, celestial, stars, galaxies, nebulas, planets, science fiction, highly detailed"
NEGATIVE_PROMPT = "realistic, photo-realistic, worst quality, greyscale, bad anatomy, bad hands, error, text"
STYLE_STRENGTH = 10  # style_ratio (%)
HEIGHT = 1024
WIDTH = 1024
CFG_SCALE = 5.0
SAMPLE_METHOD = "euler"

OUTPUT_DIR = "tests/outputs"
if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

stable_diffusion = StableDiffusion(stacked_id_embed_dir=STACKED_ID_EMBED_DIR, model_path=MODEL_PATH, vae_path=VAE_PATH)


def callback(step: int, steps: int, time: float):
    print("Completed step: {} of {}".format(step, steps))


try:
    # Generate images
    photomaker_images = stable_diffusion.generate_image(
        cfg_scale=CFG_SCALE,
        height=HEIGHT,
        width=WIDTH,
        style_strength=STYLE_STRENGTH,
        sample_method=SAMPLE_METHOD,
        prompt=PROMPT,
        negative_prompt=NEGATIVE_PROMPT,
        input_id_images_path=INPUT_ID_IMAGES_PATH,
        progress_callback=callback,
    )

    for i, image in enumerate(photomaker_images):
        image.save(f"{OUTPUT_DIR}/photomaker_{i}.png")

except Exception as e:
    traceback.print_exc()
    print("Test - photomaker failed: ", e)


# # ======== C++ CLI ========

# import subprocess

# stable_diffusion = None  # Clear model

# SD_CPP_CLI = "C:\\Users\\Willi\\Documents\\GitHub\\stable-diffusion.cpp\\build\\bin\\sd"


# cli_cmd = [
#     SD_CPP_CLI,
#     "--stacked-id-embd-dir",
#     STACKED_ID_EMBED_DIR,
#     "--model",
#     MODEL_PATH,
#     "--vae",
#     VAE_PATH,
#     "--prompt",
#     PROMPT,
#     "--negative-prompt",
#     NEGATIVE_PROMPT,
#     "--style-ratio",
#     str(STYLE_STRENGTH),
#     "--height",
#     str(HEIGHT),
#     "--width",
#     str(WIDTH),
#     "--cfg-scale",
#     str(CFG_SCALE),
#     "--sampling-method",
#     SAMPLE_METHOD,
#     "--input-id-images-dir",
#     INPUT_ID_IMAGES_PATH,
#     "--output",
#     f"{OUTPUT_DIR}/photomaker_cli.png",
#     "-v",
# ]
# print(" ".join(cli_cmd))
# subprocess.run(cli_cmd, check=True)
