import os

from PIL import PngImagePlugin
from conftest import OUTPUT_DIR

from stable_diffusion_cpp import StableDiffusion

MODEL_PATH = "F:\\stable-diffusion\\photomaker\\sdxlUnstableDiffusers_v11.safetensors"
PHOTO_MAKER_PATH = "F:\\stable-diffusion\\photomaker\\photomaker-v2.safetensors"
VAE_PATH = "F:\\stable-diffusion\\photomaker\\sdxl.vae.safetensors"


INPUT_ID_IMAGES_DIR = ".\\assets\\newton_man"
INPUT_ID_EMBED_PATH = ".\\assets\\newton_man\\id_embeds.bin"


PROMPT = "a man img, retro futurism, retro game art style but extremely beautiful, intricate details, masterpiece, best quality, space-themed, cosmic, celestial, stars, galaxies, nebulas, planets, science fiction, highly detailed"
NEGATIVE_PROMPT = "realistic, photo-realistic, worst quality, greyscale, bad anatomy, bad hands, error, text"
STYLE_STRENGTH = 10
HEIGHT = 1024
WIDTH = 1024
CFG_SCALE = 5.0
SAMPLE_METHOD = "euler"


def test_photomaker_2():

    stable_diffusion = StableDiffusion(
        photo_maker_path=PHOTO_MAKER_PATH,
        model_path=MODEL_PATH,
        vae_path=VAE_PATH,
    )

    def progress_callback(step: int, steps: int, time: float):
        print("Completed step: {} of {}".format(step, steps))

    # Generate image
    image = stable_diffusion.generate_image(
        cfg_scale=CFG_SCALE,
        height=HEIGHT,
        width=WIDTH,
        sample_method=SAMPLE_METHOD,
        prompt=PROMPT,
        negative_prompt=NEGATIVE_PROMPT,
        pm_style_strength=STYLE_STRENGTH,
        pm_id_images=[
            os.path.join(INPUT_ID_IMAGES_DIR, f)
            for f in os.listdir(INPUT_ID_IMAGES_DIR)
            if f.lower().endswith((".png", ".jpg", ".jpeg", ".bmp"))
        ],
        pm_id_embed_path=INPUT_ID_EMBED_PATH,
        progress_callback=progress_callback,
    )[0]

    # Save image
    pnginfo = PngImagePlugin.PngInfo()
    pnginfo.add_text("Parameters", ", ".join([f"{k.replace('_', ' ').title()}: {v}" for k, v in image.info.items()]))
    image.save(f"{OUTPUT_DIR}/photomaker_2.png", pnginfo=pnginfo)


# ===========================================
# C++ CLI
# ===========================================

# import subprocess

# from conftest import SD_CPP_CLI

# stable_diffusion = None  # Clear model


# cli_cmd = [
#     SD_CPP_CLI,
#     "--photo-maker",
#     PHOTO_MAKER_PATH,
#     "--model",
#     MODEL_PATH,
#     "--vae",
#     VAE_PATH,
#     "--prompt",
#     PROMPT,
#     "--negative-prompt",
#     NEGATIVE_PROMPT,
#     "--pm-style-strength",
#     str(STYLE_STRENGTH),
#     "--height",
#     str(HEIGHT),
#     "--width",
#     str(WIDTH),
#     "--cfg-scale",
#     str(CFG_SCALE),
#     "--sampling-method",
#     SAMPLE_METHOD,
#     "--pm-id-images-dir",
#     INPUT_ID_IMAGES_PATH,
#     "--pm-id-embed-path",
#     INPUT_ID_EMBED_PATH,
#     "--output",
#     f"{OUTPUT_DIR}/photomaker_2_cli.png",
#     "-v",
# ]
# print(" ".join(cli_cmd))
# subprocess.run(cli_cmd, check=True)
