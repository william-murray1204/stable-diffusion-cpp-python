from PIL import PngImagePlugin
from conftest import OUTPUT_DIR

from stable_diffusion_cpp import StableDiffusion

MODEL_PATH = "C:\\stable-diffusion\\turbovisionxlSuperFastXLBasedOnNew_tvxlV431Bakedvae.safetensors"


INPUT_IMAGE_PATH = "assets\\input.png"
MASK_IMAGE_PATH = "assets\\mask.png"

PROMPT = "blue eyes"
STRENGTH = 0.4


def test_inpainting():

    stable_diffusion = StableDiffusion(model_path=MODEL_PATH)

    def callback(step: int, steps: int, time: float):
        print("Completed step: {} of {}".format(step, steps))

    # Generate images
    images = stable_diffusion.generate_image(
        prompt=PROMPT,
        init_image=INPUT_IMAGE_PATH,
        mask_image=MASK_IMAGE_PATH,
        strength=STRENGTH,
        progress_callback=callback,
    )

    # Save images
    for i, image in enumerate(images):
        pnginfo = PngImagePlugin.PngInfo()
        pnginfo.add_text("Parameters", ", ".join([f"{k.replace('_', ' ').title()}: {v}" for k, v in image.info.items()]))
        image.save(f"{OUTPUT_DIR}/inpainting_{i}.png", pnginfo=pnginfo)


# ===========================================
# C++ CLI
# ===========================================

# import subprocess

# stable_diffusion = None  # Clear model

# SD_CPP_CLI = "C:\\Users\\Willi\\Documents\\GitHub\\stable-diffusion.cpp\\build\\bin\\sd"


# cli_cmd = [
#     SD_CPP_CLI,
#     "--model",
#     MODEL_PATH,
#     "--prompt",
#     PROMPT,
#     "--init-img",
#     INPUT_IMAGE_PATH,
#     "--strength",
#     str(STRENGTH),
#     "--mask",
#     MASK_IMAGE_PATH,
#     "--output",
#     f"{OUTPUT_DIR}/inpainting_cli.png",
#     "-v",
# ]
# print(" ".join(cli_cmd))
# subprocess.run(cli_cmd, check=True)
