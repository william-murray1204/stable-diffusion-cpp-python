from PIL import PngImagePlugin
from conftest import OUTPUT_DIR

from stable_diffusion_cpp import StableDiffusion

DIFFUSION_MODEL_PATH = "F:\\stable-diffusion\\flex\\Flex.2-preview-Q8_0.gguf"
T5XXL_PATH = "F:\\stable-diffusion\\flux\\t5xxl_q8_0.gguf"
CLIP_L_PATH = "F:\\stable-diffusion\\flux\\clip_l-q8_0.gguf"
VAE_PATH = "F:\\stable-diffusion\\flux\\ae-f16.gguf"

INPUT_IMAGE_PATH = "assets\\input.png"
PROMPT = "the cat has a hat"
STEPS = 20


def test_flex2():

    stable_diffusion = StableDiffusion(
        diffusion_model_path=DIFFUSION_MODEL_PATH,
        clip_l_path=CLIP_L_PATH,
        t5xxl_path=T5XXL_PATH,
        vae_path=VAE_PATH,
        keep_clip_on_cpu=True,
        vae_decode_only=True,
    )

    def progress_callback(step: int, steps: int, time: float):
        print("Completed step: {} of {}".format(step, steps))

    # Generate image
    image = stable_diffusion.generate_image(
        prompt=PROMPT,
        control_image=INPUT_IMAGE_PATH,
        sample_steps=STEPS,
        progress_callback=progress_callback,
    )[0]

    # Save image
    pnginfo = PngImagePlugin.PngInfo()
    pnginfo.add_text("Parameters", ", ".join([f"{k.replace('_', ' ').title()}: {v}" for k, v in image.info.items()]))
    image.save(f"{OUTPUT_DIR}/flex2.png", pnginfo=pnginfo)


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
#     "--control-image",
#     INPUT_IMAGE_PATH,
#     "--vae",
#     VAE_PATH,
#     "--t5xxl",
#     T5XXL_PATH,
#     "--clip_l",
#     CLIP_L_PATH,
#     "--prompt",
#     PROMPT,
#     "--steps",
#     str(STEPS),
#     "--clip-on-cpu",
#     "--output",
#     f"{OUTPUT_DIR}/flex2_cli.png",
#     "-v",
# ]
# print(" ".join(cli_cmd))
# subprocess.run(cli_cmd, check=True)
