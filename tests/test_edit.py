import os
import traceback

from stable_diffusion_cpp import StableDiffusion

DIFFUSION_MODEL_PATH = "C:\\stable-diffusion\\flux-kontext\\flux1-kontext-dev-Q3_K_S.gguf"
T5XXL_PATH = "C:\\stable-diffusion\\flux\\t5xxl_q8_0.gguf"
CLIP_L_PATH = "C:\\stable-diffusion\\flux\\clip_l-q8_0.gguf"
VAE_PATH = "C:\\stable-diffusion\\flux\\ae-f16.gguf"

INPUT_IMAGE_PATHS = [
    "assets\\input.png",
    # "assets\\box.png",
]

stable_diffusion = StableDiffusion(
    diffusion_model_path=DIFFUSION_MODEL_PATH,
    clip_l_path=CLIP_L_PATH,
    t5xxl_path=T5XXL_PATH,
    vae_path=VAE_PATH,
)


def callback(step: int, steps: int, time: float):
    print("Completed step: {} of {}".format(step, steps))


try:
    # Edit image
    images = stable_diffusion.edit(
        prompt="make the cat blue",
        images=INPUT_IMAGE_PATHS,
        sample_steps=4,
        cfg_scale=1.0,
        sample_method="euler",
        progress_callback=callback,
    )

    OUTPUT_DIR = "tests/outputs"
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)

    # Save images
    for i, image in enumerate(images):
        image.save(f"{OUTPUT_DIR}/edit_{i}.png")

except Exception as e:
    traceback.print_exc()
    print("Test - edit failed: ", e)
