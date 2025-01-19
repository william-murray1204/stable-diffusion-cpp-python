import os
import traceback

from stable_diffusion_cpp import StableDiffusion

MODEL_PATH = "C:\\stable-diffusion\\v1-5-pruned-emaonly.safetensors"

stable_diffusion = StableDiffusion(model_path=MODEL_PATH, wtype="f16", vae_tiling=True)


def callback(step: int, steps: int, time: float):
    print("Completed step: {} of {}".format(step, steps))


try:
    OUTPUT_DIR = "tests/outputs"
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)

    # Generate images
    images = stable_diffusion.txt_to_img(
        prompt="a fantasy character, detailed background, colorful",
        sample_steps=20,
        seed=42,
        width=512,
        height=512,
        sample_method="euler",
        progress_callback=callback,
    )

    # Save images
    for i, image in enumerate(images):
        image.save(f"{OUTPUT_DIR}/vulkan_{i}.png")

except Exception as e:
    traceback.print_exc()
    print("Test - txt2img failed: ", e)
