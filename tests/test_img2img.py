import os
import traceback

from stable_diffusion_cpp import StableDiffusion

MODEL_PATH = "C:\\stable-diffusion\\turbovisionxlSuperFastXLBasedOnNew_tvxlV431Bakedvae.safetensors"

INPUT_IMAGE_PATH = "assets\\input.png"

stable_diffusion = StableDiffusion(model_path=MODEL_PATH)


def callback(step: int, steps: int, time: float):
    print("Completed step: {} of {}".format(step, steps))


try:
    # Generate images
    images = stable_diffusion.img_to_img(
        prompt="blue eyes",
        image=INPUT_IMAGE_PATH,
        strength=0.4,
        progress_callback=callback,
    )

    OUTPUT_DIR = "tests/outputs"
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)

    # Save images
    for i, image in enumerate(images):
        image.save(f"{OUTPUT_DIR}/img2img_{i}.png")

except Exception as e:
    traceback.print_exc()
    print("Test - img2img failed: ", e)
