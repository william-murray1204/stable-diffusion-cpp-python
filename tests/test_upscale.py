import os
import traceback
from stable_diffusion_cpp import StableDiffusion

UPSCALER_MODEL_PATH = "C:\\stable-diffusion\\RealESRGAN_x4plus.pth"

input_images = ["assets\\input.png"]

stable_diffusion = StableDiffusion(upscaler_path=UPSCALER_MODEL_PATH)


def callback(step: int, steps: int, time: float):
    print("Completed step: {} of {}".format(step, steps))


try:
    # Upscale images
    images = stable_diffusion.upscale(images=input_images, upscale_factor=2, progress_callback=callback)

    OUTPUT_DIR = "tests/outputs"
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)

    # Save images
    for i, image in enumerate(images):
        image.save(f"{OUTPUT_DIR}/upscale_{i}.png")

except Exception as e:
    traceback.print_exc()
    print("Test - upscale failed: ", e)
