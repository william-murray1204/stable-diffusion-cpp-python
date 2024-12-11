import os
import traceback

from stable_diffusion_cpp import StableDiffusion

INPUT_IMAGE_PATH = "assets\\input.png"

stable_diffusion = StableDiffusion()

try:
    # Apply canny edge detection
    image = stable_diffusion.preprocess_canny(image=INPUT_IMAGE_PATH)

    OUTPUT_DIR = "tests/outputs"
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)

    # Save image
    image.save(f"{OUTPUT_DIR}/preprocess_canny.png")

except Exception as e:
    traceback.print_exc()
    print("Test - preprocess_canny failed: ", e)
