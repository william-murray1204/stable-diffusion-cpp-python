from conftest import OUTPUT_DIR

from stable_diffusion_cpp import StableDiffusion

INPUT_IMAGE_PATH = "assets\\input.png"


def test_preprocess_canny():

    stable_diffusion = StableDiffusion()

    # Apply canny edge detection
    image = stable_diffusion.preprocess_canny(image=INPUT_IMAGE_PATH)

    # Save image
    image.save(f"{OUTPUT_DIR}/preprocess_canny.png")
