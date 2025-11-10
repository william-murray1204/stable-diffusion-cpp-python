from conftest import OUTPUT_DIR

from stable_diffusion_cpp import StableDiffusion

UPSCALER_MODEL_PATH = "F:\\stable-diffusion\\RealESRGAN_x4plus.pth"


INPUT_IMAGES = ["assets\\input.png"]

UPSCALE_FACTOR = 2


def test_upscale():

    stable_diffusion = StableDiffusion(upscaler_path=UPSCALER_MODEL_PATH)

    def progress_callback(step: int, steps: int, time: float):
        print("Completed step: {} of {}".format(step, steps))

    # Upscale images
    images = stable_diffusion.upscale(
        images=INPUT_IMAGES,
        upscale_factor=UPSCALE_FACTOR,
        progress_callback=progress_callback,
    )

    # Save images
    for i, image in enumerate(images):
        image.save(f"{OUTPUT_DIR}/upscale_{i}.png")
