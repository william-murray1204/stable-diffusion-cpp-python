from stable_diffusion_cpp import StableDiffusion

UPSCALER_MODEL_PATH = "C:\\stable-diffusion\\RealESRGAN_x4plus.pth"

input_images = ["assets\\input.png"]

stable_diffusion = StableDiffusion(upscaler_path=UPSCALER_MODEL_PATH)


def callback(step: int, steps: int, time: float):
    print("Completed step: {} of {}".format(step, steps))


images = stable_diffusion.upscale(
    images=input_images, upscale_factor=2, progress_callback=callback
)

for i, image in enumerate(images):
    image.save(f"output_upscale_{i}.png")
