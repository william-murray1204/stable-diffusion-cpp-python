from stable_diffusion_cpp import StableDiffusion

UPSCALER_MODEL_PATH = "C:\\stable-diffusion\\RealESRGAN_x4plus.pth"

input_image = "assets\\input.png"

stable_diffusion = StableDiffusion(upscaler_path=UPSCALER_MODEL_PATH)

images = stable_diffusion.upscale(images=input_image, upscale_factor=2)

for image in images:
    image.save("output_upscale.png")
