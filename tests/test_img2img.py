from stable_diffusion_cpp import StableDiffusion

MODEL_PATH = "C:\\stable-diffusion\\turbovisionxlSuperFastXLBasedOnNew_tvxlV431Bakedvae.q8_0.gguf"

input_image = "assets\\input.png"

stable_diffusion = StableDiffusion(model_path=MODEL_PATH)

images = stable_diffusion.img_to_img(prompt="a scary cat", image=input_image)

for i, image in enumerate(images):
    image.save(f"output_img2img_{i}.png")


