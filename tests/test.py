from stable_diffusion_cpp import StableDiffusion

MODEL_PATH = "C:\\stable-diffusion\\turbovisionxlSuperFastXLBasedOnNew_tvxlV431Bakedvae.q8_0.gguf"

stable_diffusion = StableDiffusion(model_path=MODEL_PATH)

images = stable_diffusion.txt_to_img(prompt="a lovely cat")

for image in images:
    image.save("output.png")
