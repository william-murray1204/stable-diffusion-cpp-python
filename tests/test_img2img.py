from stable_diffusion_cpp import StableDiffusion

MODEL_PATH = "C:\\stable-diffusion\\turbovisionxlSuperFastXLBasedOnNew_tvxlV431Bakedvae.q8_0.gguf"

input_image = "assets\\input.png"

stable_diffusion = StableDiffusion(model_path=MODEL_PATH)


def callback(step: int, steps: int, time: float):
    print("Completed step: {} of {}".format(step, steps))


images = stable_diffusion.img_to_img(
    prompt="a scary cat", image=input_image, progress_callback=callback
)

for i, image in enumerate(images):
    image.save(f"output_img2img_{i}.png")
