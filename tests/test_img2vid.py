from stable_diffusion_cpp import StableDiffusion

MODEL_PATH = "C:\\stable-diffusion\\svd_xt.safetensors"

input_image = "assets\\input.png"

stable_diffusion = StableDiffusion(model_path=MODEL_PATH)


def callback(step: int, steps: int, time: float):
    print("Completed step: {} of {}".format(step, steps))


images = stable_diffusion.img_to_vid(image=input_image, progress_callback=callback)

for i, image in enumerate(images):
    image.save(f"output_img2vid_{i}.png")
