from stable_diffusion_cpp import StableDiffusion

MODEL_PATH = "C:\\stable-diffusion\\turbovisionxlSuperFastXLBasedOnNew_tvxlV431Bakedvae.q8_0.gguf"

INPUT_IMAGE_PATH = "assets\\input.png"

stable_diffusion = StableDiffusion(model_path=MODEL_PATH)


def callback(step: int, steps: int, time: float):
    print("Completed step: {} of {}".format(step, steps))


try:
    # Generate images
    images = stable_diffusion.img_to_img(
        prompt="blue eyes",
        image=INPUT_IMAGE_PATH,
        control_cond=INPUT_IMAGE_PATH,
        canny=False,
        strength=0.4,
        progress_callback=callback,
    )
    # Save images
    for i, image in enumerate(images):
        image.save(f"output_img2img_{i}.png")

except Exception as e:
    print("Test - img2img failed: ", e)
