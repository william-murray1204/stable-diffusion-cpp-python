import os
import traceback

from stable_diffusion_cpp import StableDiffusion

MODEL_PATH = "C:\\stable-diffusion\\svd_xt.safetensors"

input_image = "assets\\input.png"

stable_diffusion = StableDiffusion(model_path=MODEL_PATH)


def callback(step: int, steps: int, time: float):
    print("Completed step: {} of {}".format(step, steps))


try:
    images = stable_diffusion.generate_video(init_image=input_image, progress_callback=callback)

    OUTPUT_DIR = "tests/outputs"
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)

    for i, image in enumerate(images):
        image.save(f"{OUTPUT_DIR}/img2vid_{i}.png")

except Exception as e:
    traceback.print_exc()
    print("Test - img2vid failed: ", e)


# !!!!!!!!!!!!!!!!!! NOT WORKING (waiting for support in https://github.com/leejet/stable-diffusion.cpp ) !!!!!!!!!!!!!!!!!!
