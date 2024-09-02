import os
import time
import traceback
from stable_diffusion_cpp import StableDiffusion

MODEL_PATH = "C:\\stable-diffusion\\turbovisionxlSuperFastXLBasedOnNew_tvxlV431Bakedvae.safetensors"
VAE_PATH = "C:\\stable-diffusion\\vaes\\sdxl_vae.safetensors"

stable_diffusion = StableDiffusion(model_path=MODEL_PATH, vae_path=VAE_PATH)


def callback(step: int, steps: int, time: float):
    print("Completed step: {} of {}".format(step, steps))


try:
    # Generate images
    images = stable_diffusion.txt_to_img(
        prompt="a lovely cat",
        sample_steps=2,
        seed=42,
        progress_callback=callback,
    )

    OUTPUT_DIR = "tests/outputs"
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)

    images[0].save(f"{OUTPUT_DIR}/vaetxt2img.png")
    time.sleep(10)

    # Free memory
    print("Free memory")
    stable_diffusion = None

    time.sleep(10)
    print("Finishing")

except Exception as e:
    traceback.print_exc()
    print("Test - txt2img failed: ", e)
