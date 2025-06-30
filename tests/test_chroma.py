import os
import traceback

from stable_diffusion_cpp import StableDiffusion

DIFFUSION_MODEL_PATH = "C:\\stable-diffusion\\flux-chroma\\chroma-unlocked-v40-Q4_0.gguf"
T5XXL_PATH = "C:\\stable-diffusion\\flux\\t5xxl_q8_0.gguf"
VAE_PATH = "C:\\stable-diffusion\\flux\\ae-f16.gguf"

stable_diffusion = StableDiffusion(
    diffusion_model_path=DIFFUSION_MODEL_PATH,
    t5xxl_path=T5XXL_PATH,
    vae_path=VAE_PATH,
    vae_decode_only=True,
)


def callback(step: int, steps: int, time: float):
    print("Completed step: {} of {}".format(step, steps))


try:
    OUTPUT_DIR = "tests/outputs"
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)

    # Generate images
    images = stable_diffusion.txt_to_img(
        prompt="a lovely cat holding a sign says 'chroma.cpp'",
        sample_steps=4,
        cfg_scale=4.0,
        sample_method="euler",
        progress_callback=callback,
    )

    # Save images
    for i, image in enumerate(images):
        image.save(f"{OUTPUT_DIR}/chroma_{i}.png")

except Exception as e:
    traceback.print_exc()
    print("Test - chroma failed: ", e)
