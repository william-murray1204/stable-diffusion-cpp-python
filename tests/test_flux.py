import os
import traceback
from stable_diffusion_cpp import StableDiffusion

# DIFFUSION_MODEL_PATH = "C:\\stable-diffusion\\flux\\flux1-dev-q8_0.gguf"
# T5XXL_PATH = "C:\\stable-diffusion\\flux\\flux_t5xxl_fp16.safetensors"
# CLIP_L_PATH = "C:\\stable-diffusion\\flux\\flux_clip_l.safetensors"
# VAE_PATH = "C:\\stable-diffusion\\flux\\flux_vae.safetensors"


DIFFUSION_MODEL_PATH = "C:\\stable-diffusion\\flux\\flux1-schnell-q3_k.gguf"
T5XXL_PATH = "C:\\stable-diffusion\\flux\\t5xxl_q8_0.gguf"
CLIP_L_PATH = "C:\\stable-diffusion\\flux\\clip_l-q8_0.gguf"
VAE_PATH = "C:\\stable-diffusion\\flux\\ae-f16.gguf"

stable_diffusion = StableDiffusion(
    diffusion_model_path=DIFFUSION_MODEL_PATH,
    clip_l_path=CLIP_L_PATH,
    t5xxl_path=T5XXL_PATH,
    vae_path=VAE_PATH,
)


def callback(step: int, steps: int, time: float):
    print("Completed step: {} of {}".format(step, steps))


try:
    # Generate images
    images = stable_diffusion.txt_to_img(
        prompt="a lovely cat holding a sign says 'flux.cpp'",
        width=1024,
        height=1024,
        sample_steps=4,
        cfg_scale=1.0,
        sample_method="euler",
        progress_callback=callback,
    )

    OUTPUT_DIR = "tests/outputs"
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)

    # Save images
    for i, image in enumerate(images):
        image.save(f"{OUTPUT_DIR}/flux_{i}.png")

except Exception as e:
    traceback.print_exc()
    print("Test - flux failed: ", e)