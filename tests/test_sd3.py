import os
import traceback
from stable_diffusion_cpp import StableDiffusion

MODEL_PATH = "C:\\stable-diffusion\\sd3.5\\sd3.5_large-q4_k_5_0.gguf"
CLIP_L_PATH = "C:\\stable-diffusion\\sd3.5\\clip_l.safetensors"
CLIP_G_PATH = "C:\\stable-diffusion\\sd3.5\\clip_g.safetensors"
T5XXL_PATH = "C:\\stable-diffusion\\sd3.5\\t5xxl_fp16.safetensors"

stable_diffusion = StableDiffusion(
    model_path=MODEL_PATH,
    clip_l_path=CLIP_L_PATH,
    clip_g_path=CLIP_G_PATH,
    t5xxl_path=T5XXL_PATH,
)


def callback(step: int, steps: int, time: float):
    print("Completed step: {} of {}".format(step, steps))


try:
    # Generate images
    images = stable_diffusion.txt_to_img(
        prompt="a lovely cat holding a sign says 'Stable diffusion 3.5 Large'",
        height=832,
        width=832,
        cfg_scale=4.5,
        sample_method="euler",
    )

    OUTPUT_DIR = "tests/outputs"
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)

    # Save images
    for i, image in enumerate(images):
        image.save(f"{OUTPUT_DIR}/sd3_{i}.png")

except Exception as e:
    traceback.print_exc()
    print("Test - sd3 failed: ", e)
