import os
import traceback

from stable_diffusion_cpp import StableDiffusion

MODEL_PATH = "C:\\stable-diffusion\\sd3.5\\sd3.5_large-q4_k_5_0.gguf"
CLIP_L_PATH = "C:\\stable-diffusion\\sd3.5\\clip_l.safetensors"
CLIP_G_PATH = "C:\\stable-diffusion\\sd3.5\\clip_g.safetensors"
T5XXL_PATH = "C:\\stable-diffusion\\sd3.5\\t5xxl_fp16.safetensors"

PROMPT = "a lovely cat holding a sign says 'Stable diffusion 3.5 Large'"
HEIGHT = 832
WIDTH = 832
CFG_SCALE = 4.5
SAMPLE_METHOD = "euler"


OUTPUT_DIR = "tests/outputs"
if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

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
    images = stable_diffusion.generate_image(
        prompt=PROMPT,
        height=HEIGHT,
        width=WIDTH,
        cfg_scale=CFG_SCALE,
        sample_method=SAMPLE_METHOD,
    )

    # Save images
    for i, image in enumerate(images):
        image.save(f"{OUTPUT_DIR}/sd3_{i}.png")

except Exception as e:
    traceback.print_exc()
    print("Test - sd3 failed: ", e)


# # ======== C++ CLI ========

# import subprocess

# stable_diffusion = None  # Clear model

# SD_CPP_CLI = "C:\\Users\\Willi\\Documents\\GitHub\\stable-diffusion.cpp\\build\\bin\\sd"


# cli_cmd = [
#     SD_CPP_CLI,
#     "--model",
#     MODEL_PATH,
#     "--t5xxl",
#     T5XXL_PATH,
#     "--clip_l",
#     CLIP_L_PATH,
#     "--clip_g",
#     CLIP_G_PATH,
#     "--prompt",
#     PROMPT,
#     "--height",
#     str(HEIGHT),
#     "--width",
#     str(WIDTH),
#     "--cfg-scale",
#     str(CFG_SCALE),
#     "--sample-method",
#     SAMPLE_METHOD,
#     "--output",
#     f"{OUTPUT_DIR}/sd3_cli.png",
#     "-v",
# ]
# print(" ".join(cli_cmd))
# subprocess.run(cli_cmd, check=True)
