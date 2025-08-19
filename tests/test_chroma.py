import os
import traceback

from stable_diffusion_cpp import StableDiffusion

DIFFUSION_MODEL_PATH = "C:\\stable-diffusion\\flux-chroma\\chroma-unlocked-v40-Q4_0.gguf"
T5XXL_PATH = "C:\\stable-diffusion\\flux\\t5xxl_q8_0.gguf"
VAE_PATH = "C:\\stable-diffusion\\flux\\ae-f16.gguf"


PROMPT = "a lovely cat holding a sign says 'chroma.cpp'"
STEPS = 4
CFG_SCALE = 4.0
SAMPLE_METHOD = "euler"

OUTPUT_DIR = "tests/outputs"
if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)


stable_diffusion = StableDiffusion(
    diffusion_model_path=DIFFUSION_MODEL_PATH,
    t5xxl_path=T5XXL_PATH,
    vae_path=VAE_PATH,
    vae_decode_only=True,
)


def callback(step: int, steps: int, time: float):
    print("Completed step: {} of {}".format(step, steps))


try:
    # Generate images
    images = stable_diffusion.generate_image(
        prompt=PROMPT,
        sample_steps=STEPS,
        cfg_scale=CFG_SCALE,
        sample_method=SAMPLE_METHOD,
        progress_callback=callback,
    )

    # Save images
    for i, image in enumerate(images):
        image.save(f"{OUTPUT_DIR}/chroma_{i}.png")

except Exception as e:
    traceback.print_exc()
    print("Test - chroma failed: ", e)


# # ======== C++ CLI ========

# import subprocess

# stable_diffusion = None  # Clear model

# SD_CPP_CLI = "C:\\Users\\Willi\\Documents\\GitHub\\stable-diffusion.cpp\\build\\bin\\sd"

# cli_cmd = [
#     SD_CPP_CLI,
#     "--diffusion-model",
#     DIFFUSION_MODEL_PATH,
#     "--vae",
#     VAE_PATH,
#     "--t5xxl",
#     T5XXL_PATH,
#     "--prompt",
#     PROMPT,
#     "--steps",
#     str(STEPS),
#     "--cfg-scale",
#     str(CFG_SCALE),
#     "--sampling-method",
#     SAMPLE_METHOD,
#     "--output",
#     f"{OUTPUT_DIR}/chroma_cli.png",
#     "-v",
# ]
# print(" ".join(cli_cmd))
# subprocess.run(cli_cmd, check=True)
