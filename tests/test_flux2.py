from PIL import PngImagePlugin
from conftest import OUTPUT_DIR

from stable_diffusion_cpp import StableDiffusion

DIFFUSION_MODEL_PATH = "F:\\stable-diffusion\\flux2\\flux2-dev-Q2_K.gguf"
LLM_PATH = "F:\\stable-diffusion\\flux2\\Mistral-Small-3.2-24B-Instruct-2506-Q4_K_M.gguf"
VAE_PATH = "F:\\stable-diffusion\\flux2\\ae.safetensors"

INPUT_IMAGE_PATHS = [
    "assets\\input.png",
]

PROMPT = "the cat has a hat"
STEPS = 4
CFG_SCALE = 1.0


def test_flux2():

    stable_diffusion = StableDiffusion(
        diffusion_model_path=DIFFUSION_MODEL_PATH,
        llm_path=LLM_PATH,
        vae_path=VAE_PATH,
        # vae_decode_only=True,
        offload_params_to_cpu=True,
        # diffusion_flash_attn=True,
    )

    def progress_callback(step: int, steps: int, time: float):
        print("Completed step: {} of {}".format(step, steps))

    # Generate image
    image = stable_diffusion.generate_image(
        prompt=PROMPT,
        ref_images=INPUT_IMAGE_PATHS,
        sample_steps=STEPS,
        cfg_scale=CFG_SCALE,
        progress_callback=progress_callback,
    )[0]

    # Save image
    pnginfo = PngImagePlugin.PngInfo()
    pnginfo.add_text("Parameters", ", ".join([f"{k.replace('_', ' ').title()}: {v}" for k, v in image.info.items()]))
    image.save(f"{OUTPUT_DIR}/flux2.png", pnginfo=pnginfo)


# ===========================================
# C++ CLI
# ===========================================

# import subprocess

# from conftest import SD_CPP_CLI

# stable_diffusion = None  # Clear model

# cli_cmd = [
#     SD_CPP_CLI,
#     "--diffusion-model",
#     DIFFUSION_MODEL_PATH,
#     "--llm",
#     LLM_PATH,
#     "--vae",
#     VAE_PATH,
#     "--prompt",
#     PROMPT,
#     "--ref-image",
#     ",".join(INPUT_IMAGE_PATHS),
#     "--steps",
#     str(STEPS),
#     "--cfg-scale",
#     str(CFG_SCALE),
#     "--offload-to-cpu",
#     "--output",
#     f"{OUTPUT_DIR}/flux2_cli.png",
#     "-v",
# ]
# print(" ".join(cli_cmd))
# subprocess.run(cli_cmd, check=True)
