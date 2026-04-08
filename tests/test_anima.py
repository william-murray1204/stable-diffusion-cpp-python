from PIL import PngImagePlugin
from conftest import OUTPUT_DIR

from stable_diffusion_cpp import StableDiffusion

DIFFUSION_MODEL_PATH = "F:\\stable-diffusion\\anima\\anima-preview3-base.safetensors"
VAE_PATH = "F:\\stable-diffusion\\anima\\qwen_image_vae.safetensors"
LLM_PATH = "F:\\stable-diffusion\\anima\\qwen_3_06b_base.safetensors"


PROMPT = "a lovely cat holding a sign says 'anima.cpp'"

CFG_SCALE = 6.0
SAMPLE_METHOD = "euler"


def test_anima():

    stable_diffusion = StableDiffusion(
        diffusion_model_path=DIFFUSION_MODEL_PATH,
        llm_path=LLM_PATH,
        vae_path=VAE_PATH,
        offload_params_to_cpu=True,
    )

    def progress_callback(step: int, steps: int, time: float):
        print("Completed step: {} of {}".format(step, steps))

    # Generate image
    image = stable_diffusion.generate_image(
        prompt=PROMPT,
        cfg_scale=CFG_SCALE,
        sample_method=SAMPLE_METHOD,
        progress_callback=progress_callback,
        preview_method="proj",
    )[0]

    # Save image
    pnginfo = PngImagePlugin.PngInfo()
    pnginfo.add_text("Parameters", ", ".join([f"{k.replace('_', ' ').title()}: {v}" for k, v in image.info.items()]))
    image.save(f"{OUTPUT_DIR}/anima.png", pnginfo=pnginfo)


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
#     "--vae",
#     VAE_PATH,
#     "--llm",
#     LLM_PATH,
#     "--prompt",
#     PROMPT,
#     "--cfg-scale",
#     str(CFG_SCALE),
#     "--sampling-method",
#     SAMPLE_METHOD,
#     "--offload-to-cpu",
#     "--output",
#     f"{OUTPUT_DIR}/anima_cli.png",
#     "-v",
# ]
# print(" ".join(cli_cmd))
# subprocess.run(cli_cmd, check=True)
