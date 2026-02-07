from PIL import PngImagePlugin
from conftest import OUTPUT_DIR

from stable_diffusion_cpp import StableDiffusion

DIFFUSION_MODEL_PATH = "F:\\stable-diffusion\\qwen\\Qwen_Image_Edit-Q2_K.gguf"
VAE_PATH = "F:\\stable-diffusion\\qwen\\qwen_image_vae.safetensors"
LLM_PATH = "F:\\stable-diffusion\\qwen\\Qwen2.5-VL-7B-Instruct.Q4_K_M.gguf"


PROMPT = "put a party hat on the cat"
INPUT_IMAGE_PATHS = ["assets\\input.png"]

STEPS = 10
CFG_SCALE = 2.5
SAMPLE_METHOD = "euler"
FLOW_SHIFT = 3


def test_qwen_image_edit():

    stable_diffusion = StableDiffusion(
        diffusion_model_path=DIFFUSION_MODEL_PATH,
        llm_path=LLM_PATH,
        vae_path=VAE_PATH,
        offload_params_to_cpu=True,
        flow_shift=FLOW_SHIFT,
    )

    def progress_callback(step: int, steps: int, time: float):
        print("Completed step: {} of {}".format(step, steps))

    # Generate image
    image = stable_diffusion.generate_image(
        prompt=PROMPT,
        sample_steps=STEPS,
        cfg_scale=CFG_SCALE,
        ref_images=INPUT_IMAGE_PATHS,
        sample_method=SAMPLE_METHOD,
        progress_callback=progress_callback,
    )[0]

    # Save image
    pnginfo = PngImagePlugin.PngInfo()
    pnginfo.add_text("Parameters", ", ".join([f"{k.replace('_', ' ').title()}: {v}" for k, v in image.info.items()]))
    image.save(f"{OUTPUT_DIR}/qwen_image_edit.png", pnginfo=pnginfo)


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
#     "--ref-image",
#     ",".join(INPUT_IMAGE_PATHS),
#     "--prompt",
#     PROMPT,
#     "--steps",
#     str(STEPS),
#     "--cfg-scale",
#     str(CFG_SCALE),
#     "--sampling-method",
#     SAMPLE_METHOD,
#     "--flow-shift",
#     str(FLOW_SHIFT),
#     "--offload-to-cpu",
#     "--output",
#     f"{OUTPUT_DIR}/qwen_image_edit_cli.png",
#     "-v",
# ]
# print(" ".join(cli_cmd))
# subprocess.run(cli_cmd, check=True)
