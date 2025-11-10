from PIL import PngImagePlugin
from conftest import OUTPUT_DIR

from stable_diffusion_cpp import StableDiffusion

DIFFUSION_MODEL_PATH = "F:\\stable-diffusion\\flux\\flux1-schnell-q3_k.gguf"
# DIFFUSION_MODEL_PATH = "F:\\stable-diffusion\\flux\\flux1-dev-q8_0.gguf"
T5XXL_PATH = "F:\\stable-diffusion\\flux\\t5xxl_q8_0.gguf"
CLIP_L_PATH = "F:\\stable-diffusion\\flux\\clip_l-q8_0.gguf"
VAE_PATH = "F:\\stable-diffusion\\flux\\ae-f16.gguf"
LORA_DIR = "F:\\stable-diffusion\\loras"


PROMPTS = [
    # {
    #     "add": "_lora",
    #     "prompt": "a lovely cat holding a sign says 'flux.cpp' <lora:realism_lora_comfy_converted:1>",
    # },  # With LORA
    {"add": "", "prompt": "a lovely cat holding a sign says 'flux.cpp'"},  # Without LORA
]
STEPS = 4
CFG_SCALE = 1.0


def test_flux():

    stable_diffusion = StableDiffusion(
        diffusion_model_path=DIFFUSION_MODEL_PATH,
        clip_l_path=CLIP_L_PATH,
        t5xxl_path=T5XXL_PATH,
        vae_path=VAE_PATH,
        lora_model_dir=LORA_DIR,
        keep_clip_on_cpu=True,
        vae_decode_only=True,
    )

    def progress_callback(step: int, steps: int, time: float):
        print("Completed step: {} of {}".format(step, steps))

    for prompt in PROMPTS:
        # Generate image
        image = stable_diffusion.generate_image(
            prompt=prompt["prompt"],
            sample_steps=STEPS,
            cfg_scale=CFG_SCALE,
            progress_callback=progress_callback,
        )[0]

        # Save image
        pnginfo = PngImagePlugin.PngInfo()
        pnginfo.add_text("Parameters", ", ".join([f"{k.replace('_', ' ').title()}: {v}" for k, v in image.info.items()]))
        image.save(f"{OUTPUT_DIR}/flux{prompt['add']}.png", pnginfo=pnginfo)


# ===========================================
# C++ CLI
# ===========================================

# import subprocess

# from conftest import SD_CPP_CLI

# stable_diffusion = None  # Clear model

# for prompt in PROMPTS:
#     cli_cmd = [
#         SD_CPP_CLI,
#         "--diffusion-model",
#         DIFFUSION_MODEL_PATH,
#         "--vae",
#         VAE_PATH,
#         "--t5xxl",
#         T5XXL_PATH,
#         "--clip_l",
#         CLIP_L_PATH,
#         "--prompt",
#         prompt["prompt"],
#         "--steps",
#         str(STEPS),
#         "--cfg-scale",
#         str(CFG_SCALE),
#         "--clip-on-cpu",
#         "--output",
#         f"{OUTPUT_DIR}/flux{prompt['add']}_cli.png",
#         "-v",
#     ]
#     print(" ".join(cli_cmd))
#     subprocess.run(cli_cmd, check=True)
