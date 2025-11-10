from PIL import PngImagePlugin
from conftest import OUTPUT_DIR

from stable_diffusion_cpp import StableDiffusion

DIFFUSION_MODEL_PATH = "F:\\stable-diffusion\\flux-kontext\\flux1-kontext-dev-Q3_K_S.gguf"
T5XXL_PATH = "F:\\stable-diffusion\\flux\\t5xxl_q8_0.gguf"
CLIP_L_PATH = "F:\\stable-diffusion\\flux\\clip_l-q8_0.gguf"
VAE_PATH = "F:\\stable-diffusion\\flux\\ae-f16.gguf"


INPUT_IMAGE_PATHS = [
    "assets\\input.png",
    # "assets\\box.png",
]

PROMPT = "make the cat blue"
STEPS = 4
CFG_SCALE = 1.0
IMAGE_CFG_SCALE = 1.0
SAMPLE_METHOD = "euler"


def test_edit():

    stable_diffusion = StableDiffusion(
        diffusion_model_path=DIFFUSION_MODEL_PATH,
        clip_l_path=CLIP_L_PATH,
        t5xxl_path=T5XXL_PATH,
        vae_path=VAE_PATH,
        keep_clip_on_cpu=True,
    )

    def progress_callback(step: int, steps: int, time: float):
        print("Completed step: {} of {}".format(step, steps))

    # Edit image
    image = stable_diffusion.generate_image(
        prompt=PROMPT,
        ref_images=INPUT_IMAGE_PATHS,
        sample_steps=STEPS,
        cfg_scale=CFG_SCALE,
        image_cfg_scale=IMAGE_CFG_SCALE,
        sample_method=SAMPLE_METHOD,
        progress_callback=progress_callback,
    )[0]

    # Save image
    pnginfo = PngImagePlugin.PngInfo()
    pnginfo.add_text("Parameters", ", ".join([f"{k.replace('_', ' ').title()}: {v}" for k, v in image.info.items()]))
    image.save(f"{OUTPUT_DIR}/edit.png", pnginfo=pnginfo)


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
#     "--t5xxl",
#     T5XXL_PATH,
#     "--clip_l",
#     CLIP_L_PATH,
#     "--prompt",
#     PROMPT,
#     "--ref-image",
#     ",".join(INPUT_IMAGE_PATHS),
#     "--steps",
#     str(STEPS),
#     "--cfg-scale",
#     str(CFG_SCALE),
#     "--img-cfg-scale",
#     str(IMAGE_CFG_SCALE),
#     "--sampling-method",
#     SAMPLE_METHOD,
#     "--clip-on-cpu",
#     "--output",
#     f"{OUTPUT_DIR}/edit_cli.png",
#     "-v",
# ]
# print(" ".join(cli_cmd))
# subprocess.run(cli_cmd, check=True)
