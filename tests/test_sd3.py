from PIL import PngImagePlugin
from conftest import OUTPUT_DIR

from stable_diffusion_cpp import StableDiffusion

MODEL_PATH = "F:\\stable-diffusion\\sd3.5\\sd3.5_large-q4_k_5_0.gguf"
CLIP_L_PATH = "F:\\stable-diffusion\\sd3.5\\clip_l.safetensors"
CLIP_G_PATH = "F:\\stable-diffusion\\sd3.5\\clip_g.safetensors"
T5XXL_PATH = "F:\\stable-diffusion\\sd3.5\\t5xxl_fp8_e4m3fn.safetensors"


PROMPT = "a lovely cat holding a sign says 'Stable diffusion 3.5 Large'"
HEIGHT = 512
WIDTH = 512
CFG_SCALE = 4.5
SAMPLE_METHOD = "euler"
SAMPLE_STEPS = 4


def test_sd3():

    stable_diffusion = StableDiffusion(
        model_path=MODEL_PATH,
        clip_l_path=CLIP_L_PATH,
        clip_g_path=CLIP_G_PATH,
        t5xxl_path=T5XXL_PATH,
        keep_clip_on_cpu=True,
    )

    def progress_callback(step: int, steps: int, time: float):
        print("Completed step: {} of {}".format(step, steps))

    # Generate image
    image = stable_diffusion.generate_image(
        prompt=PROMPT,
        height=HEIGHT,
        width=WIDTH,
        cfg_scale=CFG_SCALE,
        sample_method=SAMPLE_METHOD,
        sample_steps=SAMPLE_STEPS,
        progress_callback=progress_callback,
    )[0]

    # Save image
    pnginfo = PngImagePlugin.PngInfo()
    pnginfo.add_text("Parameters", ", ".join([f"{k.replace('_', ' ').title()}: {v}" for k, v in image.info.items()]))
    image.save(f"{OUTPUT_DIR}/sd3.png", pnginfo=pnginfo)


# ===========================================
# C++ CLI
# ===========================================

# import subprocess

# from conftest import SD_CPP_CLI

# stable_diffusion = None  # Clear model


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
#     "--sampling-method",
#     SAMPLE_METHOD,
#     "--steps",
#     str(SAMPLE_STEPS),
#     "--clip-on-cpu",
#     "--output",
#     f"{OUTPUT_DIR}/sd3_cli.png",
#     "-v",
# ]
# print(" ".join(cli_cmd))
# subprocess.run(cli_cmd, check=True)
