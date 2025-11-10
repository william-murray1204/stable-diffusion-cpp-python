from PIL import PngImagePlugin
from conftest import OUTPUT_DIR

from stable_diffusion_cpp import StableDiffusion

MODEL_PATH = "F:\\stable-diffusion\\v1-5-pruned-emaonly.safetensors"
CONTROLNET_MODEL_PATH = "F:\\stable-diffusion\\control_nets\\control_openpose-fp16.safetensors"


INPUT_IMAGE_PATH = "assets\\input.png"

PROMPTS = [
    {"add": "", "prompt": "a lovely cat", "canny": False},
    {"add": "_canny", "prompt": "a lovely cat", "canny": True},
]


def test_controlnet():
    stable_diffusion = StableDiffusion(
        model_path=MODEL_PATH,
        control_net_path=CONTROLNET_MODEL_PATH,
    )

    def progress_callback(step: int, steps: int, time: float):
        print("Completed step: {} of {}".format(step, steps))

    for prompt in PROMPTS:
        # Generate image
        image = stable_diffusion.generate_image(
            prompt=prompt["prompt"],
            control_image=INPUT_IMAGE_PATH,
            canny=prompt["canny"],
            progress_callback=progress_callback,
        )[0]

        # Save image
        pnginfo = PngImagePlugin.PngInfo()
        pnginfo.add_text("Parameters", ", ".join([f"{k.replace('_', ' ').title()}: {v}" for k, v in image.info.items()]))
        image.save(f"{OUTPUT_DIR}/controlnet{prompt['add']}.png", pnginfo=pnginfo)


# ===========================================
# C++ CLI
# ===========================================

# import subprocess

# from conftest import SD_CPP_CLI

# stable_diffusion = None  # Clear model

# for prompt in PROMPTS:
#     cli_cmd = [
#         SD_CPP_CLI,
#         "--model",
#         MODEL_PATH,
#         "--control-net",
#         CONTROLNET_MODEL_PATH,
#         "--prompt",
#         prompt["prompt"],
#         "--control-image",
#         INPUT_IMAGE_PATH,
#         "--canny" if prompt["canny"] else None,
#         "--output",
#         f"{OUTPUT_DIR}/controlnet{prompt['add']}_cli.png",
#         "-v",
#     ]

#     # Remove None values
#     cli_cmd = [arg for arg in cli_cmd if arg is not None]

#     print(" ".join(cli_cmd))
#     subprocess.run(cli_cmd, check=True)
