import os
import traceback

from stable_diffusion_cpp import StableDiffusion

MODEL_PATH = "C:\\stable-diffusion\\v1-5-pruned-emaonly.safetensors"
CONTROLNET_MODEL_PATH = "C:\\stable-diffusion\\control_nets\\control_openpose-fp16.safetensors"

INPUT_IMAGE_PATH = "assets\\input.png"


PROMPTS = [
    {"add": "", "prompt": "a lovely cat", "canny": False},
    {"add": "_canny", "prompt": "a lovely cat", "canny": True},
]

OUTPUT_DIR = "tests/outputs"
if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)


stable_diffusion = StableDiffusion(model_path=MODEL_PATH, control_net_path=CONTROLNET_MODEL_PATH)


def callback(step: int, steps: int, time: float):
    print("Completed step: {} of {}".format(step, steps))


try:
    for prompt in PROMPTS:
        # Generate images
        images = stable_diffusion.generate_image(
            prompt=prompt["prompt"],
            control_cond=INPUT_IMAGE_PATH,
            canny=prompt["canny"],
            progress_callback=callback,
        )

        # Save images
        for i, image in enumerate(images):
            image.save(f"{OUTPUT_DIR}/controlnet{prompt['add']}_{i}.png")

except Exception as e:
    traceback.print_exc()
    print("Test - controlnet failed: ", e)


# # ======== C++ CLI ========

# import subprocess

# stable_diffusion = None  # Clear model

# SD_CPP_CLI = "C:\\Users\\Willi\\Documents\\GitHub\\stable-diffusion.cpp\\build\\bin\\sd"

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
#         "--canny" if prompt["canny"] else "",
#         "--output",
#         f"{OUTPUT_DIR}/controlnet{prompt['add']}_cli.png",
#         "-v",
#     ]
#     print(" ".join(cli_cmd))
#     subprocess.run(cli_cmd, check=True)
