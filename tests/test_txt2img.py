from PIL import PngImagePlugin
from conftest import OUTPUT_DIR

from stable_diffusion_cpp import StableDiffusion

MODEL_PATH = "F:\\stable-diffusion\\catCitronAnimeTreasure_v10.safetensors"
LORA_DIR = "F:\\stable-diffusion\\loras"


PROMPTS = [
    {"add": "_lora", "prompt": "a cute cat glass statue <lora:glass_statue_v1:1>"},  # With LORA
    {"add": "", "prompt": "a cute cat glass statue"},  # Without LORA
]
STEPS = 20


def test_txt2img():

    stable_diffusion = StableDiffusion(
        model_path=MODEL_PATH,
        lora_model_dir=LORA_DIR,
    )

    def callback(step: int, steps: int, time: float):
        print("Completed step: {} of {}".format(step, steps))

    for prompt in PROMPTS:
        # Generate image
        image = stable_diffusion.generate_image(
            prompt=prompt["prompt"],
            sample_steps=STEPS,
            progress_callback=callback,
        )[0]

        # Save image
        pnginfo = PngImagePlugin.PngInfo()
        pnginfo.add_text("Parameters", ", ".join([f"{k.replace('_', ' ').title()}: {v}" for k, v in image.info.items()]))
        image.save(f"{OUTPUT_DIR}/txt2img{prompt['add']}.png", pnginfo=pnginfo)


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
#         "--lora-model-dir",
#         LORA_DIR,
#         "--prompt",
#         prompt["prompt"],
#         "--steps",
#         str(STEPS),
#         "--output",
#         f"{OUTPUT_DIR}/txt2img{prompt['add']}_cli.png",
#         "-v",
#     ]
#     print(" ".join(cli_cmd))
#     subprocess.run(cli_cmd, check=True)
