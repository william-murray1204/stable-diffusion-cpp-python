import os
import traceback

from stable_diffusion_cpp import StableDiffusion

# MODEL_PATH = "C:\\stable-diffusion\\turbovisionxlSuperFastXLBasedOnNew_tvxlV431Bakedvae.q8_0.gguf"  # GGUF model wont work for LORAs (GGML_ASSERT error)
# MODEL_PATH = "C:\\stable-diffusion\\turbovisionxlSuperFastXLBasedOnNew_tvxlV431Bakedvae.safetensors"
MODEL_PATH = "C:\\stable-diffusion\\catCitronAnimeTreasure_v10.safetensors"

LORA_DIR = "C:\\stable-diffusion\\loras"

PROMPTS = [
    # {"add": "_lora", "prompt": "a lovely cat <lora:realism_lora:1>"},  # With LORA
    # {"add": "", "prompt": "a lovely cat"},  # Without LORA
    # {"add": "_lora", "prompt": "a cute cat glass statue <lora:glass_statue_v1:1>"},  # With LORA
    {"add": "", "prompt": "a cute cat glass statue"},  # Without LORA
]
STEPS = 4

OUTPUT_DIR = "tests/outputs"
if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)


stable_diffusion = StableDiffusion(
    model_path=MODEL_PATH,
    lora_model_dir=LORA_DIR,
)


def callback(step: int, steps: int, time: float):
    print("Completed step: {} of {}".format(step, steps))


try:
    for prompt in PROMPTS:
        # Generate images
        images = stable_diffusion.generate_image(
            prompt=prompt["prompt"],
            sample_steps=STEPS,
            progress_callback=callback,
        )

        # Save images
        for i, image in enumerate(images):
            image.save(f"{OUTPUT_DIR}/txt2img{prompt['add']}_{i}.png")

except Exception as e:
    traceback.print_exc()
    print("Test - txt2img failed: ", e)

# # ======== C++ CLI ========

# import subprocess

# stable_diffusion = None  # Clear model

# SD_CPP_CLI = "C:\\Users\\Willi\\Documents\\GitHub\\stable-diffusion.cpp\\build\\bin\\sd"

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
