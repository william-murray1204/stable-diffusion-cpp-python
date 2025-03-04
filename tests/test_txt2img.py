import os
import traceback

from stable_diffusion_cpp import StableDiffusion

# MODEL_PATH = "C:\\stable-diffusion\\turbovisionxlSuperFastXLBasedOnNew_tvxlV431Bakedvae.q8_0.gguf"  # GGUF model wont work for LORAs (GGML_ASSERT error)
# MODEL_PATH = "C:\\stable-diffusion\\turbovisionxlSuperFastXLBasedOnNew_tvxlV431Bakedvae.safetensors"
MODEL_PATH = "C:\\stable-diffusion\\catCitronAnimeTreasure_v10.safetensors"

LORA_DIR = "C:\\stable-diffusion\\loras"

stable_diffusion = StableDiffusion(
    model_path=MODEL_PATH,
    lora_model_dir=LORA_DIR,
)


def callback(step: int, steps: int, time: float):
    print("Completed step: {} of {}".format(step, steps))


try:
    prompts = [
        # {"add": "_lora", "prompt": "a lovely cat <lora:realism_lora:1>"},  # With LORA
        # {"add": "", "prompt": "a lovely cat"},  # Without LORA
        {"add": "_lora", "prompt": "a cute cat glass statue <lora:glass_statue_v1:1>"},  # With LORA
        {"add": "2", "prompt": "a cute cat glass statue"},  # Without LORA
    ]

    OUTPUT_DIR = "tests/outputs"
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)

    for prompt in prompts:
        # Generate images
        images = stable_diffusion.txt_to_img(
            prompt=prompt["prompt"],
            sample_steps=4,
            progress_callback=callback,
        )

        # Save images
        for i, image in enumerate(images):
            image.save(f"{OUTPUT_DIR}/txt2img{prompt['add']}_{i}.png")

except Exception as e:
    traceback.print_exc()
    print("Test - txt2img failed: ", e)
