from stable_diffusion_cpp import StableDiffusion

# MODEL_PATH = "C:\\stable-diffusion\\turbovisionxlSuperFastXLBasedOnNew_tvxlV431Bakedvae.q8_0.gguf"  # GGUF model wont work for LORAs (GGML_ASSERT error)
MODEL_PATH = "C:\\stable-diffusion\\turbovisionxlSuperFastXLBasedOnNew_tvxlV431Bakedvae.safetensors"

LORA_DIR = "C:\\stable-diffusion\\loras"

stable_diffusion = StableDiffusion(model_path=MODEL_PATH, lora_model_dir=LORA_DIR)


def callback(step: int, steps: int, time: float):
    print("Completed step: {} of {}".format(step, steps))


try:
    prompts = [
        {"add": "_lora", "prompt": "a lovely cat <lora:pixel_art:1>"},  # With LORA
        {"add": "", "prompt": "a lovely cat"},  # Without LORA
    ]

    for prompt in prompts:
        # Generate images
        images = stable_diffusion.txt_to_img(
            prompt=prompt["prompt"],
            sample_steps=4,
            progress_callback=callback,
        )
        # Save images
        for i, image in enumerate(images):
            image.save(f"output_txt2img{prompt['add']}_{i}.png")

except Exception as e:
    print("Test - txt2img failed: ", e)
