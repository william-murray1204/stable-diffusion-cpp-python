from stable_diffusion_cpp import StableDiffusion

MODEL_PATH = "C:\\stable-diffusion\\turbovisionxlSuperFastXLBasedOnNew_tvxlV431Bakedvae.q8_0.gguf"
LORA_PATH = "C:\\Users\\Willi\Downloads\\lora_glowneon_xl_v1.safetensors"

stable_diffusion = StableDiffusion(model_path=MODEL_PATH, lora_model_dir=LORA_PATH)


def callback(step: int, steps: int, time: float):
    print("Completed step: {} of {}".format(step, steps))


images = stable_diffusion.txt_to_img(
    prompt="a lovely cat", progress_callback=callback, batch_count=2
)

for i, image in enumerate(images):
    image.save(f"output_txt2img_{i}.png")
