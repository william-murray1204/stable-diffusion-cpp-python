from stable_diffusion_cpp import StableDiffusion

MODEL_PATH = "C:\\stable-diffusion\\v15PrunedEmaonly.safetensors"
CONTROLNET_MODEL_PATH = (
    "C:\\stable-diffusion\\control_nets\\control_openpose-fp16.safetensors"
)

INPUT_IMAGE_PATH = "assets\\input.png"

stable_diffusion = StableDiffusion(
    model_path=MODEL_PATH, control_net_path=CONTROLNET_MODEL_PATH
)


def callback(step: int, steps: int, time: float):
    print("Completed step: {} of {}".format(step, steps))


try:
    prompts = [
        {"add": "", "prompt": "a lovely cat", "canny": False},
        {"add": "_canny", "prompt": "a lovely cat", "canny": True},
    ]

    for prompt in prompts:
        # Generate images
        images = stable_diffusion.txt_to_img(
            prompt=prompt["prompt"],
            control_cond=INPUT_IMAGE_PATH,
            canny=prompt["canny"],
            progress_callback=callback,
        )
        # Save images
        for i, image in enumerate(images):
            image.save(f"output_controlnet{prompt['add']}_{i}.png")

except Exception as e:
    print("Test - controlnet failed: ", e)
