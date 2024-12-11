import os
import traceback

from stable_diffusion_cpp import StableDiffusion

MODEL_PATH = "C:\\stable-diffusion\\photomaker\\sdxlUnstableDiffusers_v11.safetensors"
STACKED_ID_EMBED_DIR = "C:\\stable-diffusion\\photomaker\\photomaker-v1.safetensors"
VAE_PATH = "C:\\stable-diffusion\\photomaker\\sdxl.vae.safetensors"

INPUT_ID_IMAGES_PATH = ".\\assets\\newton_man"

stable_diffusion = StableDiffusion(stacked_id_embed_dir=STACKED_ID_EMBED_DIR, model_path=MODEL_PATH, vae_path=VAE_PATH)


def callback(step: int, steps: int, time: float):
    print("Completed step: {} of {}".format(step, steps))


try:
    # Generate images
    # images = stable_diffusion.txt_to_img(
    #     cfg_scale=5.0,
    #     height=1024,
    #     width=1024,
    #     style_strength=10,  # style_ratio (%)
    #     sample_method="euler",
    #     prompt="a man img, retro futurism, retro game art style but extremely beautiful, intricate details, masterpiece, best quality, space-themed, cosmic, celestial, stars, galaxies, nebulas, planets, science fiction, highly detailed",
    #     negative_prompt="realistic, photo-realistic, worst quality, greyscale, bad anatomy, bad hands, error, text",
    #     progress_callback=callback,
    # )

    # Generate images
    photomaker_images = stable_diffusion.txt_to_img(
        cfg_scale=5.0,
        height=1024,
        width=1024,
        style_strength=10,  # style_ratio (%)
        sample_method="euler",
        prompt="a man img, retro futurism, retro game art style but extremely beautiful, intricate details, masterpiece, best quality, space-themed, cosmic, celestial, stars, galaxies, nebulas, planets, science fiction, highly detailed",
        negative_prompt="realistic, photo-realistic, worst quality, greyscale, bad anatomy, bad hands, error, text",
        input_id_images_path=INPUT_ID_IMAGES_PATH,
        progress_callback=callback,
    )

    OUTPUT_DIR = "tests/outputs"
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)

    # # Save images
    # for i, image in enumerate(images):
    #     image.save(f"{OUTPUT_DIR}/non_photomaker_{i}.png")

    for i, image in enumerate(photomaker_images):
        image.save(f"{OUTPUT_DIR}/photomaker_{i}.png")

except Exception as e:
    traceback.print_exc()
    print("Test - photomaker failed: ", e)
