import os
import multiprocessing

from PIL import PngImagePlugin
from conftest import OUTPUT_DIR

from stable_diffusion_cpp import StableDiffusion

MODEL_PATH = "F:\\stable-diffusion\\turbovisionxlSuperFastXLBasedOnNew_tvxlV431Bakedvae.safetensors"

PROMPT = "a cute cat"
STEPS = 20


def generate_on_gfx(hip_visible_devices: int, hsa_override_gfx_version: str):
    os.environ["HIP_VISIBLE_DEVICES"] = str(hip_visible_devices)
    os.environ["HSA_OVERRIDE_GFX_VERSION"] = hsa_override_gfx_version

    stable_diffusion = StableDiffusion(model_path=MODEL_PATH)

    def progress_callback(step: int, steps: int, time: float):
        print("{} - Completed step: {} of {}".format(hsa_override_gfx_version, step, steps))

    # Generate image
    image = stable_diffusion.generate_image(
        prompt=PROMPT,
        sample_steps=STEPS,
        progress_callback=progress_callback,
    )[0]

    # Save image
    pnginfo = PngImagePlugin.PngInfo()
    pnginfo.add_text("Parameters", ", ".join([f"{k.replace('_', ' ').title()}: {v}" for k, v in image.info.items()]))
    image.save(f"{OUTPUT_DIR}/multi_gpu_{hsa_override_gfx_version}.png", pnginfo=pnginfo)


def test_multi_gpu():

    jobs = [
        (0, "10.3.0"),  # 6800XT
        (1, "11.0.1"),  # 7800XT
    ]

    with multiprocessing.Pool(len(jobs)) as pool:
        pool.starmap(generate_on_gfx, jobs)
