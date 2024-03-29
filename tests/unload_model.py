import time
from stable_diffusion_cpp import StableDiffusion

import faulthandler
faulthandler.enable()

MODEL_PATH = "C:\\stable-diffusion\\turbovisionxlSuperFastXLBasedOnNew_tvxlV431Bakedvae.q8_0.gguf"
UPSCALER_MODEL_PATH = "C:\\stable-diffusion\\RealESRGAN_x4plus.pth"

# stable_diffusion = StableDiffusion(upscaler_path=UPSCALER_MODEL_PATH, verbose=True)

stable_diffusion = StableDiffusion(model_path=MODEL_PATH, verbose=True)

stable_diffusion = None
print("Model unloaded properly - WATCH FOR MEMORY LEAKS!")

time.sleep(10)
