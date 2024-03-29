import time
from stable_diffusion_cpp import StableDiffusion

import faulthandler
faulthandler.enable()

MODEL_PATH = "C:\\stable-diffusion\\turbovisionxlSuperFastXLBasedOnNew_tvxlV431Bakedvae.q8_0.gguf"

stable_diffusion = StableDiffusion(model_path=MODEL_PATH, verbose=True)

stable_diffusion = None
print("Model unloaded properly - WATCH FOR MEMORY LEAKS!")

time.sleep(10)
