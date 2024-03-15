import stable_diffusion_cpp.stable_diffusion_cpp as stable_diffusion_cpp
from .stable_diffusion_types import StableDiffusionToken
from typing import List, Literal, Any


class StableDiffusion:
    def __init__(self, model_path, n_threads=1):
        self.params.n_threads = n_threads
        self.params.model_path = model_path

    def generate(self):
        print("Generating")
