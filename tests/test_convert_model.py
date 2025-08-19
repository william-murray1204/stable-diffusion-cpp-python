import os
import traceback

import stable_diffusion_cpp.stable_diffusion_cpp as sd_cpp

MODEL_PATH = "C:\\stable-diffusion\\turbovisionxlSuperFastXLBasedOnNew_tvxlV431Bakedvae.safetensors"

OUTPUT_MODEL_DIR = "tests\\outputs"
OUTPUT_MODEL_PATH = f"{OUTPUT_MODEL_DIR}\\new_model.gguf"

try:
    if not os.path.exists(OUTPUT_MODEL_DIR):
        os.makedirs(OUTPUT_MODEL_DIR)

    model_converted = sd_cpp.convert(
        MODEL_PATH.encode("utf-8"),
        "".encode("utf-8"),
        OUTPUT_MODEL_PATH.encode("utf-8"),
        sd_cpp.GGMLType.SD_TYPE_Q8_0,
        "".encode("utf-8"),
    )
    print("Model converted: ", model_converted)

except Exception as e:
    traceback.print_exc()
    print("Test - convert_model failed: ", e)
