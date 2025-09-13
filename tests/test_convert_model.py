from conftest import OUTPUT_DIR

import stable_diffusion_cpp.stable_diffusion_cpp as sd_cpp

MODEL_PATH = "C:\\stable-diffusion\\turbovisionxlSuperFastXLBasedOnNew_tvxlV431Bakedvae.safetensors"


def test_convert_model():

    model_converted = sd_cpp.convert(
        MODEL_PATH.encode("utf-8"),
        "".encode("utf-8"),
        f"{OUTPUT_DIR}/new_model.gguf".encode("utf-8"),
        sd_cpp.GGMLType.SD_TYPE_Q8_0,
        "".encode("utf-8"),
    )
    print("Model converted: ", model_converted)
