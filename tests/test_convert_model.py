from conftest import OUTPUT_DIR

from stable_diffusion_cpp import StableDiffusion

MODEL_PATH = "F:\\stable-diffusion\\turbovisionxlSuperFastXLBasedOnNew_tvxlV431Bakedvae.safetensors"


def test_convert_model():

    stable_diffusion = StableDiffusion()

    model_converted = stable_diffusion.convert(
        input_path=MODEL_PATH,
        output_path=f"{OUTPUT_DIR}/convert_model.gguf",
        output_type="q8_0",
    )
    print("Model converted: ", model_converted)
