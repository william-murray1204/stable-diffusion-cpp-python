from PIL import PngImagePlugin
from conftest import OUTPUT_DIR

from stable_diffusion_cpp import StableDiffusion

DIFFUSION_MODEL_PATH = "F:\\stable-diffusion\\qwen\\Qwen_Image-Q4_K_M.gguf"
VAE_PATH = "F:\\stable-diffusion\\qwen\\qwen_image_vae.safetensors"
LLM_PATH = "F:\\stable-diffusion\\qwen\\Qwen2.5-VL-7B-Instruct.Q8_0.gguf"


PROMPT = '一个穿着"QWEN"标志的T恤的中国美女正拿着黑色的马克笔面相镜头微笑。她身后的玻璃板上手写体写着 “一、Qwen-Image的技术路线： 探索视觉生成基础模型的极限，开创理解与生成一体化的未来。二、Qwen-Image的模型特色：1、复杂文字渲染。支持中英渲染、自动布局； 2、精准图像编辑。支持文字编辑、物体增减、风格变换。三、Qwen-Image的未来愿景：赋能专业内容创作、助力生成式AI发展。”'

STEPS = 10
CFG_SCALE = 2.5
SAMPLE_METHOD = "euler"
FLOW_SHIFT = 3


def test_qwen_image():

    stable_diffusion = StableDiffusion(
        diffusion_model_path=DIFFUSION_MODEL_PATH,
        llm_path=LLM_PATH,
        vae_path=VAE_PATH,
        offload_params_to_cpu=True,
        flow_shift=FLOW_SHIFT,
    )

    def progress_callback(step: int, steps: int, time: float):
        print("Completed step: {} of {}".format(step, steps))

    # Generate image
    image = stable_diffusion.generate_image(
        prompt=PROMPT,
        sample_steps=STEPS,
        cfg_scale=CFG_SCALE,
        sample_method=SAMPLE_METHOD,
        progress_callback=progress_callback,
    )[0]

    # Save image
    pnginfo = PngImagePlugin.PngInfo()
    pnginfo.add_text("Parameters", ", ".join([f"{k.replace('_', ' ').title()}: {v}" for k, v in image.info.items()]))
    image.save(f"{OUTPUT_DIR}/qwen_image.png", pnginfo=pnginfo)


# ===========================================
# C++ CLI
# ===========================================

# import subprocess

# from conftest import SD_CPP_CLI

# stable_diffusion = None  # Clear model


# cli_cmd = [
#     SD_CPP_CLI,
#     "--diffusion-model",
#     DIFFUSION_MODEL_PATH,
#     "--vae",
#     VAE_PATH,
#     "--llm",
#     LLM_PATH,
#     "--prompt",
#     PROMPT,
#     "--steps",
#     str(STEPS),
#     "--cfg-scale",
#     str(CFG_SCALE),
#     "--sampling-method",
#     SAMPLE_METHOD,
#     "--flow-shift",
#     str(FLOW_SHIFT),
#     "--offload-to-cpu",
#     "--output",
#     f"{OUTPUT_DIR}/qwen_image_cli.png",
#     "-v",
# ]
# print(" ".join(cli_cmd))
# subprocess.run(cli_cmd, check=True)
