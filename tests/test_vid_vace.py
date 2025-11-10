import os

from conftest import OUTPUT_DIR, save_video_ffmpeg

from stable_diffusion_cpp import StableDiffusion

DIFFUSION_MODEL_PATH = "F:\\stable-diffusion\\wan\\wan2.1-v3-vace-1.3b-q8_0.gguf"
T5XXL_PATH = "F:\\stable-diffusion\\wan\\umt5-xxl-encoder-Q8_0.gguf"
VAE_PATH = "F:\\stable-diffusion\\wan\\wan_2.1_vae.safetensors"

VIDEO_FRAMES_DIR = "assets\\frames"


PROMPT = "dogs boxing"
NEGATIVE_PROMPT = "色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指，画得不好的手部，画得不好的脸部， 畸形的，毁容的，形态畸形的肢体，手指融合，静止不动的画面，杂乱的背景，三条腿，背景人很多，倒着走"
STEPS = 4
CFG_SCALE = 6.0
SAMPLE_METHOD = "euler"
WIDTH = 512
HEIGHT = 512
VIDEO_FRAMES = 41
FPS = 16


def test_vid_vace():

    stable_diffusion = StableDiffusion(
        diffusion_model_path=DIFFUSION_MODEL_PATH,
        t5xxl_path=T5XXL_PATH,
        vae_path=VAE_PATH,
        keep_clip_on_cpu=True,
    )

    def progress_callback(step: int, steps: int, time: float):
        print("Completed step: {} of {}".format(step, steps))

    # Generate video frames
    images = stable_diffusion.generate_video(
        prompt=PROMPT,
        negative_prompt=NEGATIVE_PROMPT,
        cfg_scale=CFG_SCALE,
        sample_steps=STEPS,
        sample_method=SAMPLE_METHOD,
        width=WIDTH,
        height=HEIGHT,
        video_frames=VIDEO_FRAMES,
        control_frames=[
            os.path.join(VIDEO_FRAMES_DIR, f)
            for f in os.listdir(VIDEO_FRAMES_DIR)
            if f.lower().endswith((".png", ".jpg", ".jpeg", ".bmp"))
        ],
        progress_callback=progress_callback,
    )

    # Save video
    save_video_ffmpeg(images, fps=FPS, out_path=f"{OUTPUT_DIR}/vid_vace.mp4")

    # VID_DIR = f"{OUTPUT_DIR}/vid"
    # if not os.path.exists(VID_DIR):
    #     os.makedirs(VID_DIR)
    # for i, image in enumerate(images):
    #     image.save(f"{VID_DIR}/vid_{i}.png")


# ===========================================
# C++ CLI
# ===========================================


# import subprocess

# from conftest import SD_CPP_CLI

# stable_diffusion = None  # Clear model


# cli_cmd = [
#     SD_CPP_CLI,
#     "--mode",
#     "vid_gen",
#     "--diffusion-model",
#     DIFFUSION_MODEL_PATH,
#     "--vae",
#     VAE_PATH,
#     "--t5xxl",
#     T5XXL_PATH,
#     "--prompt",
#     PROMPT,
#     "--negative-prompt",
#     NEGATIVE_PROMPT,
#     "--steps",
#     str(STEPS),
#     "--cfg-scale",
#     str(CFG_SCALE),
#     "--sampling-method",
#     SAMPLE_METHOD,
#     "--width",
#     str(WIDTH),
#     "--height",
#     str(HEIGHT),
#     "--video-frames",
#     str(VIDEO_FRAMES),
#     "--fps",
#     str(FPS),
#     "--control-video",
#     VIDEO_FRAMES_DIR,
#     "--clip-on-cpu",
#     "--output",
#     f"{OUTPUT_DIR}/vid_vace_cli.png",
#     "-v",
# ]
# print(" ".join(cli_cmd))
# subprocess.run(cli_cmd, check=True)
