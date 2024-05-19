from stable_diffusion_cpp import StableDiffusion

INPUT_IMAGE_PATH = "assets\\input.png"

stable_diffusion = StableDiffusion()

try:
    # Apply canny edge detection
    image = stable_diffusion.preprocess_canny(image=INPUT_IMAGE_PATH)

    # Save image
    image.save(f"output_preprocess_canny.png")

except Exception as e:
    print("Test - preprocess_canny failed: ", e)
