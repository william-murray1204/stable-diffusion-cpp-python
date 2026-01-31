# 🖼️ Python Bindings for [`stable-diffusion.cpp`](https://github.com/leejet/stable-diffusion.cpp)

Simple Python bindings for **@leejet's** [`stable-diffusion.cpp`](https://github.com/leejet/stable-diffusion.cpp) library.

[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](https://opensource.org/licenses/MIT)
[![PyPi version](https://badgen.net/pypi/v/pywhispercpp)](https://pypi.org/project/stable-diffusion-cpp-python/)
[![Downloads](https://static.pepy.tech/badge/stable-diffusion-cpp-python)](https://pepy.tech/project/stable-diffusion-cpp-python)

This package provides:

- Low-level access to C API via `ctypes` interface.
- High-level Python API for Stable Diffusion, FLUX and Wan image/video generation.

## Installation

Requirements:

- Python 3.8+
- C compiler
  - Linux: gcc or clang
  - Windows: Visual Studio or MinGW
  - MacOS: Xcode

To install the package, run:

```bash
pip install stable-diffusion-cpp-python
```

This will also build `stable-diffusion.cpp` from source and install it alongside this python package.

If this fails, add `--verbose` to the `pip install` to see the full cmake build log.

### Installation Configuration

`stable-diffusion.cpp` supports a number of hardware acceleration backends to speed up inference as well as backend specific options. See the [stable-diffusion.cpp README](https://github.com/leejet/stable-diffusion.cpp#build) for a full list.

All `stable-diffusion.cpp` cmake build options can be set via the `CMAKE_ARGS` environment variable or via the `--config-settings / -C` cli flag during installation.

<details open>
<summary>Environment Variables</summary>

```bash
# Linux and Mac
CMAKE_ARGS="-DSD_CUDA=ON" pip install stable-diffusion-cpp-python
```

```powershell
# Windows
$env:CMAKE_ARGS="-DSD_CUDA=ON"
pip install stable-diffusion-cpp-python
```

</details>

<details>
<summary>CLI / requirements.txt</summary>

They can also be set via `pip install -C / --config-settings` command and saved to a `requirements.txt` file:

```bash
pip install --upgrade pip # ensure pip is up to date
pip install stable-diffusion-cpp-python -C cmake.args="-DSD_CUDA=ON"
```

```txt
# requirements.txt

stable-diffusion-cpp-python -C cmake.args="-DSD_CUDA=ON"
```

</details>

### Supported Backends

Below are some common backends, their build commands and any additional environment variables required.

<!-- CUDA -->
<details>
<summary>Using CUDA (CUBLAS)</summary>

This provides BLAS acceleration using the CUDA cores of your Nvidia GPU. Make sure you have the CUDA toolkit installed. You can download it from your Linux distro's package manager (e.g. `apt install nvidia-cuda-toolkit`) or from here: [CUDA Toolkit](https://developer.nvidia.com/cuda-downloads). You can check your installed CUDA toolkit version by running `nvcc --version`.

- It is recommended you have at least 4 GB of VRAM.

```bash
CMAKE_ARGS="-DSD_CUDA=ON" pip install stable-diffusion-cpp-python
```

</details>

<!-- HIPBLAS -->
<details>
<summary>Using HIPBLAS (ROCm)</summary>

This provides BLAS acceleration using the ROCm cores of your AMD GPU. Make sure you have the ROCm toolkit installed and that you replace the `$GFX_NAME` value with that of your GPU architecture (`gfx1030` for consumer RDNA2 cards for example).Windows users refer to [docs/hipBLAS_on_Windows.md](docs%2FhipBLAS_on_Windows.md) for a comprehensive guide and troubleshooting tips.

```bash
if command -v rocminfo; then export GFX_NAME=$(rocminfo | awk '/ *Name: +gfx[1-9]/ {print $2; exit}'); else echo "rocminfo missing!"; fi
if [ -z "${GFX_NAME}" ]; then echo "Error: Couldn't detect GPU!"; else echo "Building for GPU: ${GFX_NAME}"; fi

CMAKE_ARGS="-G Ninja -DCMAKE_C_COMPILER=clang -DCMAKE_CXX_COMPILER=clang++ -DSD_HIPBLAS=ON -DCMAKE_BUILD_TYPE=Release -DGPU_TARGETS=$GFX_NAME -DAMDGPU_TARGETS=$GFX_NAME -DCMAKE_BUILD_WITH_INSTALL_RPATH=ON -DCMAKE_POSITION_INDEPENDENT_CODE=ON" pip install stable-diffusion-cpp-python
```

</details>

<!-- Metal -->
<details>
<summary>Using Metal</summary>

Using Metal runs the computation on Apple Silicon. Currently, there are some issues with Metal when performing operations on very large matrices, making it highly inefficient. Performance improvements are expected in the near future.

```bash
CMAKE_ARGS="-DSD_METAL=ON" pip install stable-diffusion-cpp-python
```

</details>

<!-- Vulkan -->
<details>
<summary>Using Vulkan</summary>
Install Vulkan SDK from https://www.lunarg.com/vulkan-sdk/.

```bash
CMAKE_ARGS="-DSD_VULKAN=ON" pip install stable-diffusion-cpp-python
```

</details>

<!-- SYCL -->
<details>
<summary>Using SYCL</summary>

Using SYCL runs the computation on an Intel GPU. Please make sure you have installed the related driver and [Intel® oneAPI Base toolkit](https://www.intel.com/content/www/us/en/developer/tools/oneapi/base-toolkit.html) before starting. For more details refer to [llama.cpp SYCL backend](https://github.com/ggml-org/llama.cpp/blob/master/docs/backend/SYCL.md#linux).

```bash
# Export relevant ENV variables
source /opt/intel/oneapi/setvars.sh

# Option 1: Use FP32 (recommended for better performance in most cases)
CMAKE_ARGS="-DSD_SYCL=ON -DCMAKE_C_COMPILER=icx -DCMAKE_CXX_COMPILER=icpx" pip install stable-diffusion-cpp-python

# Option 2: Use FP16
CMAKE_ARGS="-DSD_SYCL=ON -DCMAKE_C_COMPILER=icx -DCMAKE_CXX_COMPILER=icpx -DGGML_SYCL_F16=ON" pip install stable-diffusion-cpp-python
```

</details>

<!-- OpenBLAS -->
<details>
<summary>Using OpenBLAS</summary>

```bash
CMAKE_ARGS="-DGGML_OPENBLAS=ON" pip install stable-diffusion-cpp-python
```

</details>

<!-- MUSA -->
<details>
<summary>Using MUSA</summary>

This provides BLAS acceleration using the MUSA cores of your Moore Threads GPU. Make sure to have the MUSA toolkit installed.

```bash
CMAKE_ARGS="-DCMAKE_C_COMPILER=/usr/local/musa/bin/clang -DCMAKE_CXX_COMPILER=/usr/local/musa/bin/clang++ -DSD_MUSA=ON -DCMAKE_BUILD_TYPE=Release" pip install stable-diffusion-cpp-python
```

</details>

<!-- OpenCL -->
<details>
<summary>Using OpenCL (Adreno GPU)</summary>

Currently, it only supports Adreno GPUs and is primarily optimized for Q4_0 type.

To build for Windows ARM please refers to [Windows 11 Arm64](https://github.com/ggml-org/llama.cpp/blob/master/docs/backend/OPENCL.md#windows-11-arm64)

Building for Android:

Android NDK:

- Download and install the Android NDK from the [official Android developer site](https://developer.android.com/ndk/downloads).

Setup OpenCL Dependencies for NDK:
You need to provide OpenCL headers and the ICD loader library to your NDK sysroot.

- OpenCL Headers:

  ```bash
  # In a temporary working directory
  git clone https://github.com/KhronosGroup/OpenCL-Headers
  cd OpenCL-Headers

  # Replace <YOUR_NDK_PATH> with your actual NDK installation path
  # e.g., cp -r CL /path/to/android-ndk-r26c/toolchains/llvm/prebuilt/linux-x86_64/sysroot/usr/include
  sudo cp -r CL <YOUR_NDK_PATH>/toolchains/llvm/prebuilt/linux-x86_64/sysroot/usr/include
  cd ..
  ```

- OpenCL ICD Loader:

  ```bash
  # In the same temporary working directory
  git clone https://github.com/KhronosGroup/OpenCL-ICD-Loader
  cd OpenCL-ICD-Loader
  mkdir build_ndk && cd build_ndk

  # Replace <YOUR_NDK_PATH> in the CMAKE_TOOLCHAIN_FILE and OPENCL_ICD_LOADER_HEADERS_DIR
  cmake .. -G Ninja -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_TOOLCHAIN_FILE=<YOUR_NDK_PATH>/build/cmake/android.toolchain.cmake \
    -DOPENCL_ICD_LOADER_HEADERS_DIR=<YOUR_NDK_PATH>/toolchains/llvm/prebuilt/linux-x86_64/sysroot/usr/include \
    -DANDROID_ABI=arm64-v8a \
    -DANDROID_PLATFORM=24 \
    -DANDROID_STL=c++_shared

  ninja
  # Replace <YOUR_NDK_PATH>
  # e.g., cp libOpenCL.so /path/to/android-ndk-r26c/toolchains/llvm/prebuilt/linux-x86_64/sysroot/usr/lib/aarch64-linux-android
  sudo cp libOpenCL.so <YOUR_NDK_PATH>/toolchains/llvm/prebuilt/linux-x86_64/sysroot/usr/lib/aarch64-linux-android
  cd ../..
  ```

Build `stable-diffusion-cpp-python` for Android with (untested):

```bash
# Replace <YOUR_NDK_PATH> with your actual NDK installation path
# e.g., -DCMAKE_TOOLCHAIN_FILE=/path/to/android-ndk-r26c/build/cmake/android.toolchain.cmake
CMAKE_ARGS="-G Ninja -DCMAKE_TOOLCHAIN_FILE=<YOUR_NDK_PATH>/build/cmake/android.toolchain.cmake -DANDROID_ABI=arm64-v8a -DANDROID_PLATFORM=android-28 -DGGML_OPENMP=OFF -DSD_OPENCL=ON
```

_(Note: Don't forget to include `LD_LIBRARY_PATH=/vendor/lib64` in your command line before running the binary)_

</details>

### Upgrading and Reinstalling

To upgrade and rebuild `stable-diffusion-cpp-python` add `--upgrade --force-reinstall --no-cache-dir` flags to the `pip install` command to ensure the package is rebuilt from source.

### Using Flash Attention

Enabling flash attention for the diffusion model reduces memory usage by varying amounts of MB, e.g.:

- **flux 768x768** ~600mb
- **SD2 768x768** ~1400mb

For most backends, it slows things down, but for cuda it generally speeds it up too.
At the moment, it is only supported for some models and some backends (like `cpu`, `cuda/rocm` and `metal`).

Run by passing `diffusion_flash_attn=True` to the `StableDiffusion` class and watch for:

```log
[INFO] stable-diffusion.cpp:312  - Using flash attention in the diffusion model
```

and the compute buffer shrink in the debug log:

```log
[DEBUG] ggml_extend.hpp:1004 - flux compute buffer size: 650.00 MB(VRAM)
```

## High-level API

The high-level API provides a simple managed interface through the `StableDiffusion` class.

Below is a short example demonstrating how to use the high-level API to generate a simple image:

### <u>Text to Image</u>

```python
from PIL import Image
from stable_diffusion_cpp import StableDiffusion

def progress_callback(step: int, steps: int, time: float):
    print("Completed step: {} of {}".format(step, steps))

def preview_callback(step: int, images: list[Image.Image], is_noisy: bool):
    images[0].save(f"preview/{step}.png")

stable_diffusion = StableDiffusion(
      model_path="../models/v1-5-pruned-emaonly.safetensors",
      # wtype="default", # Weight type (e.g. "q8_0", "f16", etc) (The "default" setting is automatically applied and determines the weight type of a model file)
)
output = stable_diffusion.generate_image(
      prompt="a lovely cat",
      width=512,
      height=512,
      progress_callback=progress_callback,
      # seed=1337, # Uncomment to set a specific seed (use -1 for a random seed)
      preview_method="proj",
      preview_interval=2,  # Call every 2 steps
      preview_callback=preview_callback,
)
output[0].save("output.png") # Output returned as list of PIL Images

# Model and generation paramaters accessible via .info
print(output[0].info)
```

#### <u>With LoRA (Stable Diffusion)</u>

You can specify the directory where the lora weights are stored via `lora_model_dir`. If not specified, the default is the current working directory.

- LoRA is specified via prompt, just like [stable-diffusion-webui](https://github.com/AUTOMATIC1111/stable-diffusion-webui/wiki/Features#lora). (e.g. `<lora:marblesh:1>`)
- LoRAs will not work when using quantized models. You must instead use a full precision `.safetensors` model.

Here's a simple example:

```python
from stable_diffusion_cpp import StableDiffusion

stable_diffusion = StableDiffusion(
      model_path="../models/v1-5-pruned-emaonly.safetensors",
      lora_model_dir="../models/", # This should point to folder where LoRA weights are stored (not an individual file)
)
output = stable_diffusion.generate_image(
      prompt="a lovely cat<lora:marblesh:1>",
)
```

- The `lora_model_dir` argument is used in the same way for FLUX image generation.

---

### <u>FLUX Image Generation</u>

FLUX models should be run using the same implementation as the [stable-diffusion.cpp FLUX documentation](https://github.com/leejet/stable-diffusion.cpp/blob/master/docs/flux.md) where the `diffusion_model_path` argument is used in place of the `model_path`. The `clip_l_path`, `t5xxl_path`, and `vae_path` arguments are also required for inference to function (for most models).

Download the weights from the links below:

- Preconverted gguf weights from [FLUX.1-dev-gguf](https://huggingface.co/leejet/FLUX.1-dev-gguf) or [FLUX.1-schnell](https://huggingface.co/leejet/FLUX.1-schnell-gguf), this way you don't have to do the conversion yourself.
- Download `vae` from https://huggingface.co/black-forest-labs/FLUX.1-dev/blob/main/ae.safetensors
- Download `clip_l` from https://huggingface.co/comfyanonymous/flux_text_encoders/blob/main/clip_l.safetensors
- Download `t5xxl` from https://huggingface.co/comfyanonymous/flux_text_encoders/blob/main/t5xxl_fp16.safetensors

```python
from stable_diffusion_cpp import StableDiffusion

stable_diffusion = StableDiffusion(
    diffusion_model_path="../models/flux1-schnell-q3_k.gguf", # In place of model_path
    clip_l_path="../models/clip_l.safetensors",
    t5xxl_path="../models/t5xxl_fp16.safetensors",
    vae_path="../models/ae.safetensors",
    vae_decode_only=True, # Can be True if not generating image to image
    keep_clip_on_cpu=True, # Prevents black images when using some T5 models
)
output = stable_diffusion.generate_image(
      prompt="a lovely cat holding a sign says 'flux.cpp'",
      cfg_scale=1.0, # a cfg_scale of 1 is recommended for FLUX
      # sample_method="euler", # euler is recommended for FLUX, set automatically if "default" is specified
)
```

#### <u>FLUX.2</u>

Download the weights from the links below:

- Download `FLUX.2-dev`
  - gguf: https://huggingface.co/city96/FLUX.2-dev-gguf/tree/main
- Download `vae`
  - safetensors: https://huggingface.co/black-forest-labs/FLUX.2-dev/tree/main
- Download `Mistral-Small-3.2-24B-Instruct-2506-GGUF`
  - gguf: https://huggingface.co/unsloth/Mistral-Small-3.2-24B-Instruct-2506-GGUF/tree/main

```python
from stable_diffusion_cpp import StableDiffusion

stable_diffusion = StableDiffusion(
      diffusion_model_path="../models/flux2-dev-Q4_K_M.gguf",
      llm_path="../models/Mistral-Small-3.2-24B-Instruct-2506-Q4_K_M.gguf",
      vae_path="../models/ae.safetensors",
      offload_params_to_cpu=True,
      diffusion_flash_attn=True,
)

output = stable_diffusion.generate_image(
      prompt="the cat has a hat",
      ref_images=["input.png"],
      sample_steps=4,
      cfg_scale=1.0,
)
```

#### <u>With LoRA (FLUX)</u>

LoRAs can be used with FLUX models in the same way as Stable Diffusion models ([as shown above](#with-lora-stable-diffusion)).

Note that:

- It is recommended you use LoRAs with naming formats compatible with ComfyUI.
- LoRAs will only work with `Flux-dev q8_0`.
- You can download FLUX LoRA models from https://huggingface.co/XLabs-AI/flux-lora-collection/tree/main (you must use a comfy converted version!!!).

#### <u>Kontext (FLUX)</u>

Download the weights from the links below:

- Preconverted gguf model from [FLUX.1-Kontext-dev-GGUF](https://huggingface.co/QuantStack/FLUX.1-Kontext-dev-GGUF)
- Otherwise, download FLUX.1-Kontext-dev from [black-forest-labs/FLUX.1-Kontext-dev](https://huggingface.co/black-forest-labs/FLUX.1-Kontext-dev/blob/main/flux1-kontext-dev.safetensors)
- The `vae`, `clip_l`, and `t5xxl` models are the same as for FLUX image generation linked above.

```python
from stable_diffusion_cpp import StableDiffusion

stable_diffusion = StableDiffusion(
    diffusion_model_path="../models/flux1-kontext-dev-Q5_K_S.gguf", # In place of model_path
    clip_l_path="../models/clip_l.safetensors",
    t5xxl_path="../models/t5xxl_fp16.safetensors",
    vae_path="../models/ae.safetensors",
    vae_decode_only=False, # Must be False for FLUX Kontext
    keep_clip_on_cpu=True, # Prevents black images when using some T5 models
)
output = stable_diffusion.generate_image(
      prompt="make the cat blue",
      ref_images=["input.png"],
      cfg_scale=1.0, # a cfg_scale of 1 is recommended for FLUX
)
```

#### <u>Chroma (FLUX)</u>

Download the weights from the links below:

- Preconverted gguf model from [silveroxides/Chroma1-Flash-GGUF](https://huggingface.co/silveroxides/Chroma1-Flash-GGUF), [silveroxides/Chroma1-Base-GGUF](https://huggingface.co/silveroxides/Chroma1-Base-GGUF) or [silveroxides/Chroma1-HD-GGUF](https://huggingface.co/silveroxides/Chroma1-HD-GGUF) ([silveroxides/Chroma-GGUF](https://huggingface.co/silveroxides/Chroma-GGUF) is DEPRECATED)
- Otherwise, download chroma's safetensors from [lodestones/Chroma1-Flash](https://huggingface.co/lodestones/Chroma1-Flash), [lodestones/Chroma1-Base](https://huggingface.co/lodestones/Chroma1-Base) or [lodestones/Chroma1-HD](https://huggingface.co/lodestones/Chroma1-HD) ([lodestones/Chroma](https://huggingface.co/lodestones/Chroma) is DEPRECATED)
- The `vae` and `t5xxl` models are the same as for FLUX image generation linked above (`clip_l` not required).

or Chroma Radiance models from:

- safetensors: https://huggingface.co/lodestones/Chroma1-Radiance/tree/main
- gguf: https://huggingface.co/silveroxides/Chroma1-Radiance-GGUF/tree/main
- t5xxl: https://huggingface.co/comfyanonymous/flux_text_encoders/blob/main/t5xxl_fp16.safetensors

```python
from stable_diffusion_cpp import StableDiffusion

stable_diffusion = StableDiffusion(
    diffusion_model_path="../models/Chroma1-HD-Flash-Q4_0.gguf", # In place of model_path
    t5xxl_path="../models/t5xxl_fp16.safetensors",
    vae_path="../models/ae.safetensors",
    vae_decode_only=True, # Can be True if we are not generating image to image
    chroma_use_dit_mask=False,
    keep_clip_on_cpu=True, # Prevents black images when using some T5 models
)
output = stable_diffusion.generate_image(
      prompt="a lovely cat holding a sign says 'chroma.cpp'",
      cfg_scale=4.0, # a cfg_scale of 4 is recommended for Chroma
)
```

---

### <u>Some SD1.x and SDXL distilled models</u>

See [docs/distilled_sd.md](./docs/distilled_sd.md) for instructions on using distilled SD models.

---

### <u>SD3.5 Image Generation</u>

Download the weights from the links below:

- Download `sd3.5_large` from https://huggingface.co/stabilityai/stable-diffusion-3.5-large/blob/main/sd3.5_large.safetensors
- Download `clip_g` from https://huggingface.co/Comfy-Org/stable-diffusion-3.5-fp8/blob/main/text_encoders/clip_g.safetensors
- Download `clip_l` from https://huggingface.co/Comfy-Org/stable-diffusion-3.5-fp8/blob/main/text_encoders/clip_l.safetensors
- Download `t5xxl` from https://huggingface.co/Comfy-Org/stable-diffusion-3.5-fp8/blob/main/text_encoders/t5xxl_fp16.safetensors

```python
from stable_diffusion_cpp import StableDiffusion

stable_diffusion = StableDiffusion(
    model_path="../models/sd3.5_large.safetensors",
    clip_l_path="../models/clip_l.safetensors",
    clip_g_path="../models/clip_g.safetensors",
    t5xxl_path="../models/t5xxl_fp16.safetensors",
    keep_clip_on_cpu=True, # Prevents black images when using some T5 models
)
output = stable_diffusion.generate_image(
      prompt="a lovely cat holding a sign says 'Stable diffusion 3.5 Large'",
      height=1024,
      width=1024,
      cfg_scale=4.5,
      sample_method="euler",
)
```

---

### <u>Image to Image</u>

```python
from stable_diffusion_cpp import StableDiffusion
# from PIL import Image

INPUT_IMAGE = "../input.png"
# INPUT_IMAGE = Image.open("../input.png") # or alternatively, pass as PIL Image

stable_diffusion = StableDiffusion(model_path="../models/v1-5-pruned-emaonly.safetensors")

output = stable_diffusion.generate_image(
      prompt="blue eyes",
      init_image=INPUT_IMAGE, # Note: The input image will be automatically resized to the match the width and height arguments (default: 512x512)
      strength=0.4,
)
```

---

### <u>Inpainting</u>

```python
from stable_diffusion_cpp import StableDiffusion

# Note: Inpainting with a base model gives poor results. A model fine-tuned for inpainting is recommended.
stable_diffusion = StableDiffusion(model_path="../models/v1-5-pruned-emaonly.safetensors")

output = stable_diffusion.generate_image(
      prompt="blue eyes",
      init_image="../input.png",
      mask_image="../mask.png", # A grayscale image where 0 is masked and 255 is unmasked
      strength=0.4,
)
```

---

### <u>PhotoMaker</u>

You can use [PhotoMaker](https://github.com/TencentARC/PhotoMaker) to personalize generated images with your own ID.

**NOTE**, currently PhotoMaker **ONLY** works with **SDXL** (any SDXL model files will work).

Download PhotoMaker model file (in safetensor format) [here](https://huggingface.co/bssrdf/PhotoMaker). The official release of the model file (in .bin format) does not work with `stablediffusion.cpp`.

In prompt, make sure you have a class word followed by the trigger word `"img"` (hard-coded for now). The class word could be one of `"man, woman, girl, boy"`. If input ID images contain asian faces, add `Asian` before the class word.

```python
import os
from stable_diffusion_cpp import StableDiffusion

stable_diffusion = StableDiffusion(
      model_path="../models/sdxl.vae.safetensors",
      vae_path="../models/sdxl.vae.safetensors",
      photo_maker_path="../models/photomaker-v1.safetensors",
      # keep_vae_on_cpu=True,  # If on low memory GPUs (<= 8GB), setting this to True is recommended to get artifact free images
)

INPUT_ID_IMAGES_DIR = "../assets/newton_man"

output = stable_diffusion.generate_image(
      cfg_scale=5.0, # a cfg_scale of 5.0 is recommended for PhotoMaker
      height=1024,
      width=1024,
      pm_style_strength=10,  # (0-100)% Default is 20 and 10-20 typically gets good results. Lower ratio means more faithfully following input ID (not necessarily better quality).
      sample_method="euler",
      prompt="a man img, retro futurism, retro game art style but extremely beautiful, intricate details, masterpiece, best quality, space-themed, cosmic, celestial, stars, galaxies, nebulas, planets, science fiction, highly detailed",
      negative_prompt="realistic, photo-realistic, worst quality, greyscale, bad anatomy, bad hands, error, text",
      pm_id_images=[
            os.path.join(INPUT_ID_IMAGES_DIR, f)
            for f in os.listdir(INPUT_ID_IMAGES_DIR)
            if f.lower().endswith((".png", ".jpg", ".jpeg", ".bmp"))
      ],
)
```

#### <u>PhotoMaker Version 2</u>

[PhotoMaker Version 2 (PMV2)](https://github.com/TencentARC/PhotoMaker/blob/main/README_pmv2.md) has some key improvements. Unfortunately it has a very heavy dependency which makes running it a bit involved.

Running PMV2 Requires running a python script `face_detect.py` (found here [stable-diffusion.cpp/face_detect.py](https://github.com/leejet/stable-diffusion.cpp/blob/master/face_detect.py)) to obtain `id_embeds` for the given input images.

```bash
python face_detect.py <input_image_dir>
```

An `id_embeds.bin` file will be generated in `input_images_dir`.

**Note: This step only needs to be run once — the resulting `id_embeds` can be reused.**

- Run the same command as in version 1 but replacing `photomaker-v1.safetensors` with `photomaker-v2.safetensors` and pass the `id_embeds.bin` path into the `pm_id_embed_path` parameter.
  Download `photomaker-v2.safetensors` from [bssrdf/PhotoMakerV2](https://huggingface.co/bssrdf/PhotoMakerV2).
- All other parameters from Version 1 remain the same for Version 2.

---

### <u>QWEN Image</u>

Download the weights from the links below:

- Download `Qwen Image`
  - safetensors: https://huggingface.co/Comfy-Org/Qwen-Image_ComfyUI/tree/main/split_files/diffusion_models
  - gguf: https://huggingface.co/QuantStack/Qwen-Image-GGUF/tree/main
- Download `vae`
  - safetensors: https://huggingface.co/Comfy-Org/Qwen-Image_ComfyUI/tree/main/split_files/vae
- Download `qwen_2.5_vl 7b`
  - safetensors: https://huggingface.co/Comfy-Org/Qwen-Image_ComfyUI/tree/main/split_files/text_encoders
  - gguf: https://huggingface.co/mradermacher/Qwen2.5-VL-7B-Instruct-GGUF/tree/main

```python
from stable_diffusion_cpp import StableDiffusion

stable_diffusion = StableDiffusion(
      diffusion_model_path="../models/qwen-image-Q8_0.gguf",
      llm_path="../models/Qwen2.5-VL-7B-Instruct.Q8_0.gguf",
      vae_path="../models/qwen_image_vae.safetensors",
      offload_params_to_cpu=True,
      flow_shift=3,
)

output = stable_diffusion.generate_image(
      prompt='一个穿着"QWEN"标志的T恤的中国美女正拿着黑色的马克笔面相镜头微笑。她身后的玻璃板上手写体写着 “一、Qwen-Image的技术路线： 探索视觉生成基础模型的极限，开创理解与生成一体化的未来。二、Qwen-Image的模型特色：1、复杂文字渲染。支持中英渲染、自动布局； 2、精准图像编辑。支持文字编辑、物体增减、风格变换。三、Qwen-Image的未来愿景：赋能专业内容创作、助力生成式AI发展。”',
      cfg_scale=2.5,
      sample_method='euler',
)
```

#### <u>QWEN Image Edit</u>

Download the weights from the links below:

- Download `Qwen Image Edit`
  - Qwen Image Edit
    - safetensors: https://huggingface.co/Comfy-Org/Qwen-Image-Edit_ComfyUI/tree/main/split_files/diffusion_models
    - gguf: https://huggingface.co/QuantStack/Qwen-Image-Edit-GGUF/tree/main
  - Qwen Image Edit 2509
    - safetensors: https://huggingface.co/Comfy-Org/Qwen-Image-Edit_ComfyUI/tree/main/split_files/diffusion_models
    - gguf: https://huggingface.co/QuantStack/Qwen-Image-Edit-2509-GGUF/tree/main
- Download `vae`
  - safetensors: https://huggingface.co/Comfy-Org/Qwen-Image_ComfyUI/tree/main/split_files/vae
- Download `qwen_2.5_vl 7b`
  - safetensors: https://huggingface.co/Comfy-Org/Qwen-Image_ComfyUI/tree/main/split_files/text_encoders
  - gguf: https://huggingface.co/mradermacher/Qwen2.5-VL-7B-Instruct-GGUF/tree/main

```python
from stable_diffusion_cpp import StableDiffusion

stable_diffusion = StableDiffusion(
      diffusion_model_path="../models/Qwen_Image_Edit-Q8_0.gguf",
      llm_path="../models/Qwen2.5-VL-7B-Instruct.Q8_0.gguf",
      vae_path="../models/qwen_image_vae.safetensors",
      offload_params_to_cpu=True,
      flow_shift=3,
)

output = stable_diffusion.generate_image(
      prompt="make the cat blue",
      ref_images=["input.png"],
      cfg_scale=2.5,
      sample_method='euler',
)
```

---

### <u>Z-Image</u>

Download the weights from the links below:

- Download `Z-Image-Turbo`
  - safetensors: https://huggingface.co/Comfy-Org/z_image_turbo/tree/main/split_files/diffusion_models
  - gguf: https://huggingface.co/leejet/Z-Image-Turbo-GGUF/tree/main
- Download `vae`
  - safetensors: https://huggingface.co/black-forest-labs/FLUX.1-schnell/tree/main
- Download `Qwen3 4b`
  - safetensors: https://huggingface.co/Comfy-Org/z_image_turbo/tree/main/split_files/text_encoders
  - gguf: https://huggingface.co/unsloth/Qwen3-4B-Instruct-2507-GGUF/tree/main

```python
from stable_diffusion_cpp import StableDiffusion

stable_diffusion = StableDiffusion(
      diffusion_model_path="../models/z_image_turbo-Q3_K.gguf",
      llm_path="../models/Qwen3-4B-Instruct-2507-Q4_K_M.gguf",
      vae_path="../models/ae.safetensors",
      offload_params_to_cpu=True,
      diffusion_flash_attn=True,
)

output = stable_diffusion.generate_image(
      prompt="A cinematic, melancholic photograph of a solitary hooded figure walking through a sprawling, rain-slicked metropolis at night. The city lights are a chaotic blur of neon orange and cool blue, reflecting on the wet asphalt. The scene evokes a sense of being a single component in a vast machine. Superimposed over the image in a sleek, modern, slightly glitched font is the philosophical quote: 'THE CITY IS A CIRCUIT BOARD, AND I AM A BROKEN TRANSISTOR.' -- moody, atmospheric, profound, dark academic",
      height=1024,
      width=512,
      cfg_scale=1.0,
)
```

---

### <u>Ovis</u>

Download the weights from the links below:

- Download `Ovis-Image-7B`
  - safetensors: https://huggingface.co/Comfy-Org/Ovis-Image/tree/main/split_files/diffusion_models
  - gguf: https://huggingface.co/leejet/Ovis-Image-7B-GGUF
- Download `vae`
  - safetensors: https://huggingface.co/black-forest-labs/FLUX.1-schnell/tree/main
- Download `Ovis 2.5`
  - safetensors: https://huggingface.co/Comfy-Org/Ovis-Image/tree/main/split_files/text_encoders

```python
from stable_diffusion_cpp import StableDiffusion

stable_diffusion = StableDiffusion(
      diffusion_model_path="../models/ovis_image-Q4_0.gguf",
      llm_path="../models/ovis_2.5.safetensors",
      vae_path="../models/ae.safetensors",
      diffusion_flash_attn=True,
)

output = stable_diffusion.generate_image(
      prompt="a lovely cat",
      cfg_scale=5.0,
)
```

---

### <u>Wan Video Generation</u>

See [stable-diffusion.cpp Wan download weights](https://github.com/leejet/stable-diffusion.cpp/blob/master/docs/wan.md#download-weights) for a complete list of Wan models.

```python
from stable_diffusion_cpp import StableDiffusion

stable_diffusion = StableDiffusion(
      diffusion_model_path="../models/wan2.1_t2v_1.3B_fp16.safetensors", # In place of model_path
      t5xxl_path="../models/umt5-xxl-encoder-Q8_0.gguf",
      vae_path="../models/wan_2.1_vae.safetensors",
      flow_shift=3.0,
      keep_clip_on_cpu=True, # Prevents black images when using some T5 models
)

output = stable_diffusion.generate_video(
      prompt="a cute dog jumping",
      negative_prompt="色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指，画得不好的手部，画得不好的脸部， 畸形的，毁容的，形态畸形的肢体，手指融合，静止不动的画面，杂乱的背景，三条腿，背景人很多，倒着走",
      height=832,
      width=480,
      cfg_scale=6.0,
      sample_method="euler",
      video_frames=33,
) # Output is a list of PIL Images (video frames)
```

As the output is simply a list of images (video frames), you can convert it into a video using any library you prefer. The example below uses `ffmpeg-python`. Alternatively, libraries such **OpenCV** or **MoviePy** can also be used.

> **Note**
>
> - You'll require **Python bindings for FFmpeg**, `python-ffmpeg` (`pip install ffmpeg-python`) in addition to an **FFmpeg installation on your system**, accessible in your PATH. Check with `ffmpeg -version`.

```python
from typing import List
from PIL import Image
import numpy as np
import ffmpeg

def save_video_ffmpeg(frames: List[Image.Image], fps: int, out_path: str) -> None:
      if not frames:
            raise ValueError("No frames provided")

      width, height = frames[0].size

      # Concatenate frames into raw RGB bytes
      raw_bytes = b"".join(np.array(frame.convert("RGB"), dtype=np.uint8).tobytes() for frame in frames)
      (
            ffmpeg.input(
                  "pipe:",
                  format="rawvideo",
                  pix_fmt="rgb24",
                  s=f"{width}x{height}",
                  r=fps,
            )
            .output(
                  out_path,
                  vcodec="libx264",
                  pix_fmt="yuv420p",
                  r=fps,
                  movflags="+faststart",
            )
            .overwrite_output()
            .run(input=raw_bytes)
      )

save_video_ffmpeg(output, fps=16, out_path="output.mp4")
```

#### <u>Wan VACE</u>

Use FFmpeg to extract frames from a video to use as control frames for Wan VACE.

```bash
mkdir assets/frames
ffmpeg -i  assets/test.mp4 -qscale:v 1 -vf fps=8 assets/frames/frame_%04d.jpg
```

```python
output = stable_diffusion.generate_video(
      ...
      # Add control frames for VACE (PIL Images or file paths)
      control_frames=[
            os.path.join('assets/frames', f)
            for f in os.listdir('assets/frames')
            if f.lower().endswith((".png", ".jpg", ".jpeg", ".bmp"))
      ],
)
```

---

### <u>GGUF Model Conversion</u>

You can convert models to GGUF format using the `convert` method.

```python
from stable_diffusion_cpp import StableDiffusion

stable_diffusion = StableDiffusion()

stable_diffusion.convert(
      input_path="../models/v1-5-pruned-emaonly.safetensors",
      output_path="new_model.gguf",
      output_type="q8_0",
)
```

---

### <u>Listing LoRA apply modes, GGML model/prediction/RNG types, sample/preview methods and schedulers</u>

Access the LoRA apply modes, GGML model/prediction/RNG types, sample/preview methods and schedulers via the following maps:

```python
from stable_diffusion_cpp import GGML_TYPE_MAP, RNG_TYPE_MAP, SCHEDULER_MAP, SAMPLE_METHOD_MAP, PREDICTION_MAP, PREVIEW_MAP, LORA_APPLY_MODE_MAP, SD_CACHE_MODE_MAP

print("GGML model types:", list(GGML_TYPE_MAP))
print("RNG types:", list(RNG_TYPE_MAP))
print("Schedulers:", list(SCHEDULER_MAP))
print("Sample methods:", list(SAMPLE_METHOD_MAP))
print("Prediction types:", list(PREDICTION_MAP))
print("Preview methods:", list(PREVIEW_MAP))
print("LoRA apply modes:", list(LORA_APPLY_MODE_MAP))
print("SD cache modes:", list(SD_CACHE_MODE_MAP))
```

---

### <u>Other High-level API Examples</u>

Other examples for the high-level API (such as upscaling and model conversion) can be found in the [tests](tests) directory.

## Low-level API

The low-level API is a direct [`ctypes`](https://docs.python.org/3/library/ctypes.html) binding to the C API provided by `stable-diffusion.cpp`.
The entire low-level API can be found in [stable_diffusion_cpp/stable_diffusion_cpp.py](https://github.com/william-murray1204/stable-diffusion-cpp-python/blob/main/stable_diffusion_cpp/stable_diffusion_cpp.py) and directly mirrors the C API in [stable-diffusion.h](https://github.com/leejet/stable-diffusion.cpp/blob/master/stable-diffusion.h).

Below is a short example demonstrating low-level API usage:

```python
import stable_diffusion_cpp as sd_cpp
import ctypes
from PIL import Image

img = Image.open("path/to/image.png")
img_bytes = img.tobytes()

c_image = sd_cpp.sd_image_t(
      width=img.width,
      height=img.height,
      channel=channel,
      data=ctypes.cast(
            (ctypes.c_byte * len(img_bytes))(*img_bytes),
            ctypes.POINTER(ctypes.c_uint8),
      ),
) # Create a new C sd_image_t
```

## Development

To get started, clone the repository and install the package in editable / development mode.

```bash
git clone --recurse-submodules https://github.com/william-murray1204/stable-diffusion-cpp-python.git
cd stable-diffusion-cpp-python

# Upgrade pip (required for editable mode)
pip install --upgrade pip

# Install with pip
pip install -e .
```

Now you can make changes to the code within the `stable_diffusion_cpp` directory and test them in your python environment.

## References

- [stable-diffusion.cpp](https://github.com/leejet/stable-diffusion.cpp)
- [llama-cpp-python](https://github.com/abetlen/llama-cpp-python)
- [llama.cpp](https://github.com/ggml-org/llama.cpp)
- [whisper-cpp-python](https://github.com/carloscdias/whisper-cpp-python)
- [Golang stable-diffusion](https://github.com/seasonjs/stable-diffusion)
- [StableDiffusion.NET](https://github.com/DarthAffe/StableDiffusion.NET)

## License

This project is licensed under the terms of the MIT license. See [LICENSE](LICENSE) for details.
