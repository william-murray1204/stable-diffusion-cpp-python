# 🖼️ Python Bindings for [`stable-diffusion.cpp`](https://github.com/leejet/stable-diffusion.cpp)

Simple Python bindings for **@leejet's** [`stable-diffusion.cpp`](https://github.com/leejet/stable-diffusion.cpp) library.

[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](https://opensource.org/licenses/MIT)
[![PyPi version](https://badgen.net/pypi/v/pywhispercpp)](https://pypi.org/project/stable-diffusion-cpp-python/)
[![Downloads](https://static.pepy.tech/badge/stable-diffusion-cpp-python)](https://pepy.tech/project/stable-diffusion-cpp-python)

This package provides:

- Low-level access to C API via `ctypes` interface.
- High-level Python API for Stable Diffusion and FLUX image generation.

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
CMAKE_ARGS="-DSD_CUBLAS=ON" pip install stable-diffusion-cpp-python
```

```powershell
# Windows
$env:CMAKE_ARGS="-DSD_CUBLAS=ON"
pip install stable-diffusion-cpp-python
```

</details>

<details>
<summary>CLI / requirements.txt</summary>

They can also be set via `pip install -C / --config-settings` command and saved to a `requirements.txt` file:

```bash
pip install --upgrade pip # ensure pip is up to date
pip install stable-diffusion-cpp-python -C cmake.args="-DSD_CUBLAS=ON"
```

```txt
# requirements.txt

stable-diffusion-cpp-python -C cmake.args="-DSD_CUBLAS=ON"
```

</details>

### Supported Backends

Below are some common backends, their build commands and any additional environment variables required.

<!-- CUBLAS -->
<details>
<summary>Using CUBLAS (CUDA)</summary>

This provides BLAS acceleration using the CUDA cores of your Nvidia GPU. Make sure you have the CUDA toolkit installed. You can download it from your Linux distro's package manager (e.g. `apt install nvidia-cuda-toolkit`) or from here: [CUDA Toolkit](https://developer.nvidia.com/cuda-downloads). You can check your installed CUDA toolkit version by running `nvcc --version`.

- It is recommended you have at least 4 GB of VRAM.

```bash
CMAKE_ARGS="-DSD_CUBLAS=ON" pip install stable-diffusion-cpp-python
```

</details>

<!-- HIPBLAS -->
<details>
<summary>Using HIPBLAS (ROCm)</summary>

This provides BLAS acceleration using the ROCm cores of your AMD GPU. Make sure you have the ROCm toolkit installed and that you replace the `-DAMDGPU_TARGETS=` value with that of your GPU architecture.
Windows users refer to [docs/hipBLAS_on_Windows.md](docs%2FhipBLAS_on_Windows.md) for a comprehensive guide and troubleshooting tips.

```bash
CMAKE_ARGS="-G Ninja -DCMAKE_C_COMPILER=clang -DCMAKE_CXX_COMPILER=clang++ -DSD_HIPBLAS=ON -DCMAKE_BUILD_TYPE=Release -DAMDGPU_TARGETS=gfx1101" pip install stable-diffusion-cpp-python
```

</details>

<!-- Metal -->
<details>
<summary>Using Metal</summary>

Using Metal makes the computation run on the GPU. Currently, there are some issues with Metal when performing operations on very large matrices, making it highly inefficient at the moment. Performance improvements are expected in the near future.

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

Using SYCL makes the computation run on the Intel GPU. Please make sure you have installed the related driver and [Intel® oneAPI Base toolkit](https://www.intel.com/content/www/us/en/developer/tools/oneapi/base-toolkit.html) before start. More details and steps can refer to [llama.cpp SYCL backend](https://github.com/ggerganov/llama.cpp/blob/master/docs/backend/SYCL.md#linux).

```bash
# Export relevant ENV variables
source /opt/intel/oneapi/setvars.sh

# Option 1: Use FP32 (recommended for better performance in most cases)
CMAKE_ARGS="-DSD_SYCL=ON -DCMAKE_C_COMPILER=icx -DCMAKE_CXX_COMPILER=icpx" pip install stable-diffusion-cpp-python

# Option 2: Use FP16
CMAKE_ARGS="-DSD_SYCL=ON -DCMAKE_C_COMPILER=icx -DCMAKE_CXX_COMPILER=icpx -DGGML_SYCL_F16=ON" pip install stable-diffusion-cpp-python
```

</details>

<!-- Flash Attention -->
<details>
<summary>Using Flash Attention</summary>

Enabling flash attention reduces memory usage by at least 400 MB. At the moment, it is not supported when CUBLAS is enabled because the kernel implementation is missing.

```bash
CMAKE_ARGS="-DSD_FLASH_ATTN=ON" pip install stable-diffusion-cpp-python
```

</details>

<!-- OpenBLAS -->
<details>
<summary>Using OpenBLAS</summary>

```bash
CMAKE_ARGS="-DGGML_OPENBLAS=ON" pip install stable-diffusion-cpp-python
```

</details>

### Upgrading and Reinstalling

To upgrade and rebuild `stable-diffusion-cpp-python` add `--upgrade --force-reinstall --no-cache-dir` flags to the `pip install` command to ensure the package is rebuilt from source.

## High-level API

The high-level API provides a simple managed interface through the `StableDiffusion` class.

Below is a short example demonstrating how to use the high-level API to generate a simple image:

### Text to Image

```python
from stable_diffusion_cpp import StableDiffusion

def callback(step: int, steps: int, time: float):
    print("Completed step: {} of {}".format(step, steps))

stable_diffusion = StableDiffusion(
      model_path="../models/v1-5-pruned-emaonly.safetensors",
      wtype="default", # Weight type (default: automatically determines weight type of model file)
)
output = stable_diffusion.txt_to_img(
      prompt="a lovely cat",
      width=512, # Must be a multiple of 64
      height=512, # Must be a multiple of 64
      progress_callback=callback,
      # seed=1337, # Uncomment to set a specific seed
)
output[0].save("output.png") # Image returned as list of PIL Images
```

#### With LoRA (Stable Diffusion)

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
output = stable_diffusion.txt_to_img(
      prompt="a lovely cat<lora:marblesh:1>",
)
```

- The `lora_model_dir` argument is used in the same way for FLUX image generation.

### FLUX Image Generation

FLUX models should be run using the same implementation as the [stable-diffusion.cpp FLUX documentation](https://github.com/leejet/stable-diffusion.cpp/blob/master/docs/flux.md) where the `diffusion_model_path` argument is used in place of the `model_path`. The `clip_l_path`, `t5xxl_path`, and `vae_path` arguments are also required for inference to function.

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
)
output = stable_diffusion.txt_to_img(
      prompt="a lovely cat holding a sign says 'flux.cpp'",
      sample_steps=4,
      cfg_scale=1.0, # a cfg_scale of 1 is recommended for FLUX
      sample_method="euler", # euler is recommended for FLUX
)
```

#### With LoRA (FLUX)

LoRAs can be used with FLUX models in the same way as Stable Diffusion models ([as shown above](#with-lora-stable-diffusion)).

Note that:

- It is recommended to use LoRA with naming formats compatible with ComfyUI.
- Only the Flux-dev q8_0 will work with LoRAs.
- You can download FLUX LoRA models from https://huggingface.co/XLabs-AI/flux-lora-collection/tree/main (you must use a comfy converted version!!!).

### SD3.5 Image Generation

Download the weights from the links below:

- Download sd3.5_large from https://huggingface.co/stabilityai/stable-diffusion-3.5-large/blob/main/sd3.5_large.safetensors
- Download clip_g from https://huggingface.co/Comfy-Org/stable-diffusion-3.5-fp8/blob/main/text_encoders/clip_g.safetensors
- Download clip_l from https://huggingface.co/Comfy-Org/stable-diffusion-3.5-fp8/blob/main/text_encoders/clip_l.safetensors
- Download t5xxl from https://huggingface.co/Comfy-Org/stable-diffusion-3.5-fp8/blob/main/text_encoders/t5xxl_fp16.safetensors

```python
from stable_diffusion_cpp import StableDiffusion

stable_diffusion = StableDiffusion(
    model_path="../models/sd3.5_large.safetensors",
    clip_l_path="../models/clip_l.safetensors",
    clip_g_path="../models/clip_g.safetensors",
    t5xxl_path="../models/t5xxl_fp16.safetensors",
)
output = stable_diffusion.txt_to_img(
      prompt="a lovely cat holding a sign says 'Stable diffusion 3.5 Large'",
      height=1024,
      width=1024,
      cfg_scale=4.5,
      sample_method="euler",
)
```

### Other High-level API Examples

Other examples for the high-level API (such as image to image, upscaling and model conversion) can be found in the [tests](tests) directory.

## Low-level API

The low-level API is a direct [`ctypes`](https://docs.python.org/3/library/ctypes.html) binding to the C API provided by `stable-diffusion.cpp`.
The entire low-level API can be found in [stable_diffusion_cpp/stable_diffusion_cpp.py](https://github.com/william-murray1204/stable-diffusion-cpp-python/blob/main/stable_diffusion_cpp/stable_diffusion_cpp.py) and directly mirrors the C API in [stable-diffusion.h](https://github.com/leejet/stable-diffusion.cpp/blob/master/stable-diffusion.h).

Below is a short example demonstrating how to use the low-level API:

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

img = sd_cpp.upscale(
      self.upscaler,
      image_bytes,
      upscale_factor,
) # Upscale the image
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

### Cleanup

To clear the cache.

```bash
make clean
```

## References

- [stable-diffusion.cpp](https://github.com/leejet/stable-diffusion.cpp)
- [llama-cpp-python](https://github.com/abetlen/llama-cpp-python)
- [llama.cpp](https://github.com/ggerganov/llama.cpp)
- [whisper-cpp-python](https://github.com/carloscdias/whisper-cpp-python)
- [Golang stable-diffusion](https://github.com/seasonjs/stable-diffusion)
- [StableDiffusion.NET](https://github.com/DarthAffe/StableDiffusion.NET)

## License

This project is licensed under the terms of the MIT license. See [LICENSE](LICENSE) for details.
