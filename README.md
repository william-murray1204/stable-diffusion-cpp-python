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
CMAKE_ARGS="-DGGML_OPENBLAS=ON" \
  pip install stable-diffusion-cpp-python
```

```powershell
# Windows
$env:CMAKE_ARGS = "-DGGML_OPENBLAS=ON"
pip install stable-diffusion-cpp-python
```

</details>

<details>
<summary>CLI / requirements.txt</summary>

They can also be set via `pip install -C / --config-settings` command and saved to a `requirements.txt` file:

```bash
pip install --upgrade pip # ensure pip is up to date
pip install stable-diffusion-cpp-python \
  -C cmake.args="-DGGML_OPENBLAS=ON"
```

```txt
# requirements.txt

stable-diffusion-cpp-python -C cmake.args="-DGGML_OPENBLAS=ON"
```

</details>

### Supported Backends

Below are some common backends, their build commands and any additional environment variables required.

<details>
<summary>Using OpenBLAS (CPU)</summary>

```bash
CMAKE_ARGS="-DGGML_OPENBLAS=ON" pip install stable-diffusion-cpp-python
```

</details>

<details>
<summary>Using cuBLAS (CUDA)</summary>

This provides BLAS acceleration using the CUDA cores of your Nvidia GPU. Make sure to have the CUDA toolkit installed. You can download it from your Linux distro's package manager (e.g. `apt install nvidia-cuda-toolkit`) or from here: [CUDA Toolkit](https://developer.nvidia.com/cuda-downloads). Recommended to have at least 4 GB of VRAM.

```bash
CMAKE_ARGS="-DSD_CUBLAS=on" pip install stable-diffusion-cpp-python
```

</details>

<details>
<summary>Using hipBLAS (ROCm)</summary>

This provides BLAS acceleration using the ROCm cores of your AMD GPU. Make sure to have the ROCm toolkit installed.
Windows Users Refer to [docs/hipBLAS_on_Windows.md](docs%2FhipBLAS_on_Windows.md) for a comprehensive guide.

```bash
CMAKE_ARGS="-G Ninja -DCMAKE_C_COMPILER=clang -DCMAKE_CXX_COMPILER=clang++ -DSD_HIPBLAS=ON -DCMAKE_BUILD_TYPE=Release -DAMDGPU_TARGETS=gfx1101" pip install stable-diffusion-cpp-python
```

</details>

<details>
<summary>Using Metal</summary>

Using Metal makes the computation run on the GPU. Currently, there are some issues with Metal when performing operations on very large matrices, making it highly inefficient at the moment. Performance improvements are expected in the near future.

```bash
CMAKE_ARGS="-DSD_METAL=ON" pip install stable-diffusion-cpp-python
```

</details>

<details>
<summary>Using Flash Attention</summary>

Enabling flash attention reduces memory usage by at least 400 MB. At the moment, it is not supported when CUBLAS is enabled because the kernel implementation is missing.

```bash
CMAKE_ARGS="-DSD_FLASH_ATTN=ON" pip install stable-diffusion-cpp-python
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
      wtype="default", # Weight type (default: automatically determines the weight type of the model file)
      progress_callback=callback,
)
output = stable_diffusion.txt_to_img(
      "a lovely cat", # Prompt
      # seed=1337, # Uncomment to set a specific seed
)
```

#### With LoRA

You can specify the directory where the lora weights are stored via `lora_model_dir`. If not specified, the default is the current working directory.

- LoRA is specified via prompt, just like [stable-diffusion-webui](https://github.com/AUTOMATIC1111/stable-diffusion-webui/wiki/Features#lora). (e.g. `<lora:marblesh:1>`)
- LoRAs will not work when using quantized models. You must instead use a full precision `.safetensors` model.

Here's a simple example:

```python
from stable_diffusion_cpp import StableDiffusion

stable_diffusion = StableDiffusion(
      model_path="../models/v1-5-pruned-emaonly.safetensors",
      lora_model_dir="../models/",
)
output = stable_diffusion.txt_to_img(
      "a lovely cat<lora:marblesh:1>", # Prompt
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
    diffusion_model_path="../models/flux1-schnell-q3_k.gguf", # in place of model_path
    clip_l_path="../models/t5xxl_q8_0.gguf",
    t5xxl_path="../models/clip_l-q8_0.gguf",
    vae_path="../models/ae-f16.gguf",
)
output = stable_diffusion.flux_img(
      prompt="a lovely cat holding a sign says 'flux.cpp'",
      sample_steps=4,
      cfg_scale=1.0, # a cfg_scale of 1 is recommended
      sample_method="euler", # euler is recommended for FLUX
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

sd_cpp.free_image(c_image)
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
