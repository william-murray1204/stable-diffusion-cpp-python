# 🖼️ Python Bindings for [`stable-diffusion.cpp`](https://github.com/leejet/stable-diffusion.cpp)

Simple Python bindings for **@leejet's** [`stable-diffusion.cpp`](https://github.com/leejet/stable-diffusion.cpp) library.
This package provides:

- Low-level access to C API via `ctypes` interface.
- High-level Python API for stable diffusion image generation.

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

If this fails, add `--verbose` to the `pip install` see the full cmake build log.

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
CMAKE_ARGS="-G "Ninja" -DCMAKE_C_COMPILER=clang -DCMAKE_CXX_COMPILER=clang++ -DSD_HIPBLAS=ON -DCMAKE_BUILD_TYPE=Release -DAMDGPU_TARGETS=gfx1100" pip install stable-diffusion-cpp-python
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

```python
>>> from stable_diffusion_cpp import StableDiffusion
>>> stable_diffusion = StableDiffusion(
      model_path="../models/v1-5-pruned-emaonly.safetensors",
      # seed=1337, # Uncomment to set a specific seed
)
>>> output = stable_diffusion.txt_to_img(
      "a lovely cat", # Prompt
)
```

## Low-level API

The low-level API is a direct [`ctypes`](https://docs.python.org/3/library/ctypes.html) binding to the C API provided by `stable-diffusion.cpp`.
The entire low-level API can be found in [stable_diffusion_cpp/stable_diffusion_cpp.py](https://github.com/william-murray1204/stable-diffusion-cpp-python/blob/main/stable_diffusion_cpp/stable_diffusion_cpp.py) and directly mirrors the C API in [stable-diffusion.h](https://github.com/leejet/stable-diffusion.cpp/blob/master/stable-diffusion.h).

Below is a short example demonstrating how to use the low-level API:

```python
>>> import stable_diffusion_cpp as sd_cpp
>>> sd_ctx = sd_cpp.new_sd_ctx(<PARAMS>)
>>> sd_cpp.free_sd_ctx(sd_ctx)
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

# to clear the local build cache
make clean
```

## To Do

- [ ] test get num cores and other info low level api funcs
- [ ] Check seed and n_gpu_layers and thread can be -1 and work properly
- [ ] get img2img working
- [ ] get img2vid working
- [ ] Add proper error handling
- [ ] deploy to PyPI and test installation (CI/CD pipeline github actions)
- [ ] get GPU acceleration
- [ ] Get tests working
- [ ] Add examples to exampels folder
- [ ] Properly intergaret callback for progress
- [ ] get Loras working

## References

- [stable-diffusion.cpp](https://github.com/leejet/stable-diffusion.cpp)
- [stable-diffusion-cpp-python](https://github.com/abetlen/stable-diffusion-cpp-python)
- [llama.cpp](https://github.com/ggerganov/llama.cpp)
- [whisper-cpp-python](https://github.com/carloscdias/whisper-cpp-python)
- [Golang stable-diffusion](https://github.com/seasonjs/stable-diffusion)
- [StableDiffusion.NET](https://github.com/DarthAffe/StableDiffusion.NET)

## License

This project is licensed under the terms of the MIT license. See [LICENSE](LICENSE) for details.
