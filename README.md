# stable-diffusion-cpp-python

stable-diffusion.cpp bindings for python

## Development

To get started, clone the repository and install the package in editable / development mode.

```bash
git clone --recurse-submodules https://github.com/william-murray1204/stable-diffusion-cpp-python.git
cd stable-diffusion-cpp-python

# Upgrade pip (required for editable mode)
pip install --upgrade pip

# Set the CMAKE_ARGS environment variable
$env:CMAKE_ARGS="-D SD_BUILD_SHARED_LIBS=ON"


# Install with pip
pip install -e .

# Or (Full reinstallation)
pip install -e . --upgrade --force-reinstall --no-cache-dir

# to clear the local build cache
make clean
```
