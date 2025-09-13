from conftest import OUTPUT_DIR

from stable_diffusion_cpp import (
    RNG_TYPE_MAP,
    GGML_TYPE_MAP,
    SCHEDULER_MAP,
    SAMPLE_METHOD_MAP,
)


def test_list_maps():
    maps = {
        "GGML model types": GGML_TYPE_MAP,
        "RNG types": RNG_TYPE_MAP,
        "Schedulers": SCHEDULER_MAP,
        "Sample methods": SAMPLE_METHOD_MAP,
    }

    with open(f"{OUTPUT_DIR}/list_maps.txt", "w") as f:
        for name, mapping in maps.items():
            items = list(mapping)
            print(f"{name}: {items}")
            f.write(f"{name}: {items}\n")
