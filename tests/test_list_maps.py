from conftest import OUTPUT_DIR

from stable_diffusion_cpp import (
    PREVIEW_MAP,
    RNG_TYPE_MAP,
    GGML_TYPE_MAP,
    SCHEDULER_MAP,
    PREDICTION_MAP,
    SAMPLE_METHOD_MAP,
    LORA_APPLY_MODE_MAP,
    SD_CACHE_MODE_MAP,
)


def test_list_maps():
    maps = {
        "GGML model types": GGML_TYPE_MAP,
        "RNG types": RNG_TYPE_MAP,
        "Schedulers": SCHEDULER_MAP,
        "Sample methods": SAMPLE_METHOD_MAP,
        "Prediction types": PREDICTION_MAP,
        "Preview methods": PREVIEW_MAP,
        "LoRA apply modes": LORA_APPLY_MODE_MAP,
        "SD cache modes": SD_CACHE_MODE_MAP,
    }

    with open(f"{OUTPUT_DIR}/list_maps.txt", "w") as f:
        for name, mapping in maps.items():
            items = list(mapping)
            print(f"{name}: {items}")
            f.write(f"{name}: {items}\n")
