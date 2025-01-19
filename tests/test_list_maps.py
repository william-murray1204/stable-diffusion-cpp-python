from stable_diffusion_cpp import GGML_TYPE_MAP, RNG_TYPE_MAP, SCHEDULE_MAP, SAMPLE_METHOD_MAP

print("GGML model types:", list(GGML_TYPE_MAP))
print("RNG types:", list(RNG_TYPE_MAP))
print("Schedulers:", list(SCHEDULE_MAP))
print("Sample methods:", list(SAMPLE_METHOD_MAP))
