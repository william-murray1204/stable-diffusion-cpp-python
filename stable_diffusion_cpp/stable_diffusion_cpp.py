from __future__ import annotations

import os
import sys
import ctypes
import pathlib
import functools
from enum import IntEnum
from typing import (
    TYPE_CHECKING,
    Any,
    List,
    Union,
    Generic,
    NewType,
    TypeVar,
    Callable,
    Optional,
)

from typing_extensions import TypeAlias


# Load the library
def _load_shared_library(lib_base_name: str):
    # Construct the paths to the possible shared library names
    _base_path = pathlib.Path(os.path.abspath(os.path.dirname(__file__))) / "lib"
    # Searching for the library in the current directory under the name "libstable-diffusion" (default name
    # for stable-diffusion-cpp) and "stable-diffusion" (default name for this repo)
    _lib_paths: List[pathlib.Path] = []

    # Determine the file extension based on the platform
    if sys.platform.startswith("linux"):
        _lib_paths += [
            _base_path / f"lib{lib_base_name}.so",
        ]
    elif sys.platform == "darwin":
        _lib_paths += [
            _base_path / f"lib{lib_base_name}.so",
            _base_path / f"lib{lib_base_name}.dylib",
        ]
    elif sys.platform == "win32":
        _lib_paths += [
            _base_path / f"{lib_base_name}.dll",
            _base_path / f"lib{lib_base_name}.dll",
        ]
    else:
        raise RuntimeError("Unsupported platform")

    if "STABLE_DIFFUSION_CPP_LIB" in os.environ:
        lib_base_name = os.environ["STABLE_DIFFUSION_CPP_LIB"]
        _lib = pathlib.Path(lib_base_name)
        _base_path = _lib.parent.resolve()
        _lib_paths = [_lib.resolve()]

    cdll_args = dict()  # type: ignore
    # Add the library directory to the DLL search path on Windows (if needed)
    if sys.platform == "win32" and sys.version_info >= (3, 8):
        os.add_dll_directory(str(_base_path))
        if "CUDA_PATH" in os.environ:
            os.add_dll_directory(os.path.join(os.environ["CUDA_PATH"], "bin"))
            os.add_dll_directory(os.path.join(os.environ["CUDA_PATH"], "lib"))
        if "HIP_PATH" in os.environ:
            os.add_dll_directory(os.path.join(os.environ["HIP_PATH"], "bin"))
            os.add_dll_directory(os.path.join(os.environ["HIP_PATH"], "lib"))
        cdll_args["winmode"] = ctypes.RTLD_GLOBAL

    # Try to load the shared library, handling potential errors
    for _lib_path in _lib_paths:
        if _lib_path.exists():
            try:
                return ctypes.CDLL(str(_lib_path), **cdll_args)  # type: ignore
            except Exception as e:
                raise RuntimeError(f"Failed to load shared library '{_lib_path}': {e}")

    raise FileNotFoundError(f"Shared library with base name '{lib_base_name}' not found")


# Specify the base name of the shared library to load
_lib_base_name = "stable-diffusion"

# Load the library
_lib = _load_shared_library(_lib_base_name)

# ctypes sane type hint helpers
#
# - Generic Pointer and Array types
# - PointerOrRef type with a type hinted byref function
#
# NOTE: Only use these for static type checking not for runtime checks
# no good will come of that

if TYPE_CHECKING:
    CtypesCData = TypeVar("CtypesCData", bound=ctypes._CData)  # type: ignore

    CtypesArray: TypeAlias = ctypes.Array[CtypesCData]  # type: ignore

    CtypesPointer: TypeAlias = ctypes._Pointer[CtypesCData]  # type: ignore

    CtypesVoidPointer: TypeAlias = ctypes.c_void_p

    class CtypesRef(Generic[CtypesCData]):
        pass

    CtypesPointerOrRef: TypeAlias = Union[CtypesPointer[CtypesCData], CtypesRef[CtypesCData]]

    CtypesFuncPointer: TypeAlias = ctypes._FuncPointer  # type: ignore

F = TypeVar("F", bound=Callable[..., Any])


def ctypes_function_for_shared_library(lib: ctypes.CDLL):
    def ctypes_function(name: str, argtypes: List[Any], restype: Any, enabled: bool = True):
        def decorator(f: F) -> F:
            if enabled:
                func = getattr(lib, name)
                func.argtypes = argtypes
                func.restype = restype
                functools.wraps(f)(func)
                return func
            else:
                return f

        return decorator

    return ctypes_function


ctypes_function = ctypes_function_for_shared_library(_lib)


def byref(obj: CtypesCData, offset: Optional[int] = None) -> CtypesRef[CtypesCData]:
    """Type-annotated version of ctypes.byref"""
    ...


byref = ctypes.byref  # type: ignore


# // Abort callback
# // If not NULL, called before ggml computation
# // If it returns true, the computation is aborted
# typedef bool (*ggml_abort_callback)(void * data);
ggml_abort_callback = ctypes.CFUNCTYPE(ctypes.c_bool, ctypes.c_void_p)


################################################
# stable-diffusion.h bindings
################################################


# enum rng_type_t {
#     STD_DEFAULT_RNG,
#     CUDA_RNG,
#     CPU_RNG,
#     RNG_TYPE_COUNT
# };
class RNGType(IntEnum):
    STD_DEFAULT_RNG = 0
    CUDA_RNG = 1
    CPU_RNG = 2
    RNG_TYPE_COUNT = 3


# enum sample_method_t {
#     EULER_SAMPLE_METHOD,
#     EULER_A_SAMPLE_METHOD,
#     HEUN_SAMPLE_METHOD,
#     DPM2_SAMPLE_METHOD,
#     DPMPP2S_A_SAMPLE_METHOD,
#     DPMPP2M_SAMPLE_METHOD,
#     DPMPP2Mv2_SAMPLE_METHOD,
#     IPNDM_SAMPLE_METHOD,
#     IPNDM_V_SAMPLE_METHOD,
#     LCM_SAMPLE_METHOD,
#     DDIM_TRAILING_SAMPLE_METHOD,
#     TCD_SAMPLE_METHOD,
#     SAMPLE_METHOD_COUNT
# };
class SampleMethod(IntEnum):
    EULER_SAMPLE_METHOD = 0
    EULER_A_SAMPLE_METHOD = 1
    HEUN_SAMPLE_METHOD = 2
    DPM2_SAMPLE_METHOD = 3
    DPMPP2S_A_SAMPLE_METHOD = 4
    DPMPP2M_SAMPLE_METHOD = 5
    DPMPP2Mv2_SAMPLE_METHOD = 6
    IPNDM_SAMPLE_METHOD = 7
    IPNDM_V_SAMPLE_METHOD = 8
    LCM_SAMPLE_METHOD = 9
    DDIM_TRAILING_SAMPLE_METHOD = 10
    TCD_SAMPLE_METHOD = 11
    SAMPLE_METHOD_COUNT = 12


# enum scheduler_t {
#     DISCRETE_SCHEDULER,
#     KARRAS_SCHEDULER,
#     EXPONENTIAL_SCHEDULER,
#     AYS_SCHEDULER,
#     GITS_SCHEDULER,
#     SGM_UNIFORM_SCHEDULER,
#     SIMPLE_SCHEDULER,
#     SMOOTHSTEP_SCHEDULER,
#     KL_OPTIMAL_SCHEDULER,
#     LCM_SCHEDULER,
#     SCHEDULER_COUNT
# };
class Scheduler(IntEnum):
    DISCRETE_SCHEDULER = 0
    KARRAS_SCHEDULER = 1
    EXPONENTIAL_SCHEDULER = 2
    AYS_SCHEDULER = 3
    GITS_SCHEDULER = 4
    SGM_UNIFORM_SCHEDULER = 5
    SIMPLE_SCHEDULER = 6
    SMOOTHSTEP_SCHEDULER = 7
    KL_OPTIMAL_SCHEDULER = 8
    LCM_SCHEDULER = 9
    SCHEDULER_COUNT = 10


# enum prediction_t {
#     EPS_PRED,
#     V_PRED,
#     EDM_V_PRED,
#     FLOW_PRED,
#     FLUX_FLOW_PRED,
#     FLUX2_FLOW_PRED,
#     PREDICTION_COUNT
# };
class Prediction(IntEnum):
    EPS_PRED = 0
    V_PRED = 1
    EDM_V_PRED = 2
    FLOW_PRED = 3
    FLUX_FLOW_PRED = 4
    FLUX2_FLOW_PRED = 5
    PREDICTION_COUNT = 6


# // same as enum ggml_type
# enum sd_type_t {
#     SD_TYPE_F32  = 0,
#     SD_TYPE_F16  = 1,
#     SD_TYPE_Q4_0 = 2,
#     SD_TYPE_Q4_1 = 3,
#     // SD_TYPE_Q4_2 = 4, support has been removed
#     // SD_TYPE_Q4_3 = 5, support has been removed
#     SD_TYPE_Q5_0    = 6,
#     SD_TYPE_Q5_1    = 7,
#     SD_TYPE_Q8_0    = 8,
#     SD_TYPE_Q8_1    = 9,
#     SD_TYPE_Q2_K    = 10,
#     SD_TYPE_Q3_K    = 11,
#     SD_TYPE_Q4_K    = 12,
#     SD_TYPE_Q5_K    = 13,
#     SD_TYPE_Q6_K    = 14,
#     SD_TYPE_Q8_K    = 15,
#     SD_TYPE_IQ2_XXS = 16,
#     SD_TYPE_IQ2_XS  = 17,
#     SD_TYPE_IQ3_XXS = 18,
#     SD_TYPE_IQ1_S   = 19,
#     SD_TYPE_IQ4_NL  = 20,
#     SD_TYPE_IQ3_S   = 21,
#     SD_TYPE_IQ2_S   = 22,
#     SD_TYPE_IQ4_XS  = 23,
#     SD_TYPE_I8      = 24,
#     SD_TYPE_I16     = 25,
#     SD_TYPE_I32     = 26,
#     SD_TYPE_I64     = 27,
#     SD_TYPE_F64     = 28,
#     SD_TYPE_IQ1_M   = 29,
#     SD_TYPE_BF16    = 30,
#     // SD_TYPE_Q4_0_4_4 = 31, support has been removed from gguf files
#     // SD_TYPE_Q4_0_4_8 = 32,
#     // SD_TYPE_Q4_0_8_8 = 33,
#     SD_TYPE_TQ1_0 = 34,
#     SD_TYPE_TQ2_0 = 35,
#     // SD_TYPE_IQ4_NL_4_4 = 36,
#     // SD_TYPE_IQ4_NL_4_8 = 37,
#     // SD_TYPE_IQ4_NL_8_8 = 38,
#     SD_TYPE_MXFP4 = 39,  // MXFP4 (1 block)
#     SD_TYPE_COUNT = 40,
# };
class GGMLType(IntEnum):
    SD_TYPE_F32 = 0
    SD_TYPE_F16 = 1
    SD_TYPE_Q4_0 = 2
    SD_TYPE_Q4_1 = 3
    # SD_TYPE_Q4_2 = 4 support has been removed
    # SD_TYPE_Q4_3 = 5 support has been removed
    SD_TYPE_Q5_0 = 6
    SD_TYPE_Q5_1 = 7
    SD_TYPE_Q8_0 = 8
    SD_TYPE_Q8_1 = 9
    # // k-quantizations
    SD_TYPE_Q2_K = 10
    SD_TYPE_Q3_K = 11
    SD_TYPE_Q4_K = 12
    SD_TYPE_Q5_K = 13
    SD_TYPE_Q6_K = 14
    SD_TYPE_Q8_K = 15
    SD_TYPE_IQ2_XXS = 16
    SD_TYPE_IQ2_XS = 17
    SD_TYPE_IQ3_XXS = 18
    SD_TYPE_IQ1_S = 19
    SD_TYPE_IQ4_NL = 20
    SD_TYPE_IQ3_S = 21
    SD_TYPE_IQ2_S = 22
    SD_TYPE_IQ4_XS = 23
    SD_TYPE_I8 = 24
    SD_TYPE_I16 = 25
    SD_TYPE_I32 = 26
    SD_TYPE_I64 = 27
    SD_TYPE_F64 = 28
    SD_TYPE_IQ1_M = 29
    SD_TYPE_BF16 = 30
    # SD_TYPE_Q4_0_4_4 = 31 # support has been removed from gguf files
    # SD_TYPE_Q4_0_4_8 = 32
    # SD_TYPE_Q4_0_8_8 = 33
    SD_TYPE_TQ1_0 = 34
    SD_TYPE_TQ2_0 = 35
    # SD_TYPE_IQ4_NL_4_4 = 36,
    # SD_TYPE_IQ4_NL_4_8 = 37,
    # SD_TYPE_IQ4_NL_8_8 = 38,
    SD_TYPE_MXFP4 = 39  # MXFP4 (1 block)
    SD_TYPE_COUNT = 40


# enum preview_t {
#     PREVIEW_NONE,
#     PREVIEW_PROJ,
#     PREVIEW_TAE,
#     PREVIEW_VAE,
#     PREVIEW_COUNT
# };
class Preview(IntEnum):
    PREVIEW_NONE = 0
    PREVIEW_PROJ = 1
    PREVIEW_TAE = 2
    PREVIEW_VAE = 3
    PREVIEW_COUNT = 4


# enum lora_apply_mode_t {
#     LORA_APPLY_AUTO,
#     LORA_APPLY_IMMEDIATELY,
#     LORA_APPLY_AT_RUNTIME,
#     LORA_APPLY_MODE_COUNT,
# };
class LoraApplyMode(IntEnum):
    LORA_APPLY_AUTO = 0
    LORA_APPLY_IMMEDIATELY = 1
    LORA_APPLY_AT_RUNTIME = 2
    LORA_APPLY_MODE_COUNT = 3


# enum sd_cache_mode_t {
#     SD_CACHE_DISABLED = 0,
#     SD_CACHE_EASYCACHE,
#     SD_CACHE_UCACHE,
#     SD_CACHE_DBCACHE,
#     SD_CACHE_TAYLORSEER,
#     SD_CACHE_CACHE_DIT,
# };
class SDCacheMode(IntEnum):
    SD_CACHE_DISABLED = 0
    SD_CACHE_EASYCACHE = 1
    SD_CACHE_UCACHE = 2
    SD_CACHE_DBCACHE = 3
    SD_CACHE_TAYLORSEER = 4
    SD_CACHE_CACHE_DIT = 5


# ===========================================
# Inference
# ===========================================


# -------------------------------------------
# sd_embedding_t
# -------------------------------------------


# typedef struct { const char* name; const char* path; } sd_embedding_t;
class sd_embedding_t(ctypes.Structure):
    _fields_ = [
        ("name", ctypes.c_char_p),
        ("path", ctypes.c_char_p),
    ]


# -------------------------------------------
# sd_ctx_params_t
# -------------------------------------------


# typedef struct { const char* model_path; const char* clip_l_path; const char* clip_g_path; const char* clip_vision_path; const char* t5xxl_path; const char* llm_path; const char* llm_vision_path; const char* diffusion_model_path; const char* high_noise_diffusion_model_path; const char* vae_path; const char* taesd_path; const char* control_net_path; const sd_embedding_t* embeddings; uint32_t embedding_count; const char* photo_maker_path; const char* tensor_type_rules; bool vae_decode_only; bool free_params_immediately; int n_threads; enum sd_type_t wtype; enum rng_type_t rng_type; enum rng_type_t sampler_rng_type; enum prediction_t prediction; enum lora_apply_mode_t lora_apply_mode; bool offload_params_to_cpu; bool enable_mmap; bool keep_clip_on_cpu; bool keep_control_net_on_cpu; bool keep_vae_on_cpu; bool diffusion_flash_attn; bool tae_preview_only; bool diffusion_conv_direct; bool vae_conv_direct; bool circular_x; bool circular_y; bool force_sdxl_vae_conv_scale; bool chroma_use_dit_mask; bool chroma_use_t5_mask; int chroma_t5_mask_pad; bool qwen_image_zero_cond_t; float flow_shift; } sd_ctx_params_t;
class sd_ctx_params_t(ctypes.Structure):
    _fields_ = [
        ("model_path", ctypes.c_char_p),
        ("clip_l_path", ctypes.c_char_p),
        ("clip_g_path", ctypes.c_char_p),
        ("clip_vision_path", ctypes.c_char_p),
        ("t5xxl_path", ctypes.c_char_p),
        ("llm_path", ctypes.c_char_p),
        ("llm_vision_path", ctypes.c_char_p),
        ("diffusion_model_path", ctypes.c_char_p),
        ("high_noise_diffusion_model_path", ctypes.c_char_p),
        ("vae_path", ctypes.c_char_p),
        ("taesd_path", ctypes.c_char_p),
        ("control_net_path", ctypes.c_char_p),
        ("embeddings", ctypes.POINTER(sd_embedding_t)),
        ("embedding_count", ctypes.c_uint32),
        ("photo_maker_path", ctypes.c_char_p),
        ("tensor_type_rules", ctypes.c_char_p),
        ("vae_decode_only", ctypes.c_bool),
        ("free_params_immediately", ctypes.c_bool),
        ("n_threads", ctypes.c_int),
        ("wtype", ctypes.c_int),  # GGMLType
        ("rng_type", ctypes.c_int),  # RNGType
        ("sampler_rng_type", ctypes.c_int),  # RNGType
        ("prediction", ctypes.c_int),  # Prediction
        ("lora_apply_mode", ctypes.c_int),  # LoraApplyMode
        ("offload_params_to_cpu", ctypes.c_bool),
        ("enable_mmap", ctypes.c_bool),
        ("keep_clip_on_cpu", ctypes.c_bool),
        ("keep_control_net_on_cpu", ctypes.c_bool),
        ("keep_vae_on_cpu", ctypes.c_bool),
        ("diffusion_flash_attn", ctypes.c_bool),
        ("tae_preview_only", ctypes.c_bool),
        ("diffusion_conv_direct", ctypes.c_bool),
        ("vae_conv_direct", ctypes.c_bool),
        ("circular_x", ctypes.c_bool),
        ("circular_y", ctypes.c_bool),
        ("force_sdxl_vae_conv_scale", ctypes.c_bool),
        ("chroma_use_dit_mask", ctypes.c_bool),
        ("chroma_use_t5_mask", ctypes.c_bool),
        ("chroma_t5_mask_pad", ctypes.c_int),
        ("qwen_image_zero_cond_t", ctypes.c_bool),
        ("flow_shift", ctypes.c_float),
    ]


# -------------------------------------------
# sd_ctx_t
# -------------------------------------------


# typedef struct sd_ctx_t sd_ctx_t;
class sd_ctx_t(ctypes.Structure):
    pass


# struct sd_ctx;
sd_ctx_t_p = NewType("sd_ctx_t_p", int)
sd_ctx_t_p_ctypes = ctypes.POINTER(sd_ctx_t)


# -------------------------------------------
# new_sd_ctx
# -------------------------------------------


# SD_API sd_ctx_t* new_sd_ctx(const sd_ctx_params_t* sd_ctx_params);
@ctypes_function(
    "new_sd_ctx",
    [
        ctypes.POINTER(sd_ctx_params_t),  # sd_ctx_params
    ],
    sd_ctx_t_p_ctypes,
)
def new_sd_ctx(
    sd_ctx_params: sd_ctx_params_t,
    /,
) -> Optional[sd_ctx_t_p]: ...


# -------------------------------------------
# free_sd_ctx
# -------------------------------------------


# SD_API void free_sd_ctx(sd_ctx_t* sd_ctx);
@ctypes_function(
    "free_sd_ctx",
    [
        sd_ctx_t_p_ctypes,  # sd_ctx
    ],
    None,
)
def free_sd_ctx(
    sd_ctx: sd_ctx_t_p,
    /,
): ...


# -------------------------------------------
# sd_image_t
# -------------------------------------------


# typedef struct { uint32_t width; uint32_t height; uint32_t channel; uint8_t* data; } sd_image_t;
class sd_image_t(ctypes.Structure):
    _fields_ = [
        ("width", ctypes.c_uint32),
        ("height", ctypes.c_uint32),
        ("channel", ctypes.c_uint32),
        ("data", ctypes.POINTER(ctypes.c_uint8)),
    ]


# -------------------------------------------
# sd_pm_params_t
# -------------------------------------------


# typedef struct { sd_image_t* id_images; int id_images_count; const char* id_embed_path; float style_strength; } sd_pm_params_t;  // photo maker
class sd_pm_params_t(ctypes.Structure):
    _fields_ = [
        ("id_images", ctypes.POINTER(sd_image_t)),
        ("id_images_count", ctypes.c_int),
        ("id_embed_path", ctypes.c_char_p),
        ("style_strength", ctypes.c_float),
    ]  # photo maker


# -------------------------------------------
# sd_tiling_params_t
# -------------------------------------------


# typedef struct { bool enabled; int tile_size_x; int tile_size_y; float target_overlap; float rel_size_x; float rel_size_y; } sd_tiling_params_t;
class sd_tiling_params_t(ctypes.Structure):
    _fields_ = [
        ("enabled", ctypes.c_bool),
        ("tile_size_x", ctypes.c_int),
        ("tile_size_y", ctypes.c_int),
        ("target_overlap", ctypes.c_float),
        ("rel_size_x", ctypes.c_float),
        ("rel_size_y", ctypes.c_float),
    ]


# -------------------------------------------
# sd_slg_params_t
# -------------------------------------------


# typedef struct { int* layers; size_t layer_count; float layer_start; float layer_end; float scale; } sd_slg_params_t;
class sd_slg_params_t(ctypes.Structure):
    _fields_ = [
        ("layers", ctypes.POINTER(ctypes.c_int)),
        ("layer_count", ctypes.c_size_t),
        ("layer_start", ctypes.c_float),
        ("layer_end", ctypes.c_float),
        ("scale", ctypes.c_float),
    ]


# -------------------------------------------
# sd_guidance_params_t
# -------------------------------------------


# typedef struct { float txt_cfg; float img_cfg; float distilled_guidance; sd_slg_params_t slg; } sd_guidance_params_t;
class sd_guidance_params_t(ctypes.Structure):
    _fields_ = [
        ("txt_cfg", ctypes.c_float),
        ("img_cfg", ctypes.c_float),
        ("distilled_guidance", ctypes.c_float),
        ("slg", sd_slg_params_t),
    ]


# -------------------------------------------
# sd_sample_params_t
# -------------------------------------------


# typedef struct { sd_guidance_params_t guidance; enum scheduler_t scheduler; enum sample_method_t sample_method; int sample_steps; float eta; int shifted_timestep; float* custom_sigmas; int custom_sigmas_count; } sd_sample_params_t;
class sd_sample_params_t(ctypes.Structure):
    _fields_ = [
        ("guidance", sd_guidance_params_t),
        ("scheduler", ctypes.c_int),  # Scheduler
        ("sample_method", ctypes.c_int),  # SampleMethod
        ("sample_steps", ctypes.c_int),
        ("eta", ctypes.c_float),
        ("shifted_timestep", ctypes.c_int),
        ("custom_sigmas", ctypes.POINTER(ctypes.c_float)),
        ("custom_sigmas_count", ctypes.c_int),
    ]


# -------------------------------------------
# sd_cache_params_t
# -------------------------------------------


# typedef struct { enum sd_cache_mode_t mode; float reuse_threshold; float start_percent; float end_percent; float error_decay_rate; bool use_relative_threshold; bool reset_error_on_compute; int Fn_compute_blocks; int Bn_compute_blocks; float residual_diff_threshold; int max_warmup_steps; int max_cached_steps; int max_continuous_cached_steps; int taylorseer_n_derivatives; int taylorseer_skip_interval; const char* scm_mask; bool scm_policy_dynamic; } sd_cache_params_t;
class sd_cache_params_t(ctypes.Structure):
    _fields_ = [
        ("mode", ctypes.c_int),  # SDCacheMode
        ("reuse_threshold", ctypes.c_float),
        ("start_percent", ctypes.c_float),
        ("end_percent", ctypes.c_float),
        ("error_decay_rate", ctypes.c_float),
        ("use_relative_threshold", ctypes.c_bool),
        ("reset_error_on_compute", ctypes.c_bool),
        ("Fn_compute_blocks", ctypes.c_int),
        ("Bn_compute_blocks", ctypes.c_int),
        ("residual_diff_threshold", ctypes.c_float),
        ("max_warmup_steps", ctypes.c_int),
        ("max_cached_steps", ctypes.c_int),
        ("max_continuous_cached_steps", ctypes.c_int),
        ("taylorseer_n_derivatives", ctypes.c_int),
        ("taylorseer_skip_interval", ctypes.c_int),
        ("scm_mask", ctypes.c_char_p),
        ("scm_policy_dynamic", ctypes.c_bool),
    ]


# -------------------------------------------
# sd_lora_t
# -------------------------------------------


# typedef struct { bool is_high_noise; float multiplier; const char* path; } sd_lora_t;
class sd_lora_t(ctypes.Structure):
    _fields_ = [
        ("is_high_noise", ctypes.c_bool),
        ("multiplier", ctypes.c_float),
        ("path", ctypes.c_char_p),
    ]


# -------------------------------------------
# sd_img_gen_params_t
# -------------------------------------------


# typedef struct { const sd_lora_t* loras; uint32_t lora_count; const char* prompt; const char* negative_prompt; int clip_skip; sd_image_t init_image; sd_image_t* ref_images; int ref_images_count; bool auto_resize_ref_image; bool increase_ref_index; sd_image_t mask_image; int width; int height; sd_sample_params_t sample_params; float strength; int64_t seed; int batch_count; sd_image_t control_image; float control_strength; sd_pm_params_t pm_params; sd_tiling_params_t vae_tiling_params; sd_cache_params_t cache; } sd_img_gen_params_t;
class sd_img_gen_params_t(ctypes.Structure):
    _fields_ = [
        ("loras", ctypes.POINTER(sd_lora_t)),
        ("lora_count", ctypes.c_uint32),
        ("prompt", ctypes.c_char_p),
        ("negative_prompt", ctypes.c_char_p),
        ("clip_skip", ctypes.c_int),
        ("init_image", sd_image_t),
        ("ref_images", ctypes.POINTER(sd_image_t)),
        ("ref_images_count", ctypes.c_int),
        ("auto_resize_ref_image", ctypes.c_bool),
        ("increase_ref_index", ctypes.c_bool),
        ("mask_image", sd_image_t),
        ("width", ctypes.c_int),
        ("height", ctypes.c_int),
        ("sample_params", sd_sample_params_t),
        ("strength", ctypes.c_float),
        ("seed", ctypes.c_int64),
        ("batch_count", ctypes.c_int),
        ("control_image", sd_image_t),
        ("control_strength", ctypes.c_float),
        ("pm_params", sd_pm_params_t),
        ("vae_tiling_params", sd_tiling_params_t),
        ("cache", sd_cache_params_t),
    ]


# -------------------------------------------
# generate_image
# -------------------------------------------


# SD_API sd_image_t* generate_image(sd_ctx_t* sd_ctx, const sd_img_gen_params_t* sd_img_gen_params);
@ctypes_function(
    "generate_image",
    [
        sd_ctx_t_p_ctypes,  # sd_ctx
        ctypes.POINTER(sd_img_gen_params_t),  # sd_img_gen_params
    ],
    ctypes.POINTER(sd_image_t),
)
def generate_image(
    sd_ctx: sd_ctx_t_p,
    sd_img_gen_params: sd_img_gen_params_t,
    /,
) -> CtypesArray[sd_image_t]: ...


# -------------------------------------------
# sd_vid_gen_params_t
# -------------------------------------------


# typedef struct { const sd_lora_t* loras; uint32_t lora_count; const char* prompt; const char* negative_prompt; int clip_skip; sd_image_t init_image; sd_image_t end_image; sd_image_t* control_frames; int control_frames_size; int width; int height; sd_sample_params_t sample_params; sd_sample_params_t high_noise_sample_params; float moe_boundary; float strength; int64_t seed; int video_frames; float vace_strength; sd_tiling_params_t vae_tiling_params; sd_cache_params_t cache; } sd_vid_gen_params_t;
class sd_vid_gen_params_t(ctypes.Structure):
    _fields_ = [
        ("loras", ctypes.POINTER(sd_lora_t)),
        ("lora_count", ctypes.c_uint32),
        ("prompt", ctypes.c_char_p),
        ("negative_prompt", ctypes.c_char_p),
        ("clip_skip", ctypes.c_int),
        ("init_image", sd_image_t),
        ("end_image", sd_image_t),
        ("control_frames", ctypes.POINTER(sd_image_t)),
        ("control_frames_size", ctypes.c_int),
        ("width", ctypes.c_int),
        ("height", ctypes.c_int),
        ("sample_params", sd_sample_params_t),
        ("high_noise_sample_params", sd_sample_params_t),
        ("moe_boundary", ctypes.c_float),
        ("strength", ctypes.c_float),
        ("seed", ctypes.c_int64),
        ("video_frames", ctypes.c_int),
        ("vace_strength", ctypes.c_float),
        ("vae_tiling_params", sd_tiling_params_t),
        ("cache", sd_cache_params_t),
    ]


# -------------------------------------------
# generate_video
# -------------------------------------------


num_frames_out_p = NewType("num_frames_out_p", int)


# SD_API sd_image_t* generate_video(sd_ctx_t* sd_ctx, const sd_vid_gen_params_t* sd_vid_gen_params, int* num_frames_out);
@ctypes_function(
    "generate_video",
    [
        sd_ctx_t_p_ctypes,  # sd_ctx
        ctypes.POINTER(sd_vid_gen_params_t),  # sd_vid_gen_params
        ctypes.POINTER(ctypes.c_int),  # num_frames_out
    ],
    ctypes.POINTER(sd_image_t),
)
def generate_video(
    sd_ctx: sd_ctx_t_p,
    sd_vid_gen_params: sd_vid_gen_params_t,
    num_frames_out: num_frames_out_p,
    /,
) -> CtypesArray[sd_image_t]: ...


# -------------------------------------------
# sd_get_default_sample_method
# -------------------------------------------


# SD_API enum sample_method_t sd_get_default_sample_method(const sd_ctx_t* sd_ctx);
@ctypes_function(
    "sd_get_default_sample_method",
    [
        sd_ctx_t_p_ctypes,  # sd_ctx
    ],
    ctypes.c_int,  # SampleMethod
)
def sd_get_default_sample_method(
    sd_ctx: sd_ctx_t_p,
    /,
) -> Optional[SampleMethod]: ...


# -------------------------------------------
# sd_get_default_scheduler
# -------------------------------------------


# SD_API enum scheduler_t sd_get_default_scheduler(const sd_ctx_t* sd_ctx);
@ctypes_function(
    "sd_get_default_scheduler",
    [
        sd_ctx_t_p_ctypes,  # sd_ctx
    ],
    ctypes.c_int,  # Scheduler
)
def sd_get_default_scheduler(
    sd_ctx: sd_ctx_t_p,
    /,
) -> Optional[Scheduler]: ...


# -------------------------------------------
# upscaler_ctx_t
# -------------------------------------------


# typedef struct upscaler_ctx_t upscaler_ctx_t;
class upscaler_ctx_t(ctypes.Structure):
    pass


# struct upscaler_ctx;
upscaler_ctx_t_p = NewType("upscaler_ctx_t_p", int)
upscaler_ctx_t_p_ctypes = ctypes.POINTER(upscaler_ctx_t)


# -------------------------------------------
# new_upscaler_ctx
# -------------------------------------------


# SD_API upscaler_ctx_t* new_upscaler_ctx(const char* esrgan_path, bool offload_params_to_cpu, bool direct, int n_threads, int tile_size);
@ctypes_function(
    "new_upscaler_ctx",
    [
        ctypes.c_char_p,  # esrgan_path
        ctypes.c_bool,  # offload_params_to_cpu
        ctypes.c_bool,  # direct
        ctypes.c_int,  # n_threads
        ctypes.c_int,  # tile_size
    ],
    upscaler_ctx_t_p_ctypes,
)
def new_upscaler_ctx(
    esrgan_path: bytes,
    offload_params_to_cpu: bool,
    direct: bool,
    n_threads: int,
    tile_size: int,
    /,
) -> upscaler_ctx_t_p: ...


# -------------------------------------------
# free_upscaler_ctx
# -------------------------------------------


# SD_API void free_upscaler_ctx(upscaler_ctx_t* upscaler_ctx);
@ctypes_function(
    "free_upscaler_ctx",
    [
        upscaler_ctx_t_p_ctypes,  # upscaler_ctx
    ],
    None,
)
def free_upscaler_ctx(
    upscaler_ctx: upscaler_ctx_t_p,
    /,
) -> None: ...


# -------------------------------------------
# upscale
# -------------------------------------------


# SD_API sd_image_t upscale(upscaler_ctx_t* upscaler_ctx, sd_image_t input_image, uint32_t upscale_factor);
@ctypes_function(
    "upscale",
    [
        upscaler_ctx_t_p_ctypes,  # upscaler_ctx
        sd_image_t,  # input_image
        ctypes.c_uint32,  # upscale_factor
    ],
    sd_image_t,
)
def upscale(
    upscaler_ctx: upscaler_ctx_t_p,
    input_image: sd_image_t,
    upscale_factor: int,
    /,
) -> sd_image_t: ...


# -------------------------------------------
# get_upscale_factor
# -------------------------------------------


# SD_API int get_upscale_factor(upscaler_ctx_t* upscaler_ctx);
@ctypes_function(
    "get_upscale_factor",
    [
        upscaler_ctx_t_p_ctypes,  # upscaler_ctx
    ],
    ctypes.c_int,
)
def get_upscale_factor(
    upscaler_ctx: upscaler_ctx_t_p,
    /,
) -> int: ...


# -------------------------------------------
# convert
# -------------------------------------------


# SD_API bool convert(const char* input_path, const char* vae_path, const char* output_path, enum sd_type_t output_type, const char* tensor_type_rules, bool convert_name);
@ctypes_function(
    "convert",
    [
        ctypes.c_char_p,  # input_path
        ctypes.c_char_p,  # vae_path
        ctypes.c_char_p,  # output_path
        ctypes.c_int,  # output_type
        ctypes.c_char_p,  # tensor_type_rules
        ctypes.c_bool,  # convert_name
    ],
    ctypes.c_bool,
)
def convert(
    input_path: bytes,
    vae_path: bytes,
    output_path: bytes,
    output_type: int,
    tensor_type_rules: bytes,
    convert_name: bool,
    /,
) -> bool: ...


# -------------------------------------------
# preprocess_canny
# -------------------------------------------


# SD_API bool preprocess_canny(sd_image_t image, float high_threshold, float low_threshold, float weak, float strong, bool inverse);
@ctypes_function(
    "preprocess_canny",
    [
        sd_image_t,  # image
        ctypes.c_float,  # high_threshold
        ctypes.c_float,  # low_threshold
        ctypes.c_float,  # weak
        ctypes.c_float,  # strong
        ctypes.c_bool,  # inverse
    ],
    ctypes.c_bool,
)
def preprocess_canny(
    image: sd_image_t,
    high_threshold: float,
    low_threshold: float,
    weak: float,
    strong: float,
    inverse: bool,
    /,
) -> bool: ...


# ===========================================
# System Information
# ===========================================

# -------------------------------------------
# sd_get_num_physical_cores
# -------------------------------------------


# SD_API int32_t sd_get_num_physical_cores();
@ctypes_function(
    "sd_get_num_physical_cores",
    [],
    ctypes.c_int32,
)
def sd_get_num_physical_cores() -> int:
    """Get the number of physical cores"""
    ...


# -------------------------------------------
# sd_get_system_info
# -------------------------------------------


# SD_API const char* sd_get_system_info();
@ctypes_function(
    "sd_get_system_info",
    [],
    ctypes.c_char_p,
)
def sd_get_system_info() -> bytes:
    """Get the Stable diffusion system information"""
    ...


# -------------------------------------------
# sd_commit
# -------------------------------------------


# SD_API const char* sd_commit(void);
@ctypes_function(
    "sd_commit",
    [],
    ctypes.c_char_p,
)
def sd_commit() -> bytes:
    """Get the Stable diffusion commit hash"""
    ...


# -------------------------------------------
# sd_version
# -------------------------------------------


# SD_API const char* sd_version(void);
@ctypes_function(
    "sd_version",
    [],
    ctypes.c_char_p,
)
def sd_version() -> bytes:
    """Get the Stable diffusion version string"""
    ...


# ===========================================
# Progression
# ===========================================

# typedef void (*sd_progress_cb_t)(int step, int steps, float time, void* data);
sd_progress_callback = ctypes.CFUNCTYPE(None, ctypes.c_int, ctypes.c_int, ctypes.c_float, ctypes.c_void_p)


# SD_API void sd_set_progress_callback(sd_progress_cb_t cb, void* data);
@ctypes_function(
    "sd_set_progress_callback",
    [
        ctypes.c_void_p,  # sd_progress_cb_t
        ctypes.c_void_p,  # data
    ],
    None,
)
def sd_set_progress_callback(
    callback: Optional[CtypesFuncPointer],
    data: ctypes.c_void_p,
    /,
):
    """Set callback for diffusion progression events."""
    ...


# ===========================================
# Preview
# ===========================================

# typedef void (*sd_preview_cb_t)(int step, int frame_count, sd_image_t* frames, bool is_noisy, void* data);
sd_preview_callback = ctypes.CFUNCTYPE(
    None, ctypes.c_int, ctypes.c_int, ctypes.POINTER(sd_image_t), ctypes.c_bool, ctypes.c_void_p
)


# SD_API void sd_set_preview_callback(sd_preview_cb_t cb, enum preview_t mode, int interval, bool denoised, bool noisy, void* data);
@ctypes_function(
    "sd_set_preview_callback",
    [
        ctypes.c_void_p,  # sd_preview_cb_t
        ctypes.c_int,  # mode
        ctypes.c_int,  # interval
        ctypes.c_bool,  # denoised
        ctypes.c_bool,  # noisy
        ctypes.c_void_p,  # data
    ],
    None,
)
def sd_set_preview_callback(
    callback: Optional[CtypesFuncPointer],
    mode: int,
    interval: int,
    denoised: bool,
    noisy: bool,
    data: ctypes.c_void_p,
    /,
):
    """Set callback for preview images during generation."""
    ...


# ===========================================
# Logging
# ===========================================

sd_log_callback = ctypes.CFUNCTYPE(None, ctypes.c_int, ctypes.c_char_p, ctypes.c_void_p)


# SD_API void sd_set_log_callback(sd_log_cb_t sd_log_cb, void* data);
@ctypes_function(
    "sd_set_log_callback",
    [ctypes.c_void_p, ctypes.c_void_p],
    None,
)
def sd_set_log_callback(
    callback: Optional[CtypesFuncPointer],
    data: ctypes.c_void_p,
    /,
):
    """Set callback for all future logging events.
    If this is not called, or NULL is supplied, everything is output on stderr."""
    ...
