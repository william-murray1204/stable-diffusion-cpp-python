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


# from ggml-backend.h
# typedef bool (*ggml_backend_sched_eval_callback)(struct ggml_tensor * t, bool ask, void * user_data);
ggml_backend_sched_eval_callback = ctypes.CFUNCTYPE(ctypes.c_bool, ctypes.c_void_p, ctypes.c_bool, ctypes.c_void_p)

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
#     CUDA_RNG
# };
class RNGType(IntEnum):
    STD_DEFAULT_RNG = 0
    CUDA_RNG = 1


# enum sample_method_t {
#     EULER_A,
#     EULER,
#     HEUN,
#     DPM2,
#     DPMPP2S_A,
#     DPMPP2M,
#     DPMPP2Mv2,
#     IPNDM,
#     IPNDM_V,
#     LCM,
#     DDIM_TRAILING,
#     TCD,
#     N_SAMPLE_METHODS
# };
class SampleMethod(IntEnum):
    EULER_A = 0
    EULER = 1
    HEUN = 2
    DPM2 = 3
    DPMPP2S_A = 4
    DPMPP2M = 5
    DPMPP2Mv2 = 6
    IPNDM = 7
    IPNDM_V = 8
    LCM = 9
    DDIM_TRAILING = 10
    TCD = 11
    N_SAMPLE_METHODS = 12


# enum schedule_t {
#     DEFAULT,
#     DISCRETE,
#     KARRAS,
#     EXPONENTIAL,
#     AYS,
#     GITS,
#     N_SCHEDULES
# };
class Schedule(IntEnum):
    DEFAULT = 0
    DISCRETE = 1
    KARRAS = 2
    EXPONENTIAL = 3
    AYS = 4
    GITS = 5
    N_SCHEDULES = 6


# // same as enum ggml_type
# enum sd_type_t {
#     SD_TYPE_F32     = 0,
#     SD_TYPE_F16     = 1,
#     SD_TYPE_Q4_0    = 2,
#     SD_TYPE_Q4_1    = 3,
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
#     SD_TYPE_TQ1_0   = 34,
#     SD_TYPE_TQ2_0   = 35,
#     // SD_TYPE_IQ4_NL_4_4 = 36,
#     // SD_TYPE_IQ4_NL_4_8 = 37,
#     // SD_TYPE_IQ4_NL_8_8 = 38,
#     SD_TYPE_COUNT   = 39,
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
    SD_TYPE_COUNT = 39


# ==================================
# Inference
# ==================================

# ------------ new_sd_ctx ------------

# struct sd_context;
sd_ctx_t_p = NewType("sd_ctx_t_p", int)
sd_ctx_t_p_ctypes = ctypes.c_void_p


# SD_API sd_ctx_t* new_sd_ctx(const char* model_path, const char* clip_l_path, const char* clip_g_path, const char* t5xxl_path, const char* diffusion_model_path, const char* vae_path, const char* taesd_path, const char* control_net_path_c_str, const char* lora_model_dir, const char* embed_dir_c_str, const char* stacked_id_embed_dir_c_str, bool vae_decode_only, bool vae_tiling, bool free_params_immediately, int n_threads, enum sd_type_t wtype, enum rng_type_t rng_type, enum schedule_t s, bool keep_clip_on_cpu, bool keep_control_net_cpu, bool keep_vae_on_cpu, bool diffusion_flash_attn, bool chroma_use_dit_mask, bool chroma_use_t5_mask, int chroma_t5_mask_pad);
@ctypes_function(
    "new_sd_ctx",
    [
        ctypes.c_char_p,  # model_path
        ctypes.c_char_p,  # clip_l_path
        ctypes.c_char_p,  # clip_g_path
        ctypes.c_char_p,  # t5xxl_path
        ctypes.c_char_p,  # diffusion_model_path
        ctypes.c_char_p,  # vae_path
        ctypes.c_char_p,  # taesd_path
        ctypes.c_char_p,  # control_net_path
        ctypes.c_char_p,  # lora_model_dir
        ctypes.c_char_p,  # embed_dir
        ctypes.c_char_p,  # stacked_id_embed_dir
        ctypes.c_bool,  # vae_decode_only
        ctypes.c_bool,  # vae_tiling
        ctypes.c_bool,  # free_params_immediately
        ctypes.c_int,  # n_threads
        ctypes.c_int,  # wtype (GGMLType)
        ctypes.c_int,  # rng_type (RNGType)
        ctypes.c_int,  # s (Schedule)
        ctypes.c_bool,  # keep_clip_on_cpu
        ctypes.c_bool,  # keep_control_net_cpu
        ctypes.c_bool,  # keep_vae_on_cpu
        ctypes.c_bool,  # diffusion_flash_attn
        ctypes.c_bool,  # chroma_use_dit_mask
        ctypes.c_bool,  # chroma_use_t5_mask
        ctypes.c_int,  # chroma_t5_mask_pad
    ],
    sd_ctx_t_p_ctypes,
)
def new_sd_ctx(
    model_path: bytes,
    clip_l_path: bytes,
    clip_g_path: bytes,
    t5xxl_path: bytes,
    diffusion_model_path: bytes,
    vae_path: bytes,
    taesd_path: bytes,
    control_net_path: bytes,
    lora_model_dir: bytes,
    embed_dir: bytes,
    stacked_id_embed_dir: bytes,
    vae_decode_only: bool,
    vae_tiling: bool,
    free_params_immediately: bool,
    n_threads: int,
    wtype: int,  # GGMLType
    rng_type: int,  # RNGType
    s: int,  # Schedule
    keep_clip_on_cpu: bool,
    keep_control_net_cpu: bool,
    keep_vae_on_cpu: bool,
    diffusion_flash_attn: bool,
    chroma_use_dit_mask: bool,
    chroma_use_t5_mask: bool,
    chroma_t5_mask_pad: int,
    /,
) -> Optional[sd_ctx_t_p]: ...


# ------------ free_sd_ctx ------------


# SD_API void free_sd_ctx(sd_ctx_t* sd_ctx);
@ctypes_function(
    "free_sd_ctx",
    [sd_ctx_t_p_ctypes],  # sd_ctx
    None,
)
def free_sd_ctx(
    sd_ctx: sd_ctx_t_p,
    /,
): ...


# ------------ sd_image_t ------------


class sd_image_t(ctypes.Structure):
    _fields_ = [
        ("width", ctypes.c_uint32),
        ("height", ctypes.c_uint32),
        ("channel", ctypes.c_uint32),
        ("data", ctypes.POINTER(ctypes.c_uint8)),
    ]


sd_image_t_p = ctypes.POINTER(sd_image_t)


# ------------ txt2img ------------


# SD_API sd_image_t* txt2img(sd_ctx_t* sd_ctx, const char* prompt, const char* negative_prompt, int clip_skip, float cfg_scale, float guidance, float eta, int width, int height, enum sample_method_t sample_method, int sample_steps, int64_t seed, int batch_count, const sd_image_t* control_cond, float control_strength, float style_strength, bool normalize_input, const char* input_id_images_path, int* skip_layers, size_t skip_layers_count, float slg_scale, float skip_layer_start, float skip_layer_end);
@ctypes_function(
    "txt2img",
    [
        sd_ctx_t_p_ctypes,  # sd_ctx
        ctypes.c_char_p,  # prompt
        ctypes.c_char_p,  # negative_prompt
        ctypes.c_int,  # clip_skip
        ctypes.c_float,  # cfg_scale
        ctypes.c_float,  # guidance
        ctypes.c_float,  # eta
        ctypes.c_int,  # width
        ctypes.c_int,  # height
        ctypes.c_int,  # sample_method
        ctypes.c_int,  # sample_steps
        ctypes.c_int64,  # seed
        ctypes.c_int,  # batch_count
        sd_image_t_p,  # control_cond
        ctypes.c_float,  # control_strength
        ctypes.c_float,  # style_strength
        ctypes.c_bool,  # normalize_input
        ctypes.c_char_p,  # input_id_images_path
        ctypes.POINTER(ctypes.c_int),  # skip_layers
        ctypes.c_size_t,  # skip_layers_count
        ctypes.c_float,  # slg_scale
        ctypes.c_float,  # skip_layer_start
        ctypes.c_float,  # skip_layer_end
    ],
    sd_image_t_p,
)
def txt2img(
    sd_ctx: sd_ctx_t_p,
    prompt: bytes,
    negative_prompt: bytes,
    clip_skip: int,
    cfg_scale: float,
    guidance: float,
    eta: float,
    width: int,
    height: int,
    sample_method: int,  # SampleMethod
    sample_steps: int,
    seed: int,
    batch_count: int,
    control_cond: sd_image_t,
    control_strength: float,
    style_strength: float,
    normalize_input: bool,
    input_id_images_path: bytes,
    skip_layers: List[int],
    skip_layers_count: int,
    slg_scale: float,
    skip_layer_start: float,
    skip_layer_end: float,
    /,
) -> CtypesArray[sd_image_t]: ...


# ------------ img2img ------------


# SD_API sd_image_t* img2img(sd_ctx_t* sd_ctx, sd_image_t init_image, sd_image_t mask_image, const char* prompt, const char* negative_prompt, int clip_skip, float cfg_scale, float guidance, float eta, int width, int height, enum sample_method_t sample_method, int sample_steps, float strength, int64_t seed, int batch_count, const sd_image_t* control_cond, float control_strength, float style_strength, bool normalize_input, const char* input_id_images_path, int* skip_layers, size_t skip_layers_count, float slg_scale, float skip_layer_start, float skip_layer_end);
@ctypes_function(
    "img2img",
    [
        sd_ctx_t_p_ctypes,  # sd_ctx
        sd_image_t,  # init_image
        sd_image_t,  # mask_image
        ctypes.c_char_p,  # prompt
        ctypes.c_char_p,  # negative_prompt
        ctypes.c_int,  # clip_skip
        ctypes.c_float,  # cfg_scale
        ctypes.c_float,  # guidance
        ctypes.c_float,  # eta
        ctypes.c_int,  # width
        ctypes.c_int,  # height
        ctypes.c_int,  # sample_method
        ctypes.c_int,  # sample_steps
        ctypes.c_float,  # strength
        ctypes.c_int64,  # seed
        ctypes.c_int,  # batch_count
        sd_image_t_p,  # control_cond
        ctypes.c_float,  # control_strength
        ctypes.c_float,  # style_strength
        ctypes.c_bool,  # normalize_input
        ctypes.c_char_p,  # input_id_images_path
        ctypes.POINTER(ctypes.c_int),  # skip_layers
        ctypes.c_size_t,  # skip_layers_count
        ctypes.c_float,  # slg_scale
        ctypes.c_float,  # skip_layer_start
        ctypes.c_float,  # skip_layer_end
    ],
    sd_image_t_p,
)
def img2img(
    sd_ctx: sd_ctx_t_p,
    init_image: sd_image_t,
    mask_image: sd_image_t,
    prompt: bytes,
    negative_prompt: bytes,
    clip_skip: int,
    cfg_scale: float,
    guidance: float,
    eta: float,
    width: int,
    height: int,
    sample_method: int,  # SampleMethod
    sample_steps: int,
    strength: float,
    seed: int,
    batch_count: int,
    control_cond: sd_image_t,
    control_strength: float,
    style_strength: float,
    normalize_input: bool,
    input_id_images_path: bytes,
    skip_layers: List[int],
    skip_layers_count: int,
    slg_scale: float,
    skip_layer_start: float,
    skip_layer_end: float,
    /,
) -> CtypesArray[sd_image_t]: ...


# ------------ img2vid ------------


# SD_API sd_image_t* img2vid(sd_ctx_t* sd_ctx, sd_image_t init_image, int width, int height, int video_frames, int motion_bucket_id, int fps, float augmentation_level, float min_cfg, float cfg_scale, enum sample_method_t sample_method, int sample_steps, float strength, int64_t seed);
@ctypes_function(
    "img2vid",
    [
        sd_ctx_t_p_ctypes,  # sd_ctx
        sd_image_t,  # init_image
        ctypes.c_int,  # width
        ctypes.c_int,  # height
        ctypes.c_int,  # video_frames
        ctypes.c_int,  # motion_bucket_id
        ctypes.c_int,  # fps
        ctypes.c_float,  # augmentation_level
        ctypes.c_float,  # min_cfg
        ctypes.c_float,  # cfg_scale
        ctypes.c_int,  # sample_method
        ctypes.c_int,  # sample_steps
        ctypes.c_float,  # strength
        ctypes.c_int64,  # seed
    ],
    sd_image_t_p,
)
def img2vid(
    sd_ctx: sd_ctx_t_p,
    init_image: sd_image_t,
    width: int,
    height: int,
    video_frames: int,
    motion_bucket_id: int,
    fps: int,
    augmentation_level: float,
    min_cfg: float,
    cfg_scale: float,
    sample_method: int,  # SampleMethod
    sample_steps: int,
    strength: float,
    seed: int,
    /,
) -> CtypesArray[sd_image_t]: ...


# ------------ edit ------------


# SD_API sd_image_t* edit(sd_ctx_t* sd_ctx, sd_image_t* ref_images, int ref_images_count, const char* prompt, const char* negative_prompt, int clip_skip, float cfg_scale, float guidance, float eta, int width, int height, enum sample_method_t sample_method, int sample_steps, float strength, int64_t seed, int batch_count, const sd_image_t* control_cond, float control_strength, float style_strength, bool normalize_input, int* skip_layers, size_t skip_layers_count, float slg_scale, float skip_layer_start, float skip_layer_end);
@ctypes_function(
    "edit",
    [
        sd_ctx_t_p_ctypes,  # sd_ctx
        ctypes.POINTER(sd_image_t),  # ref_images
        ctypes.c_int,  # ref_images_count
        ctypes.c_char_p,  # prompt
        ctypes.c_char_p,  # negative_prompt
        ctypes.c_int,  # clip_skip
        ctypes.c_float,  # cfg_scale
        ctypes.c_float,  # guidance
        ctypes.c_float,  # eta
        ctypes.c_int,  # width
        ctypes.c_int,  # height
        ctypes.c_int,  # sample_method
        ctypes.c_int,  # sample_steps
        ctypes.c_float,  # strength
        ctypes.c_int64,  # seed
        ctypes.c_int,  # batch_count
        sd_image_t_p,  # control_cond
        ctypes.c_float,  # control_strength
        ctypes.c_float,  # style_strength
        ctypes.c_bool,  # normalize_input
        ctypes.POINTER(ctypes.c_int),  # skip_layers
        ctypes.c_size_t,  # skip_layers_count
        ctypes.c_float,  # slg_scale
        ctypes.c_float,  # skip_layer_start
        ctypes.c_float,  # skip_layer_end
    ],
    sd_image_t_p,
)
def edit(
    sd_ctx: sd_ctx_t_p,
    ref_images: CtypesArray[sd_image_t],
    ref_images_count: int,
    prompt: bytes,
    negative_prompt: bytes,
    clip_skip: int,
    cfg_scale: float,
    guidance: float,
    eta: float,
    width: int,
    height: int,
    sample_method: int,  # SampleMethod
    sample_steps: int,
    strength: float,
    seed: int,
    batch_count: int,
    control_cond: sd_image_t,
    control_strength: float,
    style_strength: float,
    normalize_input: bool,
    skip_layers: List[int],
    skip_layers_count: int,
    slg_scale: float,
    skip_layer_start: float,
    skip_layer_end: float,
    /,
) -> CtypesArray[sd_image_t]: ...


# ------------ new_upscaler_ctx ------------

upscaler_ctx_t_p = NewType("upscaler_ctx_t_p", int)
upscaler_ctx_t_p_ctypes = ctypes.c_void_p


# SD_API upscaler_ctx_t* new_upscaler_ctx(const char* esrgan_path, int n_threads);
@ctypes_function(
    "new_upscaler_ctx",
    [
        ctypes.c_char_p,  # esrgan_path
        ctypes.c_int,  # n_threads
    ],
    upscaler_ctx_t_p_ctypes,
)
def new_upscaler_ctx(
    esrgan_path: bytes,
    n_threads: int,
    /,
) -> upscaler_ctx_t_p: ...


# ------------ free_upscaler_ctx ------------


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


# ------------ upscale ------------


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


# ------------ convert ------------


# SD_API bool convert(const char* input_path, const char* vae_path, const char* output_path, sd_type_t output_type);
@ctypes_function(
    "convert",
    [
        ctypes.c_char_p,  # input_path
        ctypes.c_char_p,  # vae_path
        ctypes.c_char_p,  # output_path
        ctypes.c_int,  # output_type
    ],
    ctypes.c_bool,
)
def convert(
    input_path: bytes,
    vae_path: bytes,
    output_path: bytes,
    output_type: int,
    /,
) -> bool: ...


# ------------ preprocess_canny ------------


# SD_API uint8_t* preprocess_canny(uint8_t* img, int width, int height, float high_threshold, float low_threshold, float weak, float strong, bool inverse);
@ctypes_function(
    "preprocess_canny",
    [
        ctypes.POINTER(ctypes.c_uint8),  # img
        ctypes.c_int,  # width
        ctypes.c_int,  # height
        ctypes.c_float,  # high_threshold
        ctypes.c_float,  # low_threshold
        ctypes.c_float,  # weak
        ctypes.c_float,  # strong
        ctypes.c_bool,  # inverse
    ],
    ctypes.POINTER(ctypes.c_uint8),
)
def preprocess_canny(
    img: CtypesArray[ctypes.c_uint8],
    width: int,
    height: int,
    high_threshold: float,
    low_threshold: float,
    weak: float,
    strong: float,
    inverse: bool,
    /,
) -> CtypesArray[ctypes.c_uint8]: ...


# ==================================
# System Information
# ==================================


# SD_API int32_t get_num_physical_cores();
@ctypes_function(
    "get_num_physical_cores",
    [],
    ctypes.c_int32,
)
def get_num_physical_cores() -> int:
    """Get the number of physical cores"""
    ...


# SD_API const char* sd_get_system_info();
@ctypes_function(
    "sd_get_system_info",
    [],
    ctypes.c_char_p,
)
def sd_get_system_info() -> bytes:
    """Get the Stable diffusion system information"""
    ...


# ==================================
# Progression
# ==================================

sd_progress_callback = ctypes.CFUNCTYPE(None, ctypes.c_int, ctypes.c_int, ctypes.c_float, ctypes.c_void_p)


# SD_API void sd_set_progress_callback(sd_progress_cb_t cb, void* data);
@ctypes_function(
    "sd_set_progress_callback",
    [ctypes.c_void_p, ctypes.c_void_p],
    None,
)
def sd_set_progress_callback(
    callback: Optional[CtypesFuncPointer],
    data: ctypes.c_void_p,
    /,
):
    """Set callback for diffusion progression events."""
    ...


# ==================================
# Logging
# ==================================

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
