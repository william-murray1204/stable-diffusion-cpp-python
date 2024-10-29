cmake_minimum_required(VERSION 3.21)

project(stable_diffusion_cpp)

option(STABLE_DIFFUSION_BUILD "Build stable-diffusion.cpp shared library and install alongside python package" ON)

if (STABLE_DIFFUSION_BUILD)
    # Ensure we build shared libraries (dlls)
    set(BUILD_SHARED_LIBS "ON")
    option(SD_BUILD_SHARED_LIBS "" "ON")

    # Explicitly set the build type to Release (default `Debug` is slow)
    set(CMAKE_BUILD_TYPE "Release")
    option(CMAKE_BUILD_TYPE "" "Release")

    add_subdirectory(vendor/stable-diffusion.cpp)
    install(
        TARGETS stable-diffusion
        LIBRARY DESTINATION ${SKBUILD_PLATLIB_DIR}/stable_diffusion_cpp/lib
        RUNTIME DESTINATION ${SKBUILD_PLATLIB_DIR}/stable_diffusion_cpp/lib
        ARCHIVE DESTINATION ${SKBUILD_PLATLIB_DIR}/stable_diffusion_cpp/lib
        FRAMEWORK DESTINATION ${SKBUILD_PLATLIB_DIR}/stable_diffusion_cpp/lib
        RESOURCE DESTINATION ${SKBUILD_PLATLIB_DIR}/stable_diffusion_cpp/lib
    )

    message(SKBUILD_PLATLIB_DIR: ${SKBUILD_PLATLIB_DIR})
    # Temporary fix for https://github.com/scikit-build/scikit-build-core/issues/374
    install(
        TARGETS stable-diffusion
        LIBRARY DESTINATION ${CMAKE_CURRENT_SOURCE_DIR}/stable_diffusion_cpp/lib
        RUNTIME DESTINATION ${CMAKE_CURRENT_SOURCE_DIR}/stable_diffusion_cpp/lib
        ARCHIVE DESTINATION ${CMAKE_CURRENT_SOURCE_DIR}/stable_diffusion_cpp/lib
        FRAMEWORK DESTINATION ${CMAKE_CURRENT_SOURCE_DIR}/stable_diffusion_cpp/lib
        RESOURCE DESTINATION ${CMAKE_CURRENT_SOURCE_DIR}/stable_diffusion_cpp/lib
    )
    # Workaround for Windows + CUDA https://github.com/abetlen/llama-cpp-python/issues/563
    install(
        FILES $<TARGET_RUNTIME_DLLS:stable-diffusion>
        DESTINATION ${SKBUILD_PLATLIB_DIR}/stable_diffusion_cpp/lib
    )
    install(
        FILES $<TARGET_RUNTIME_DLLS:stable-diffusion>
        DESTINATION ${CMAKE_CURRENT_SOURCE_DIR}/stable_diffusion_cpp/lib
    )

endif()
