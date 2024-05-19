import stable_diffusion_cpp.stable_diffusion_cpp as sd_cpp


try:
    # Get system info
    system_info = sd_cpp.sd_get_system_info()
    num_physical_cores = sd_cpp.get_num_physical_cores()

    # Print system info
    print("System info: ", system_info)
    print("Number of physical cores: ", num_physical_cores)

except Exception as e:
    print("Test - system_info failed: ", e)
