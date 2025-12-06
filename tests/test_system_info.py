from conftest import OUTPUT_DIR

import stable_diffusion_cpp.stable_diffusion_cpp as sd_cpp


def test_system_info():
    # Get system info
    system_info = sd_cpp.sd_get_system_info()
    num_physical_cores = sd_cpp.sd_get_num_physical_cores()

    # Print system info
    print("System info: ", system_info)
    print("Number of physical cores: ", num_physical_cores)

    # Write system info to file txt
    with open(f"{OUTPUT_DIR}/system_info.txt", "w") as f:
        f.write(f"System info: {str(system_info)}\n")
        f.write(f"Number of physical cores: {str(num_physical_cores)}")
