from conftest import OUTPUT_DIR

import stable_diffusion_cpp.stable_diffusion_cpp as sd_cpp


def test_system_info():
    # Get system info
    system_info = sd_cpp.sd_get_system_info()
    num_physical_cores = sd_cpp.sd_get_num_physical_cores()
    sd_version = sd_cpp.sd_version().decode("utf-8")
    sd_commit = sd_cpp.sd_commit().decode("utf-8")

    # Print system info
    print("System info: ", system_info)
    print("Number of physical cores: ", num_physical_cores)
    print("SD Version: ", sd_version)
    print("SD Commit: ", sd_commit)

    # Write system info to file txt
    with open(f"{OUTPUT_DIR}/system_info.txt", "w") as f:
        f.write(f"System info: {str(system_info)}\n")
        f.write(f"Number of physical cores: {str(num_physical_cores)}\n")
        f.write(f"SD Version: {sd_version}\n")
        f.write(f"SD Commit: {sd_commit}\n")
