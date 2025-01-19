import os
import traceback

import stable_diffusion_cpp.stable_diffusion_cpp as sd_cpp

try:
    # Get system info
    system_info = sd_cpp.sd_get_system_info()
    num_physical_cores = sd_cpp.get_num_physical_cores()

    # Print system info
    print("System info: ", system_info)
    print("Number of physical cores: ", num_physical_cores)

    OUTPUT_DIR = "tests/outputs"
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)

    # Write system info to file txt
    with open(f"{OUTPUT_DIR}/system_info.txt", "w") as f:
        f.write(f"System info: {str(system_info)}\n")
        f.write(f"Number of physical cores: {str(num_physical_cores)}")

except Exception as e:
    traceback.print_exc()
    print("Test - system_info failed: ", e)
