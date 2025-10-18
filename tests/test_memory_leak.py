import os
import time
import subprocess

from stable_diffusion_cpp import StableDiffusion


def get_amd_vram_ps() -> float:
    """Return current VRAM usage (MB) for this process using PowerShell."""
    pid = os.getpid()
    ps_script = f"""
    $p = Get-Process -Id {pid}
    $mem = (Get-Counter "\\GPU Process Memory(pid_$($p.Id)*)\\Local Usage").CounterSamples |
           Where-Object {{ $_.CookedValue -gt 0 }} |
           Select-Object -ExpandProperty CookedValue
    if ($mem) {{ [math]::Round($mem / 1MB, 2) }} else {{ 0 }}
    """
    try:
        result = subprocess.check_output(["powershell", "-Command", ps_script], stderr=subprocess.STDOUT, text=True).strip()
        return float(result)
    except Exception as e:
        print("VRAM read error:", e)
        return -1.0


def log_vram(label: str, baseline: float = None) -> float:
    """Log VRAM at a label and optionally show difference from baseline."""
    vram = get_amd_vram_ps()
    if baseline is not None and vram >= 0:
        diff = round(vram - baseline, 2)
        print(f"[{label}] VRAM: {vram} MB ({diff} MB from start)")
    else:
        print(f"[{label}] VRAM: {vram} MB")
    return vram


MODEL_PATH = "F:\\stable-diffusion\\juggernautXL_V8+RDiffusion.safetensors"
VAE_PATH = "F:\\stable-diffusion\\vaes\\sdxl_vae.safetensors"


def generate_cat():
    sd = StableDiffusion(
        model_path=MODEL_PATH,
        vae_path=VAE_PATH,
        keep_vae_on_cpu=True,
    )
    img = sd.generate_image(
        prompt="a lovely cat",
        sample_steps=4,
    )[0]
    return sd, img


# ===========================================
# Start Test
# ===========================================


def test_memory_leak():
    start_vram = log_vram("Start")

    # First load & generate
    sd, img = generate_cat()

    # Unload
    sd = None
    time.sleep(3)
    after_first_unload = log_vram("After First Unload", baseline=start_vram)

    # Second load & generate
    sd, img = generate_cat()

    # Final unload
    sd = None
    time.sleep(3)
    after_final_unload = log_vram("After Final Unload", baseline=start_vram)

    # Leak detection
    if after_final_unload != after_first_unload:
        leak = round(after_final_unload - after_first_unload, 2)
        raise Exception(f"Possible VRAM leak detected ({leak} MB)")
    else:
        print("No VRAM leak detected")
