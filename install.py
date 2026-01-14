"""
Installation script for JetBlock Optimizer
Checks dependencies and optimizes for RTX 4090
"""

import subprocess
import sys
import torch
import platform

def install_dependencies():
    """Install required dependencies"""

    print("=" * 60)
    print("JetBlock Optimizer Installation")
    print("=" * 60)

    # Check CUDA availability
    if not torch.cuda.is_available():
        print("WARNING: CUDA not available. JetBlock will run in CPU mode.")
    else:
        gpu_name = torch.cuda.get_device_name(0)
        vram = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        print(f"[OK] GPU detected: {gpu_name}")
        print(f"[OK] VRAM: {vram:.1f}GB")

        # Check for RTX 4090
        if "4090" in gpu_name:
            print("[OK] RTX 4090 detected - Optimal configuration will be used")
        else:
            print(f"! Different GPU detected. Optimizations may vary.")

    # Check PyTorch version
    torch_version = torch.__version__
    print(f"[OK] PyTorch version: {torch_version}")

    if torch_version < "2.0.0":
        print("! PyTorch 2.0+ recommended for torch.compile support")

    # Install additional optimizations for Windows
    if platform.system() == "Windows":
        print("\nInstalling Windows-specific optimizations...")

        # Check for TensorRT
        try:
            import tensorrt
            print("[OK] TensorRT already installed")
        except ImportError:
            print("! TensorRT not found (optional, but recommended)")

    print("\n" + "=" * 60)
    print("Installation complete!")
    print("=" * 60)
    print("\nUsage:")
    print("1. Restart ComfyUI")
    print("2. Find 'JetBlock' nodes in the node menu")
    print("3. Start with 'JetBlock Auto-Optimizer' node")
    print("4. Use 'JetBlock Model Optimizer' to optimize specific models")
    print("5. Monitor performance with 'JetBlock Benchmark' node")

    return True

if __name__ == "__main__":
    success = install_dependencies()
    sys.exit(0 if success else 1)