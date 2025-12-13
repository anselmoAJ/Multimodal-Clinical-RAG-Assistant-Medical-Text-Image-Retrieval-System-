# ==============================================================================
#                      ENVIRONMENT VERSION CHECK SCRIPT
# ==============================================================================
# This script prints the versions of all important libraries used in the project.
# ==============================================================================

import sys
import pkg_resources

def check_versions():
    """
    Checks and prints the versions of key libraries in the environment.
    """
    print("\n--- Checking Project Library Versions ---\n")

    # --- Python Version ---
    print(f"🐍 Python Version: {sys.version.split(' ')[0]}")

    libraries = [
        "torch",
        "torchvision",
        "transformers",
        "sentence-transformers",
        "chromadb",
        "Pillow",
        "tqdm",
        "loguru",
        "streamlit",
        "ollama",
        "numpy",
        "open_clip_torch" # In case it was installed
    ]

    print("\n--- Library Versions ---")
    for lib in libraries:
        try:
            version = pkg_resources.get_distribution(lib).version
            print(f"✅ {lib}: {version}")
        except pkg_resources.DistributionNotFound:
            print(f"❌ {lib}: Not Found")
        except Exception as e:
            print(f"⚠️ {lib}: Could not determine version ({e})")
            
    # --- CUDA Version (if torch is installed with GPU support) ---
    try:
        import torch
        if torch.cuda.is_available():
            print(f"\n⚡ CUDA Version (via PyTorch): {torch.version.cuda}")
            print(f"   GPU Detected: {torch.cuda.get_device_name(0)}")
        else:
            print("\n- No CUDA/GPU detected by PyTorch.")
    except ImportError:
        pass # PyTorch not found
    except Exception as e:
        print(f"\n⚠️ Could not check CUDA version. Error: {e}")

    print("\n--- Version Check Complete ---\n")


if __name__ == "__main__":
    check_versions()