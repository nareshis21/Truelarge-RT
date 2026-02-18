import os
import sys

# To ensure maximum download speed, we use hf_transfer
# This requires: pip install huggingface_hub hf_transfer
os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"

try:
    from huggingface_hub import hf_hub_download
except ImportError:
    print("Error: huggingface_hub not installed.")
    print("Please run: pip install huggingface_hub hf_transfer")
    sys.exit(1)

def download_model():
    repo_id = "bartowski/Qwen2.5-Coder-32B-Instruct-GGUF"
    filename = "Qwen2.5-Coder-32B-Instruct-Q4_K_M.gguf"
    
    # Get current script directory
    base_dir = os.path.dirname(os.path.abspath(__file__))
    
    print(f"Starting optimized download of {filename}...")
    print("Using hf_transfer for maximum speed (Storage bound).")
    
    try:
        path = hf_hub_download(
            repo_id=repo_id,
            filename=filename,
            local_dir=base_dir,
            local_dir_use_symlinks=False
        )
        print(f"\nSuccess! Model downloaded to: {path}")
        print("You can now transfer this file to your Android /sdcard/Download folder.")
    except Exception as e:
        print(f"\nDownload failed: {e}")

if __name__ == "__main__":
    download_model()
