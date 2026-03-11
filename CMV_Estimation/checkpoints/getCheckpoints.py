import urllib.request
import os

url = "https://jianghz.me/files/ezflow_ckpts/raft_kubric_step100k_v2.pth"
output_dir = "checkpoints"
output_file = os.path.join(output_dir, "raft_kubric.pth")

# Create checkpoints folder if it doesn't exist
os.makedirs(output_dir, exist_ok=True)

print(f"Downloading RAFT Kubric checkpoint (~300 MB)...")
print(f"URL: {url}")
print(f"Saving to: {output_file}")

try:
    urllib.request.urlretrieve(url, output_file)
    print(f"✓ Download complete: {output_file}")
except Exception as e:
    print(f"✗ Download failed: {e}")
