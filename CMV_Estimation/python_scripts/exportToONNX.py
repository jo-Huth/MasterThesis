import os
import torch
from ezflow.models import build_model

CHECKPOINT_DIR = "checkpoints"
OUTPUT_ONNX = "results/raft_kubric_pretrained.onnx"
IMG_H, IMG_W = 512, 512

device = "cuda" if torch.cuda.is_available() else "cpu"

# Load RAFT
model = build_model("RAFT", default=True).to(device)
model.eval()

# Dummy inputs
dummy_img1 = torch.randn(1, 3, IMG_H, IMG_W, device=device)
dummy_img2 = torch.randn(1, 3, IMG_H, IMG_W, device=device)

# Export with opset 11 (MATLAB compatible) and no external data
torch.onnx.export(
    model,
    (dummy_img1, dummy_img2),
    OUTPUT_ONNX,
    input_names=["image1", "image2"],
    output_names=["flow"],
    opset_version=11,
    do_constant_folding=True,
    verbose=False,
    use_external_data_format=False
)

print(f"✅ Exported to {OUTPUT_ONNX}")


