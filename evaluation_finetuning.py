import torch
import cv2
import sys

import torch.nn.functional as F
import numpy as np
import torchvision.transforms.v2 as v2

from pathlib import Path
from torchvision.utils import flow_to_image  # For viz
from ezflow.models import build_model
from python_scripts.config_finetuning import Config
from python_scripts.dataloader_finetuning import get_dataloader, get_valdataloader

class ProxyMetrics:
    @staticmethod
    def synthetic_gt_epe(model, img1, device):  # Defined flow
        """
        Warp img1 → img2 with KNOWN flow → EPE vs model pred.
        """
        warp_flow=torch.tensor([5., 3.])
        B, C, H, W = img1.shape
        warp_flow = warp_flow.to(device).view(1, 2, 1, 1).repeat(B, 1, H, W)  # B2HW constant
        
        # Warp → img2
        img2_synth = ProxyMetrics.warp_image(img1, warp_flow, device)
        
        model.eval()
        with torch.no_grad():
            flow_pred = model(img1, img2_synth)['flow_preds'][-1]
            
            epe = torch.norm(flow_pred - warp_flow, dim=1).mean()
        
        return epe  # Return synth img2 for viz
    
    @staticmethod
    def warp_image(img, flow, device): 
        B, C, H, W = img.shape  
        
        # Meshgrid (device-aware)
        xx = torch.arange(0, W, device=device).view(1, -1).repeat(H, 1)
        yy = torch.arange(0, H, device=device).view(-1, 1).repeat(1, W)
        xx = xx.view(1, 1, H, W).repeat(B, 1, 1, 1)
        yy = yy.view(1, 1, H, W).repeat(B, 1, 1, 1)
        grid = torch.cat((xx, yy), 1).float()
        
        # Add flow
        vgrid = grid + flow
        
        # Scale [-1,1]
        vgrid[:, 0, :, :] = 2.0 * vgrid[:, 0, :, :] / max(W - 1, 1) - 1.0
        vgrid[:, 1, :, :] = 2.0 * vgrid[:, 1, :, :] / max(H - 1, 1) - 1.0
        
        vgrid = vgrid.permute(0, 2, 3, 1)  # B H W 2 
        
        output = F.grid_sample(img, vgrid, align_corners=True, padding_mode='zeros')
        return output

    @staticmethod
    def forward_warp_validity(flow, device, thresh=0.001):
        dummy = torch.ones(flow.shape[0], 1, flow.shape[2], flow.shape[3], device=device)
        warped = ProxyMetrics.warp_image(dummy, flow, device)
        return (warped > thresh).all(dim=1).float() 

    @staticmethod
    def fbce(model, img1, img2, device,valid_thresh=0.001, clamp_flow=10.0):
        flow_fw = model(img1, img2)['flow_preds'][-1]
        flow_bw = model(img2, img1)['flow_preds'][-1]
        
        # CLAMP large flows before warping (preserve parallax)
        flow_fw_clamp = torch.clamp(flow_fw, -clamp_flow, clamp_flow)
        flow_bw_clamp = torch.clamp(flow_bw, -clamp_flow, clamp_flow)
        
        # RELAXED validity (less strict)
        valid_fw = ProxyMetrics.forward_warp_validity(flow_fw_clamp, device, thresh=valid_thresh)
        valid_bw = ProxyMetrics.forward_warp_validity(flow_bw_clamp, device, thresh=valid_thresh)
        valid = valid_fw * valid_bw
        
        # Warp clamped flows
        flow_bw_warped = ProxyMetrics.warp_image(flow_bw_clamp, flow_fw_clamp, device)
        inconsistency = torch.norm(flow_fw_clamp + flow_bw_warped, dim=1)
        
        # Add small penalty everywhere (encourages consistency, doesn't erase parallax)
        fbce_val = torch.sum(inconsistency * valid) / (torch.sum(valid) + 1e-8)
        fbce_val += 0.01 * torch.norm(flow_fw, dim=1).mean()  # Global consistency nudge
        
        return fbce_val

    @staticmethod
    def proxy_epe(model, img1, img2, device):
       
        flow_pred = model(img1, img2)['flow_preds'][-1]  # First batch
        
        # Farneback on first pair
        img1_np = img1[0].cpu().permute(1,2,0).numpy()
        img2_np = img2[0].cpu().permute(1,2,0).numpy()
        gray1 = cv2.cvtColor((img1_np*255).astype(np.uint8), cv2.COLOR_RGB2GRAY)
        gray2 = cv2.cvtColor((img2_np*255).astype(np.uint8), cv2.COLOR_RGB2GRAY)
        prms = dict(pyr_scale=0.5, levels=3, winsize=15, iterations=3, 
                   poly_n=5, poly_sigma=1.2, flags=0)
        flow_gt_np = cv2.calcOpticalFlowFarneback(gray1, gray2, None, **prms)
        flow_gt = torch.from_numpy(flow_gt_np).permute(2, 0, 1).float().to(device)
        
        epe = torch.norm(flow_pred - flow_gt, dim=0).mean()
        return epe

# Validation loop (your DataLoader batches automatically)
def validate(model, valloader, device):
    fbces, epes, syn_epes, outlier_details = [], [], [], []
    model.eval()
    with torch.no_grad():
        for batch_idx, batch_data in enumerate(valloader):  # batch = (img1_BCHW, img2_BCHW)
            
            img1_b = batch_data[0]
            img2_b = batch_data[1]
            
            # Move to device
            img1_b = img1_b.to(device)
            img2_b = img2_b.to(device)
    
            toImg = v2.ToImage()
            img1 = toImg(img1_b/255.0)
            img2 = toImg(img2_b/255.0)
    
            toDtype = v2.ToDtype(torch.float32, scale=True)
            img1 = toDtype(img1)
            img2 = toDtype(img2)
    
            normalize = v2.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])   
    
            img1 = normalize(img1)
            img2 = normalize(img2)
            
            fbce = ProxyMetrics.fbce(model, img1, img2, device)
            epe = ProxyMetrics.proxy_epe(model, img1, img2, device)
            syn_epe = ProxyMetrics.synthetic_gt_epe(model, img1, device)

            fbces.append(fbce.item())
            epes.append(epe.item())
            syn_epes.append(syn_epe.item())

    return {'FBCE': np.mean(fbces), 'Proxy-EPE': np.mean(epes), 'Synth-EPE': np.mean(syn_epes)}

if __name__ == "__main__":
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Create valloader
    try:
        valloader = get_valdataloader(
            config=Config
        )
        print(f"✓ ValDataLoader created with {len(valloader)}\n")
    except Exception as e:
        print(f"\n❌ ERROR creating valloader: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

    # Load Kubric RAFT model
    try:
        model = build_model("RAFT", default=True)
        # Load checkpoint (start from pretrained Kubric or previous stage)
        checkpoint_path = Config.PRETRAINED_RAFT
        
        if checkpoint_path.exists():
            checkpoint = torch.load(checkpoint_path, map_location=device)
            print(f"Loading checkpoint: {checkpoint_path}")
        else:
            print(f"⚠️  no checkpoints loaded\n")
        
        model = model.to(device)

    except Exception as e:
        print(f"\n❌ ERROR loading model: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
    validate.QUIET = True
    metrics = validate(model, valloader, device)
    print(metrics)
