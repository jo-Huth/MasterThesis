function epe_map = compute_epe_map(flow_est, flow_gt, mask)
% COMPUTE_EPE_MAP Per-pixel Endpoint Error map
% Output: epe_map - [H,W] error map (only valid region)
    
    Vx_est = flow_est.Vx;  Vy_est = flow_est.Vy;
    Vx_gt = flow_gt.Vx; Vy_gt  = flow_gt.Vy;
    
    epe_map = sqrt( (Vx_est - Vx_gt).^2 + (Vy_est - Vy_gt).^2 );
    epe_map(~mask) = 0;
end