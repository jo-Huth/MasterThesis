function [epe, outliers] = compute_epe(flowEst, flowGt, mask, syntheticTranslation)
% COMPUTE_EPE Mean Endpoint Error between estimated and ground truth flow
%
% EPE = mean( ||flow_est - flow_gt|| ) over valid pixels
%
% Inputs:
%   flow_est - opticalFlow object or [H,W,2] array (u,v)
%   flow_gt  - ground truth opticalFlow object or [H,W,2] array
%   mask     - binary mask for valid region
%
% Output:
%   epe      - scalar mean endpoint error

    % Extract velocity fields
    % flow_est: HxWx2 or opticalFlow
    Vx_est = flowEst.Vx;  Vy_est = flowEst.Vy;
    Vx_gt = flowGt.Vx; Vy_gt  = flowGt.Vy;
    
    validEst = (sqrt( (Vx_est).^2 + (Vy_est).^2 ) > 0.01); 
    validGt = (sqrt( (Vx_gt).^2 + (Vy_gt).^2 ) > 0.01);
    

    valid = validEst & validGt;
    
    epeMap = sqrt( (Vx_est - Vx_gt).^2 + (Vy_est - Vy_gt).^2 );
    
    epeMap(~mask) = 0;
    outliers = (epeMap > 3) & valid;
    outliers = sum(outliers)/sum(valid);
    % Mean over masked region
    epe = mean(epeMap(valid));
end
