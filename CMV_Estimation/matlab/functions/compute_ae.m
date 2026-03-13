function ae = compute_ae(flow_est, flow_gt, mask)
% COMPUTE_AE Mean Angular Error between estimated and ground truth flow
%
% AE = mean( acos( dot(v_est, v_gt) / (|v_est| |v_gt|) ) ) over valid pixels
%
% Inputs: same as compute_epe
% Output: ae - scalar mean angular error (radians)

    % Extract velocity fields
    if isa(flow_est, 'opticalFlow')
        Vx_est = flow_est.Vx; Vy_est = flow_est.Vy;
        Vx_gt = flow_gt.Vx; Vy_gt  = flow_gt.Vy;
    else
        Vx_est = flow_est(:,:,1); Vy_est = flow_est(:,:,2);
        Vx_gt  = flow_gt (:,:,1); Vy_gt  = flow_gt (:,:,2);
    end

    % Avoid division by zero
    validEst = (sqrt( (Vx_est).^2 + (Vy_est).^2 ) > 0.01); 
    validGt = (sqrt( (Vx_gt).^2 + (Vy_gt).^2 ) > 0.01);
    
    valid = validEst & validGt;

    mag_est = sqrt(Vx_est.^2 + Vy_est.^2 + 1);
    mag_gt  = sqrt(Vx_gt.^2 + Vy_gt.^2 + 1);
    % Cosine similarity
    cos_sim = (Vx_est.*Vx_gt + Vy_est.*Vy_gt + 1) ./ (mag_est .* mag_gt);
    cos_sim = max(-1, min(cos_sim, 1));  % Clamp to [-1, 1]
    
    % Angular error
    ae_map = acos(cos_sim);

    ae_map(~mask) = 0;

    ae = mean(ae_map(valid));
    ae = ae * 180/pi;
end
