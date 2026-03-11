function [coverage, validFLow] = compute_coverage(flow_est, mask, syntheticTranslation)
% COMPUTE_COVERAGE Fraction of valid flow vectors in masked region
%
% Inputs:
%   flow - opticalFlow object or [H,W,2] array
%   mask - binary mask for valid region
% Output:
%   coverage - fraction [0,1] of masked pixels with valid flow

    % Extract velocity fields
    Vx = flow_est.Vx; Vy = flow_est.Vy;
    mag = sqrt( (Vx).^2 + (Vy).^2 );
   
    validFLow = (mag > 0.01); 

    validFLow(~mask) = 0;
    sumValidFlow = sum(validFLow);
    sumMask = sum(mask);
    coverage = sumValidFlow / sumMask;
end
