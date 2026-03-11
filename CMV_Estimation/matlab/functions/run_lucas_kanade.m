function [timeLk, flow1, flow2] = run_lucas_kanade(img1, img2, config)
% RUN_LUCAS_KANADE Estimate Lucas-Kanade optical flow
%
% Usage:
%   flow = run_lucas_kanade(img1, img2, config)
%
% Inputs:
%   img1, img2  - Normalized grayscale images
%   config      - Configuration struct with LK parameters
%
% Outputs:
%   flow        - opticalFlow object with estimated flow field
%
% Description:
%   Computes sparse Lucas-Kanade optical flow using MATLAB's opticalFlowLK.
%   Applies pyramidal refinement and Shi-Tomasi feature detection.

% Create LK object with parameters from config
opticLK = opticalFlowLK('NoiseThreshold', config.lkNoiseThreshold);

% Initialize with first image
tic;
flow1 = estimateFlow(opticLK, img1);

% Estimate flow on second image
flow2 = estimateFlow(opticLK, img2);
timeLk = toc;
reset(opticLK)
end
