function [timeFb, flow1,flow2] = run_farnebaeck(img1, img2, config)
% RUN_FARNEBAECK Estimate Farnebäck optical flow
%
% Usage:
%   flow = run_farnebaeck(img1, img2, config)
%
% Inputs:
%   img1, img2  - Normalized grayscale images
%   config      - Configuration struct with Farnebäck parameters
%
% Outputs:
%   flow        - opticalFlow object with dense flow field
%
% Description:
%   Computes dense Farnebäck optical flow using MATLAB's opticalFlowFarneback.
%   Parameters optimized for all-sky imager cloud motion estimation.

% Create Farnebäck object with parameters from config
opticFB = opticalFlowFarneback(...
    'NeighborhoodSize', config.fbWindowSize, ...
    'NumPyramidLevels', config.fbPyramidLevels);

% Initialize with first image
tic;
flow1= estimateFlow(opticFB, img1);

% Estimate flow on second image
flow2 = estimateFlow(opticFB, img2);
timeFb = toc;
reset(opticFB)
end
