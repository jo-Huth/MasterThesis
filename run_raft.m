function [timeRaftML, flow1, flow2] = run_raft(img1Resized, img2Resized)
% RUN_RAFT Estimate optical flow using MATLAB opticalFlowRAFT
%
% Usage:
%   flow = run_raft(img1, img2)
%
% Inputs:
%   img1, img2  - RGB images [H,W,3], double [0,1] range
%
% Outputs:
%   flow        - opticalFlow object with .Vx, .Vy fields
%
% Description:
%   Uses MATLAB's built-in opticalFlowRAFT (no Python required).
%   Exact preprocessing matches your Python pipeline:
%   - RGB input ✓
%   - 512x512 resize only ✓
%   - No fisheye mask ✓  
%   - No normalization ✓

    % RAFT model (persistent - create once per session) 
    persistent raft_model;
    if isempty(raft_model)
        fprintf('[RUN_RAFT] Initializing opticalFlowRAFT...\n');
        raft_model = opticalFlowRAFT; 
    end
    
    % img1(~mask) = 0;
    % img2(~mask) = 0;
    % figure; imshow(img2);
    % Preprocess (EXACTLY matches your Python ASIDataset._preprocess)
    

    % CLAMP after resize (fixes negative values)
    % img1_resized = max(0, min(1, img1_resized));
    % img2_resized = max(0, min(1, img2_resized));
    % Ensure double [0,1] range


    % PREDICT 
    tic;
    flowRaft1 = estimateFlow(raft_model, img1Resized);
    
    flowRaft2 = estimateFlow(raft_model, img2Resized);
    timeRaftML = toc;
    % Return opticalFlow object (consistent with MATLAB ecosystem)
    flow1 = flowRaft1;
    flow2 = flowRaft2;
    
    reset(raft_model)

end