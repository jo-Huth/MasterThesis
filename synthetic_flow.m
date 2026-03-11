function [flowGT] = synthetic_flow(imgSize, translationUv)
% SYNTHETIC_GT Creates synthetic ground truth optical flow via warping.
%   [imgWarped, flowGT] = synthetic_gt(imgResized, img_size, translation_uv, mask)
%
% Inputs:
%   imgResized  - Input image (grayscale or RGB, H x W x C, double/single [0,1])
%   img_size    - [height, width] target size for flow and warped image
%   translation_uv - [u, v] translation pixels (e.g., [2, 1] for 2px right, 1px down)
%   mask        - Binary mask (H x W logical/double) for valid region (e.g., fisheye)
%
% Outputs:
%   imgWarped   - Warped image (img_size(1) x img_size(2) x C)
%   flowGT      - Ground truth flow (img_size(1) x img_size(2) x 2, u=x, v=y)
%
% Fits your stage_06_evaluation.m pipeline: save('flowgroundtruth.mat', 'flowGT'); [file:36]

uTrans = translationUv(1);
vTrans = translationUv(2);

[height, width] = deal(imgSize(1), imgSize(2));

% Create uniform translation flow field
[Y, X] = meshgrid(1:width, 1:height);
flowGT_v = zeros(height, width, 2);
flowGT_u = zeros(height, width, 2);
flowGT_u = uTrans * ones(height, width);  % Explicit uniform u
flowGT_v = vTrans * ones(height, width);  % Explicit uniform v

% % Optional: Mask warped image to match valid region
% imgWarped(repmat(~mask, [1, 1, size(imgResized,3)])) = 0;  % Black invalid pixels
% imgResized(repmat(~mask, [1, 1, size(imgResized,3)])) = 0;  % Black invalid pixels

flowGT = opticalFlow(flowGT_u, flowGT_v);



end