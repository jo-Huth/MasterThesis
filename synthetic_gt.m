function synthetic_gt(imgResized, imgSize, translationUv, config, timestamp)
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
% Define affine transformation matrix from translation
tAffine = [1, 0, 0; 0, 1, 0; uTrans, vTrans, 1];

% Warp input image using the transformation
outputView = imref2d([height, width]);
imgWarped = imwarp(imgResized, affine2d(tAffine), 'OutputView', outputView, 'interp', 'linear');

% % Optional: Mask warped image to match valid region
% imgWarped(repmat(~mask, [1, 1, size(imgResized,3)])) = 0;  % Black invalid pixels
% imgResized(repmat(~mask, [1, 1, size(imgResized,3)])) = 0;  % Black invalid pixels

imageName = sprintf('%s_1.jpg', timestamp);
imagePath = strcat(config.synthImageDirectory,'\', imageName);  
imwrite(imgResized,fullfile(imagePath));

imageName = sprintf('%s_2.jpg', timestamp);
imagePath = strcat(config.synthImageDirectory,'\', imageName);  
imwrite(imgWarped,fullfile(imagePath));
% 


end