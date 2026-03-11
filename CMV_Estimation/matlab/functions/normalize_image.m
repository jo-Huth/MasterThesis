function imgNorm = normalize_image(img, mask)
% NORMALIZE_IMAGE Zero-mean, unit-variance normalization
%
% Usage:
%   img_norm = normalize_image(img, mask)
%
% Inputs:
%   img       - RGB image
%   mask      - Binary mask defining valid region
%
% Outputs:
%   img_norm  - Normalized image
%
% Description:
%   Normalizes image intensity within the masked region to have zero mean
%   and unit variance. Improves robustness to illumination variations.
% 


% [r, g, b] = imsplit(img);        % r,g,b are 2‑D matrices (one channel each)
% 
% % Manipulate r,g,b channels
% rEnh = im2double(r) * 1;
% rEnh = min(rEnh,1);
% 
% bEnh = im2double(b) * 1;
% bEnh = min(bEnh,1);
% 
% gEnh = im2double(r) * 1.2;
% gEnh = min(gEnh,1);
% 
% imgRedEnh = cat(3, rEnh, gEnh, bEnh);
% I_hsv = rgb2hsv(imgRedEnh);
% 
% H = I_hsv(:,:,1); S = I_hsv(:,:,2); V = I_hsv(:,:,3);
% 
% 
% S= min(S* 1.5, 1);   % 1.4 = saturation gain
% 
% I_hsv(:,:,2) = S;
% I_hsv(:,:,3) = V;
% I_enh = hsv2rgb(I_hsv);  

% 3) Grayscale from enhanced RGB (for optical flow)
imgNorm = rgb2gray(img); 
imgNorm = im2double(imgNorm);

mu = mean(imgNorm(:)) - 0.05;              % Scalar global mean
sigma = 0.5;            % Scalar global std
sigma(sigma == 0) = 1;          % Avoid div-by-zero
imgNorm = (imgNorm - mu) / sigma;
imgNorm(~mask) = 0;
