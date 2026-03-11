function mask = apply_sky_mask(img, radius_ratio, treeMask)
% APPLY_SKY_MASK Create circular mask for ASI 1536×1536 fisheye images
%
% Usage:
%   mask = apply_sky_mask(img, radius_ratio)
%
% Inputs:
%   img           - Input image (1536×1536 or compatible square size)
%   radius_ratio  - Radius as fraction of image size (0.0-1.0), default 0.98
%
% Outputs:
%   mask          - Binary mask (true = sky region, false = non-sky)
%
% Description:
%   Removes black corners and equipment artifacts by creating a circular
%   mask centered on the image. Used for all-sky imager preprocessing.

[h, w, ~] = size(img);
assert(h == w, 'Image must be square');

center = h / 2;
radius = (h / 2) * radius_ratio;

% Create meshgrid
[xx, yy] = meshgrid(1:h, 1:h);

% Create circular mask
mask = ((xx - center).^2 + (yy - center).^2) <= radius^2;
mask = mask & treeMask;
end
