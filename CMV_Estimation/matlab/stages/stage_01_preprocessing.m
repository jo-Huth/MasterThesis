% stage_01_preprocessing.m
% Applies preprocessing: distortion correction, circular mask and normalization

%% Load masks
load('mask16030.mat');

%% Normalize Images
IMG_H = 512; IMG_W = 512;

% Resize to RAFT input size (no masking, no normalization)
img1Resized = imresize(img1, [IMG_H, IMG_W]);
img2Resized = imresize(img2, [IMG_H, IMG_W]);

maskResized = imresize(double(mask16030), [IMG_H, IMG_W]) > 0.5;  % Logical mask

img1Norm = normalize_image(img1Resized,maskResized);
img2Norm = normalize_image(img2Resized,maskResized);


%% Save Processed Data

filename = sprintf('preprocessing_%s.mat', timestamp{1});
path =  strcat(config.dataProcessedDir,'\preprocessing_', timestamp{1});
prepFile = fullfile(path, filename);  

if ~exist(path, 'dir')
    mkdir(path);
end

save(prepFile, 'img1Norm', 'img2Norm');