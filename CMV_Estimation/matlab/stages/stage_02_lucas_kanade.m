% stage_02_lucas_kanade.m
% Computes Lucas-Kanade optical flow on preprocessed images

%% RUN LUCAS-KANADE

[timeLk, flowLK1, flowLK2] = run_lucas_kanade(img1Norm, img2Norm, config);

%% SAVE RESULTS

filename = sprintf('flowLK_%s.mat', timestamp{1});
path =  strcat(config.resultsDir,'\flowLK\flowLK_', timestamp{1});
prepFile = fullfile(path, filename);

if ~exist(path, 'dir')
    mkdir(path);
end

save(prepFile, 'flowLK1','flowLK2', 'timeLk');