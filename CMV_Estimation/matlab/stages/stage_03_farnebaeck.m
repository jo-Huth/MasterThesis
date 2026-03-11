% stage_03_farnebaeck.m
% Computes Farnebäck optical flow on preprocessed images

%% RUN FARNEBÄCK

[timeFb, flowFB1, flowFB2] = run_farnebaeck(img1Norm, img2Norm, config);
%% SAVE RESULTS

filename = sprintf('flowFB_%s.mat', timestamp{1});
path =  strcat(config.resultsDir,'\flowFB\flowFB_', timestamp{1});
prepFile = fullfile(path, filename);

if ~exist(path, 'dir')
    mkdir(path);
end

save(prepFile, 'flowFB1','flowFB2', 'timeFb');
