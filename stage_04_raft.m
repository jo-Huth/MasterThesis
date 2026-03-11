% stage_04_raft.m
% Computes RAFT optical flow via Matlab and PyTorch

%% RUN RAFT Matlab

[timeRaftML, flowRaftML1, flowRaftML2] = run_raft(img1Resized, img2Resized);
  
%% RUN RAFT PyTorch
% 
% pyrunfile('./python_scripts/raft_inference.py', '--checkpoints=checkpoints/raft_kubric.pth');
% 
% %Load flow field
% flowPath = sprintf('results/flowRaftPy/raft_kubric/flowRaftPy_%s/flowRaftPy_%s.mat', timestamp{1}, timestamp{1});
% flow = load(flowPath);
% flowRaftPy_kubric = opticalFlow(flow.Vx, flow.Vy);

% % Load flow field
% flowPath = sprintf('results/flowRaftPy/raft_finetune_300/flowRaftPy_%s/flowRaftPy_%s.mat', timestamp{1}, timestamp{1});
% flow = load(flowPath);
% flowRaftPy_300 = opticalFlow(flow.Vx, flow.Vy);

flowPath = sprintf('results/flowRaftPy/raft_finetune_300_best_metric/flowRaftPy_%s/flowRaftPy_%s.mat', timestamp{1}, timestamp{1});
flow = load(flowPath);
flowRaftPy_300 = opticalFlow(flow.Vx, flow.Vy);

% %Load flow field
% flowPath = sprintf('results/flowRaftPy/raft_finetune_600/flowRaftPy_%s/flowRaftPy_%s.mat', timestamp{1}, timestamp{1});
% flow = load(flowPath);
% flowRaftPy_600 = opticalFlow(flow.Vx, flow.Vy);

flowPath = sprintf('results/flowRaftPy/raft_finetune_600_best_metric/flowRaftPy_%s/flowRaftPy_%s.mat', timestamp{1}, timestamp{1});
flow = load(flowPath);
flowRaftPy_600 = opticalFlow(flow.Vx, flow.Vy);

% Load flow field
flowPath = sprintf('results/flowRaftPy/raft_finetune_900_best_metric/flowRaftPy_%s/flowRaftPy_%s.mat', timestamp{1}, timestamp{1});
flow = load(flowPath);
flowRaftPy_900 = opticalFlow(flow.Vx, flow.Vy);

% flowPath = sprintf('results/flowRaftPy/raft_finetune_900/flowRaftPy_%s/flowRaftPy_%s.mat', timestamp{1}, timestamp{1});
% flow = load(flowPath);
% flowRaftPy_900 = opticalFlow(flow.Vx, flow.Vy);

% %Load flow field
% flowPath = sprintf('results/flowRaftPy/raft_finetune_2000/flowRaftPy_%s/flowRaftPy_%s.mat', timestamp{1}, timestamp{1});
% flow = load(flowPath);
% flowRaftPy_2000 = opticalFlow(flow.Vx, flow.Vy);

flowPath = sprintf('results/flowRaftPy/raft_finetune_2000_best_metric/flowRaftPy_%s/flowRaftPy_%s.mat', timestamp{1}, timestamp{1});
flow = load(flowPath);
flowRaftPy_2000= opticalFlow(flow.Vx, flow.Vy);

%% SAVE RESULTS

filename = sprintf('flowRaftML_%s.mat', timestamp{1});
path =  strcat(config.resultsDir,'\flowRaftML\flowRaftML_', timestamp{1});
prepFile = fullfile(path, filename);

if ~exist(path, 'dir')
    mkdir(path);
end

save(prepFile, 'flowRaftML1','flowRaftML2', 'timeRaftML');
    
