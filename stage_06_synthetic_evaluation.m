% stage_06_evaluation.m
% Computes EPE, AE, coverage metrics comparing estimated flows to ground truth
% Fits seamlessly into your existing modular pipeline
if idx == 1
    fprintf('\n--- STAGE 06: EVALUATION METRICS ---\n');
end
%% === LOAD ALL FLOW RESULTS ===

msg = sprintf('Loading flow results %1i/200...', idx); %Don't forget this semicolon
fprintf([reverseStr, msg]);
reverseStr = repmat(sprintf('\b'), 1, length(msg));

%Load flow field
flowPath = sprintf('results/synFlow/flowRaftPy/raft_finetune_300/flowRaftPy_%s/flowRaftPy_%s.mat', timestamp{1}, timestamp{1});
flow = load(flowPath);
synFlowRaftPy_300 = opticalFlow(flow.Vx, flow.Vy);

%Load flow field
flowPath = sprintf('results/synFlow/flowRaftPy/raft_finetune_600/flowRaftPy_%s/flowRaftPy_%s.mat', timestamp{1}, timestamp{1});
flow = load(flowPath);
synFlowRaftPy_600 = opticalFlow(flow.Vx, flow.Vy);

flowPath = sprintf('results/synFlow/flowRaftPy/raft_finetune_900/flowRaftPy_%s/flowRaftPy_%s.mat', timestamp{1}, timestamp{1});
flow = load(flowPath);
synFlowRaftPy_900 = opticalFlow(flow.Vx, flow.Vy);

%Load flow field
flowPath = sprintf('results/synFlow/flowRaftPy/raft_finetune_2000/flowRaftPy_%s/flowRaftPy_%s.mat', timestamp{1}, timestamp{1});
flow = load(flowPath);
synFlowRaftPy_2000 = opticalFlow(flow.Vx, flow.Vy);

[timeSynLk, synFlowLk1, synFlowLK2] = run_lucas_kanade(img1Norm, img2Norm, config);
[timeSynFb, synFlowFb1, synFlowFB2] = run_farnebaeck(img1Norm, img2Norm, config);
[timeSnyRaftMl, synFlowRaftMl1, synFlowRaftMl2] = run_raft(img1Resized, img2Resized);

maskResized = imresize(double(mask16030), [IMG_H, IMG_W]) > 0.5;
maskResized = apply_sky_mask(img2Norm, 0.85, maskResized);

% synFlowLK2 = mask_flow(synFlowLK2, maskResized);
synFlowFB2 = mask_flow(synFlowFB2, maskResized);
synFlowRaftMl2 = mask_flow(synFlowRaftMl2, maskResized);
synFlowRaftPy_300 = mask_flow(synFlowRaftPy_300, maskResized);
synFlowRaftPy_600 = mask_flow(synFlowRaftPy_600, maskResized);
synFlowRaftPy_900 = mask_flow(synFlowRaftPy_900, maskResized);
synFlowRaftPy_2000 = mask_flow(synFlowRaftPy_2000, maskResized);
flowSynGt = mask_flow(flowSynGt, maskResized);

img1Norm(~maskResized) = 0;
img2Norm(~maskResized) = 0;
img1Resized(repmat(~maskResized, [1, 1, size(img1Resized,3)])) = 0;  % Black invalid pixels
img2Resized(repmat(~maskResized, [1, 1, size(img2Resized,3)])) = 0;  % Black invalid pixels
%% === COMPUTE METRICS Synthetic Flow ===

% Endpoint Error (EPE)
[synMetrics.LK.EPE(idx), synMetrics.LK.outliers(idx)]   = compute_epe(synFlowLK2,  flowSynGt, maskResized, syntheticTranslationMag);
[synMetrics.FB.EPE(idx), synMetrics.FB.outliers(idx)]   = compute_epe(synFlowFB2,  flowSynGt, maskResized, syntheticTranslationMag);
[synMetrics.RAFT.EPE(idx), synMetrics.RAFT.outliers(idx)] = compute_epe(synFlowRaftMl2, flowSynGt, maskResized, syntheticTranslationMag);
[synMetrics.RAFT_300.EPE(idx), synMetrics.RAFT_300.outliers(idx)] = compute_epe(synFlowRaftPy_300, flowSynGt, maskResized, syntheticTranslationMag);
[synMetrics.RAFT_600.EPE(idx), synMetrics.RAFT_600.outliers(idx)] = compute_epe(synFlowRaftPy_600, flowSynGt, maskResized, syntheticTranslationMag);
[synMetrics.RAFT_900.EPE(idx), synMetrics.RAFT_900.outliers(idx)] = compute_epe(synFlowRaftPy_900, flowSynGt, maskResized, syntheticTranslationMag);
[synMetrics.RAFT_2000.EPE(idx), synMetrics.RAFT_2000.outliers(idx)] = compute_epe(synFlowRaftPy_2000, flowSynGt, maskResized, syntheticTranslationMag);

% Angular Error (AE)
synMetrics.LK.AE(idx)    = compute_ae(synFlowLK2,  flowSynGt, maskResized, syntheticTranslationMag);
synMetrics.FB.AE(idx)     = compute_ae(synFlowFB2,  flowSynGt, maskResized, syntheticTranslationMag);
synMetrics.RAFT.AE(idx)   = compute_ae(synFlowRaftMl2, flowSynGt, maskResized, syntheticTranslationMag);
synMetrics.RAFT_300.AE(idx)   = compute_ae(synFlowRaftPy_300, flowSynGt, maskResized, syntheticTranslationMag);
synMetrics.RAFT_600.AE(idx)   = compute_ae(synFlowRaftPy_600, flowSynGt, maskResized, syntheticTranslationMag);
synMetrics.RAFT_900.AE(idx)   = compute_ae(synFlowRaftPy_900, flowSynGt, maskResized, syntheticTranslationMag);
synMetrics.RAFT_2000.AE(idx)   = compute_ae(synFlowRaftPy_2000, flowSynGt, maskResized, syntheticTranslationMag);


% Coverage
synMetrics.LK.coverage(idx)   = compute_coverage(synFlowLK2,  maskResized, syntheticTranslationMag);
synMetrics.FB.coverage(idx)   = compute_coverage(synFlowFB2,  maskResized, syntheticTranslationMag);
synMetrics.RAFT.coverage(idx) = compute_coverage(synFlowRaftMl2, maskResized, syntheticTranslationMag);
synMetrics.RAFT_300.coverage(idx) = compute_coverage(synFlowRaftPy_300, maskResized, syntheticTranslationMag);
synMetrics.RAFT_600.coverage(idx) = compute_coverage(synFlowRaftPy_600, maskResized, syntheticTranslationMag);
synMetrics.RAFT_900.coverage(idx) = compute_coverage(synFlowRaftPy_900, maskResized, syntheticTranslationMag);
synMetrics.RAFT_2000.coverage(idx) = compute_coverage(synFlowRaftPy_2000, maskResized, syntheticTranslationMag);

raftEPE = cat(5, synMetrics.RAFT_300.EPE, synMetrics.RAFT_600.EPE, synMetrics.RAFT_900.EPE, synMetrics.RAFT_2000.EPE, synMetrics.RAFT.EPE);
maxEPE_RAFT = prctile(raftEPE(:), 99); 

EPE = cat(2, synMetrics.FB.EPE, synMetrics.LK.EPE);
maxEPE = prctile(EPE(:), 99);
% Runtime
synMetrics.LK.runtime(idx)   = timeSynLk;
synMetrics.FB.runtime(idx)    = timeSynFb;
synMetrics.RAFT.runtime(idx)  = timeSnyRaftMl;


%% === DISPLAY RESULTS TABLE ===
if idx == 200
    fprintf('\n=== EVALUATION RESULTS ===\n');
    fprintf('current Translation Magnitude: %.3f \n', syntheticTranslationMag)
    fprintf('current Translation Angle: %.3f deg %.3f rad\n', syntheticTranslationAngle, syntheticTranslationAngle/180*pi)
    fprintf('| Method       | EPE  | outliers  | AE   | Coverage | Runtime (s) |\n');
    fprintf('|--------------|------|-----------|------|----------|-------------|\n');
    fprintf('| LK           | %.3f | %.3f | %.3f | %.1f%%   | %.3f        |\n', ...
        mean(synMetrics.LK.EPE), mean(synMetrics.LK.outliers), mean(synMetrics.LK.AE), mean(synMetrics.LK.coverage*100), mean(synMetrics.LK.runtime));
    fprintf('| Farnebäck    | %.3f | %.3f | %.3f | %.1f%%   | %.3f        |\n', ...
        mean(synMetrics.FB.EPE), mean(synMetrics.FB.outliers), mean(synMetrics.FB.AE), mean(synMetrics.FB.coverage*100), mean(synMetrics.FB.runtime));
    fprintf('| RAFT         | %.3f | %.3f | %.3f | %.1f%%   | %.3f        |\n', ...
        mean(synMetrics.RAFT.EPE), mean(synMetrics.RAFT.outliers), mean(synMetrics.RAFT.AE), mean(synMetrics.RAFT.coverage*100), mean(synMetrics.RAFT.runtime));
    fprintf('| RAFT 300     | %.3f | %.3f | %.3f | %.1f%%   | %.3f        |\n', ...
        mean(synMetrics.RAFT_300.EPE), mean(synMetrics.RAFT_300.outliers),mean( synMetrics.RAFT_300.AE), mean(synMetrics.RAFT_300.coverage*100), mean(synMetrics.RAFT_300.runtime));
    fprintf('| RAFT 600     | %.3f | %.3f | %.3f | %.1f%%   | %.3f        |\n', ...
        mean(synMetrics.RAFT_600.EPE), mean(synMetrics.RAFT_600.outliers), mean(synMetrics.RAFT_600.AE), mean(synMetrics.RAFT_600.coverage*100), mean(synMetrics.RAFT_600.runtime));
    fprintf('| RAFT 900     | %.3f | %.3f | %.3f | %.1f%%   | %.3f        |\n', ...
        mean(synMetrics.RAFT_900.EPE), mean(synMetrics.RAFT_900.outliers), mean(synMetrics.RAFT_900.AE), mean(synMetrics.RAFT_900.coverage*100), mean(synMetrics.RAFT_900.runtime));
    fprintf('| RAFT 1400    | %.3f | %.3f | %.3f | %.1f%%   | %.3f        |\n', ...
        mean(synMetrics.RAFT_2000.EPE), mean(synMetrics.RAFT_2000.outliers), mean(synMetrics.RAFT_2000.AE), mean(synMetrics.RAFT_2000.coverage*100), mean(synMetrics.RAFT_2000.runtime));

    %% === SAVE METRICS ===
    fprintf('Saving metrics...\n');
    filename = sprintf('synEvaluation_metrics_average.mat');
    path =  strcat(config.resultsDir,'\evaluation');
    metrics_file = fullfile(path, filename);
    
    if ~exist(path, 'dir')
        mkdir(path);
    end
    
    save(metrics_file, 'synMetrics');
    fprintf('✓ Saved: %s\n', metrics_file);
end
%% === VISUALIZATION ===
figure('Position', [100 100 1400 400]);

% EPE error map
subplot(1,3,1);
epe_lk = compute_epe_map(synFlowLK2, flowSynGt, maskResized);
imagesc(epe_lk); clim([0 maxEPE]); colorbar; colormap('jet'); axis image;
title('LK EPE Error Map');

subplot(1,3,2);
epe_fb = compute_epe_map(synFlowFB2, flowSynGt, maskResized);
imagesc(epe_fb); clim([0 maxEPE]); colorbar; colormap('jet'); axis image;
title('Farnebäck EPE Error Map');

subplot(1,3,3);
epe_ml = compute_epe_map(synFlowRaftMl2, flowSynGt, maskResized);
imagesc(epe_ml); clim([0 maxEPE_RAFT]); colorbar; colormap('jet'); axis image;
title('RAFT EPE Error Map');

sgtitle('Endpoint Error Maps (Red = High Error)', 'FontSize', 14);

filename = sprintf('synEvaluation_matlab_flow_%s.fig', timestamp{1});
path =  strcat(config.resultsDir,'\evaluation');
fig = fullfile(path, filename);
savefig(fig)

% Raft
figure('Position', [100 100 1400 400]);

% EPE error map 

subplot(1,4,1);
epe_300 = compute_epe_map(synFlowRaftPy_300, flowSynGt, maskResized);
imagesc(epe_300); clim([0 maxEPE_RAFT]); colorbar; colormap('jet'); axis image;
title('300 EPE Error Map');

subplot(1,4,2);
epe_600 = compute_epe_map(synFlowRaftPy_600, flowSynGt, maskResized);
imagesc(epe_600); clim([0 maxEPE_RAFT]); colorbar; colormap('jet'); axis image;
title('600 EPE Error Map');

subplot(1,4,3);
epe_900 = compute_epe_map(synFlowRaftPy_900, flowSynGt, maskResized);
imagesc(epe_900); clim([0 maxEPE_RAFT]); colorbar; colormap('jet'); axis image;
title('900 EPE Error Map');

subplot(1,4,4);
epe_2000 = compute_epe_map(synFlowRaftPy_2000, flowSynGt, maskResized);
imagesc(epe_2000); clim([0 maxEPE_RAFT]); colorbar; colormap('jet'); axis image;
title('1600 EPE Error Map');

sgtitle('Endpoint Error Maps (Red = High Error)', 'FontSize', 14);

filename = sprintf('synEvaluation_python_flow_%s.fig', timestamp{1});
path =  strcat(config.resultsDir,'\evaluation');
fig = fullfile(path, filename);
savefig(fig)


figure; imshow(img2Resized); hold on;
plot(synFlowFB2, 'DecimationFactor',[5 5],'ScaleFactor',1,'Color','blue');
title('synFlowFB2');

figure; imshow(img2Resized); hold on;
plot(synFlowLK2, 'DecimationFactor',[5 5],'ScaleFactor',1,'Color','blue');
title('synFlowLK2');

figure; imshow(img1Resized); hold on;
plot(synFlowRaftMl2, 'DecimationFactor',[5 5],'ScaleFactor',0.5,'Color','blue');
title('RAFT Matlab Flow');

figure; imshow(img1Resized); hold on;
plot(flowSynGt, 'DecimationFactor',[5 5],'ScaleFactor',0.8,'Color','blue');
title('Synthetic Flow');

figure; imshow(img1Resized); hold on;
plot(synFlowRaftPy_300, 'DecimationFactor',[5 5],'ScaleFactor',0.5,'Color','blue');
title('Flow Raft 300');

figure; imshow(img1Resized); hold on;
plot(synFlowRaftPy_600, 'DecimationFactor',[5 5],'ScaleFactor',0.5,'Color','blue');
title('Flow Raft 600');

figure; imshow(img1Resized); hold on;
plot(synFlowRaftPy_900, 'DecimationFactor',[5 5],'ScaleFactor',0.5,'Color','blue');
title('Flow Raft 900');

figure; imshow(img1Resized); hold on;
plot(synFlowRaftPy_2000, 'DecimationFactor',[5 5],'ScaleFactor',0.8,'Color','blue');
title('Flow Raft 1600');

drawnow