% stage_06_evaluation.m
% Computes EPE, AE, coverage metrics comparing estimated flows to ground truth
% Fits seamlessly into your existing modular pipeline

%% === COMPUTE METRICS ===
metrics = struct();

% Endpoint Error (EPE)
[metrics.RAFT.EPE_LK, metrics.RAFT.outliersLK]   = compute_epe(flowRaftML2,  flowLK2, maskResized);
[metrics.RAFT_300.EPE_LK , metrics.RAFT_300.outliersLK] = compute_epe(flowRaftPy_300, flowLK2, maskResized);
[metrics.RAFT_600.EPE_LK , metrics.RAFT_600.outliersLK] = compute_epe(flowRaftPy_600, flowLK2, maskResized);
[metrics.RAFT_900.EPE_LK , metrics.RAFT_900.outliersLK] = compute_epe(flowRaftPy_900, flowLK2, maskResized);
[metrics.RAFT_2000.EPE_LK , metrics.RAFT_2000.outliersLK] = compute_epe(flowRaftPy_2000, flowLK2, maskResized);

[metrics.RAFT.EPE_FB, metrics.RAFT.outliersFB]   = compute_epe(flowRaftML2,  flowFB2, maskResized);
[metrics.RAFT_300.EPE_FB , metrics.RAFT_300.outliersFB] = compute_epe(flowRaftPy_300, flowFB2, maskResized);
[metrics.RAFT_600.EPE_FB , metrics.RAFT_600.outliersFB] = compute_epe(flowRaftPy_600, flowFB2, maskResized);
[metrics.RAFT_900.EPE_FB , metrics.RAFT_900.outliersFB] = compute_epe(flowRaftPy_900, flowFB2, maskResized);
[metrics.RAFT_2000.EPE_FB , metrics.RAFT_2000.outliersFB] = compute_epe(flowRaftPy_2000, flowFB2, maskResized);

% Angular Error (AE)
metrics.RAFT.AE_LK  = compute_ae(flowRaftML2, flowLK2, maskResized);
metrics.RAFT_300.AE_LK  = compute_ae(flowRaftPy_300, flowLK2, maskResized);
metrics.RAFT_600.AE_LK  = compute_ae(flowRaftPy_600, flowLK2, maskResized);
metrics.RAFT_900.AE_LK  = compute_ae(flowRaftPy_900, flowLK2, maskResized);
metrics.RAFT_2000.AE_LK  = compute_ae(flowRaftPy_2000, flowLK2, maskResized);

metrics.RAFT.AE_FB  = compute_ae(flowRaftML2, flowFB2, maskResized);
metrics.RAFT_300.AE_FB  = compute_ae(flowRaftPy_300, flowFB2, maskResized);
metrics.RAFT_600.AE_FB  = compute_ae(flowRaftPy_600, flowFB2, maskResized);
metrics.RAFT_900.AE_FB  = compute_ae(flowRaftPy_900, flowFB2, maskResized);
metrics.RAFT_2000.AE_FB  = compute_ae(flowRaftPy_2000, flowFB2, maskResized);

% Coverage-
[metrics.LK.coverage, validFlow_LK]   = compute_coverage(flowLK2,  maskResized);
[metrics.FB.coverage, validFlow_FB]   = compute_coverage(flowFB2,  maskResized);
[metrics.RAFT.coverage, validFlow_ML] = compute_coverage(flowRaftML2, maskResized);
[metrics.RAFT_300.coverage, validFlow_300] = compute_coverage(flowRaftPy_300, maskResized);
[metrics.RAFT_600.coverage, validFlow_600] = compute_coverage(flowRaftPy_600, maskResized);
[metrics.RAFT_900.coverage, validFlow_900] = compute_coverage(flowRaftPy_900, maskResized);
[metrics.RAFT_2000.coverage, validFlow_2000] = compute_coverage(flowRaftPy_2000, maskResized);

% Runtime
metrics.LK.runtime   = timeLk;
metrics.FB.runtime   = timeFb;
metrics.RAFT.runtime = timeRaftML;
metrics.RAFT_300.runtime = timing_results_300{'mean_time_per_pair'};
metrics.RAFT_600.runtime = timing_results_600{'mean_time_per_pair'};
metrics.RAFT_900.runtime = timing_results_900{'mean_time_per_pair'};
metrics.RAFT_2000.runtime = timing_results_2000{'mean_time_per_pair'};


%% === DISPLAY RESULTS TABLE ===
if (idxPair+1)/2 == 4 || (idxPair+1)/2 == 6 || (idxPair+1)/2 == 7 || (idxPair+1)/2 == 8 || (idxPair+1)/2 ==13 || (idxPair+1)/2 == 12
    fprintf('\n=== EVALUATION RESULTS ===\n');
    % fprintf('current Translation Magnitude: %.3f \n', syntheticTranslationMag)
    % fprintf('current Translation Angle: %.3f \n', syntheticTranslationAngle)
    fprintf('| Method         | EPE   | outliers  | AE          | Coverage    | Runtime (s) |\n');
    fprintf('|----------------|-------|-----------|-------------|-------------|-------------|\n');
    fprintf('| LK & FB        |-------|-----------|-------------|%.1f%% %.1f%%|  %.3f  %.3f |\n', ...
         metrics.LK.coverage*100,  metrics.FB.coverage*100, metrics.LK.runtime, metrics.FB.runtime);
    fprintf('| RAFT vs LK     | %.3f | %.3f | %.3f |  %.1f%%   |    %.3f     |\n', ...
        metrics.RAFT.EPE_LK, metrics.RAFT.outliersLK, metrics.RAFT.AE_LK, metrics.RAFT.coverage*100, metrics.RAFT.runtime);
    fprintf('| RAFT300 vs LK  | %.3f | %.3f | %.3f |  %.1f%%   |    %.3f     |\n', ...
        metrics.RAFT_300.EPE_LK, metrics.RAFT_300.outliersLK, metrics.RAFT_300.AE_LK, metrics.RAFT_300.coverage*100, metrics.RAFT_300.runtime);
    fprintf('| RAFT600 vs LK  | %.3f | %.3f | %.3f |  %.1f%%   |    %.3f     |\n', ...
        metrics.RAFT_600.EPE_LK, metrics.RAFT_600.outliersLK, metrics.RAFT_600.AE_LK, metrics.RAFT_600.coverage*100, metrics.RAFT_300.runtime);
    fprintf('| RAFT900 vs LK  | %.3f | %.3f | %.3f |  %.1f%%   |    %.3f     |\n', ...
        metrics.RAFT_900.EPE_LK, metrics.RAFT_900.outliersLK, metrics.RAFT_900.AE_LK, metrics.RAFT_900.coverage*100, metrics.RAFT_300.runtime);
    fprintf('| RAFT1400 vs LK | %.3f | %.3f | %.3f |  %.1f%%   |    %.3f     |\n', ...
        metrics.RAFT_2000.EPE_LK, metrics.RAFT_2000.outliersLK, metrics.RAFT_2000.AE_LK, metrics.RAFT_2000.coverage*100, metrics.RAFT_300.runtime);
    fprintf('| RAFT vs FB     | %.3f | %.3f | %.3f |  %.1f%%   |    %.3f     |\n', ...
        metrics.RAFT.EPE_FB, metrics.RAFT.outliersFB, metrics.RAFT.AE_FB,  metrics.RAFT.coverage*100, metrics.RAFT.runtime);
    fprintf('| RAFT300 vs FB  | %.3f | %.3f | %.3f |  %.1f%%   |    %.3f     |\n', ...
        metrics.RAFT_300.EPE_FB, metrics.RAFT_300.outliersFB, metrics.RAFT_300.AE_FB, metrics.RAFT_300.coverage*100, metrics.RAFT_300.runtime);
    fprintf('| RAFT600 vs FB  | %.3f | %.3f | %.3f |  %.1f%%   |    %.3f     |\n', ...
        metrics.RAFT_600.EPE_FB, metrics.RAFT_600.outliersFB, metrics.RAFT_600.AE_FB, metrics.RAFT_600.coverage*100, metrics.RAFT_600.runtime);
    fprintf('| RAFT900 vs FB  | %.3f | %.3f | %.3f |  %.1f%%   |    %.3f     |\n', ...
        metrics.RAFT_900.EPE_FB, metrics.RAFT_900.outliersFB, metrics.RAFT_900.AE_FB, metrics.RAFT_900.coverage*100, metrics.RAFT_900.runtime);
    fprintf('| RAFT1400 vs FB | %.3f | %.3f | %.3f |  %.1f%%   |    %.3f     |\n', ...
        metrics.RAFT_2000.EPE_FB, metrics.RAFT_2000.outliersFB, metrics.RAFT_2000.AE_FB, metrics.RAFT_2000.coverage*100, metrics.RAFT_2000.runtime);
    
    %% === SAVE METRICS ===
    fprintf('Saving metrics...\n');
    filename = sprintf('evaluation_metrics_%s.mat', timestamp{1});
    path =  strcat(config.resultsDir,'\evaluation');
    metrics_file = fullfile(path, filename);
    
    if ~exist(path, 'dir')
        mkdir(path);
    end
    
    save(metrics_file, 'metrics');
    fprintf('✓ Saved: %s\n', metrics_file);
    
    %% === VISUALIZATION ===
    % figure('Position', [100 100 1400 400]);
    % % valid flow map
    % subplot(1,7,7);
    % imagesc(validFlow_LK); colorbar; colormap('jet'); axis image;
    % title('LK valid flow Map');
    % 
    % subplot(1,7,6);
    % imagesc(validFlow_FB); colorbar; colormap('jet'); axis image;
    % title('FB valid flow Map');
    % 
    % subplot(1,7,2);
    % imagesc(validFlow_300); colorbar; colormap('jet'); axis image;
    % title('RAFT 300 valid flow Map');
    % 
    % subplot(1,7,3);
    % imagesc(validFlow_600); colorbar; colormap('jet'); axis image;
    % title('RAFT 600 valid flow Map Map');
    % 
    % subplot(1,7,4);
    % imagesc(validFlow_900); colorbar; colormap('jet'); axis image;
    % title('RAFT 900 valid flow Map');
    % 
    % subplot(1,7,5);
    % imagesc(validFlow_2000); colorbar; colormap('jet'); axis image;
    % title('RAFT 1600 valid flow Map');
    % 
    % subplot(1,7,1);
    % imagesc(validFlow_ML); colorbar; colormap('jet'); axis image;
    % title('RAFT Matlab valid flow Map');
    % 
    % filename = sprintf('Evaluation_raft_lk_flow_%s.fig', timestamp{1});
    % path =  strcat(config.resultsDir,'\evaluation');
    % fig = fullfile(path, filename);
    % 
    % savefig(fig)
    
    figure('Position', [100 100 1400 400]);
    % EPE error map
    subplot(2,5,1);
    epe_raft_ml_lk = compute_epe_map(flowRaftML2, flowLK2, maskResized);
    imagesc(epe_raft_ml_lk); colorbar; colormap('jet'); axis image;
    title('LK vs RAFT ML EPE Error Map');
    % EPE error map
    subplot(2,5,2);
    epe_raft_300_lk = compute_epe_map(flowRaftPy_300, flowLK2, maskResized);
    imagesc(epe_raft_300_lk); colorbar; colormap('jet'); axis image;
    title('LK vs RAFT 300 EPE Error Map');
    % EPE error map
    subplot(2,5,3);
    epe_raft_600_lk = compute_epe_map(flowRaftPy_600, flowLK2, maskResized);
    imagesc(epe_raft_600_lk); colorbar; colormap('jet'); axis image;
    title('LK vs RAFT 600 EPE Error Map');
    % EPE error map
    subplot(2,5,4);
    epe_raft_900_lk = compute_epe_map(flowRaftPy_900, flowLK2, maskResized);
    imagesc(epe_raft_900_lk); colorbar; colormap('jet'); axis image;
    title('LK vs RAFT 900 EPE Error Map');
    % EPE error map
    subplot(2,5,5);
    epe_raft_2000_lk = compute_epe_map(flowRaftPy_2000, flowLK2, maskResized);
    imagesc(epe_raft_2000_lk); colorbar; colormap('jet'); axis image;
    title('LK vs RAFT 1600 EPE Error Map');
    hold on
    
    % EPE error map
    subplot(2,5,6);
    epe_raft_ml_fb = compute_epe_map(flowRaftML2, flowFB2, maskResized);
    imagesc(epe_raft_ml_fb); colorbar; colormap('jet'); axis image;
    title('Farnebäck vs RAFT ML EPE Error Map');
    % EPE error map
    subplot(2,5,7);
    epe_raft_300_fb = compute_epe_map(flowRaftPy_300, flowFB2, maskResized);
    imagesc(epe_raft_300_fb); colorbar; colormap('jet'); axis image;
    title('Farnebäck vs RAFT 300 EPE Error Map');
    % EPE error map
    subplot(2,5,8);
    epe_raft_600_fb = compute_epe_map(flowRaftPy_600, flowFB2, maskResized);
    imagesc(epe_raft_600_fb); colorbar; colormap('jet'); axis image;
    title('Farnebäck vs RAFT 600 EPE Error Map');
    % EPE error map
    subplot(2,5,9);
    epe_raft_900_fb = compute_epe_map(flowRaftPy_900, flowFB2, maskResized);
    imagesc(epe_raft_900_fb); colorbar; colormap('jet'); axis image;
    title('Farnebäck vs RAFT 900 EPE Error Map');
    % EPE error map
    subplot(2,5,10);
    epe_raft_2000_fb = compute_epe_map(flowRaftPy_2000, flowFB2, maskResized);
    imagesc(epe_raft_2000_fb); colorbar; colormap('jet'); axis image;
    title('Farnebäck vs RAFT 1600 EPE Error Map');
    
    filename = sprintf('Evaluation_raft_fb_lk_flow_%s.fig', timestamp{1});
    path =  strcat(config.resultsDir,'\evaluation');
    fig = fullfile(path, filename);
    savefig(fig)
    
    % figure; imshow(img2Resized); hold on;
    % plot(flowFB2, 'DecimationFactor',[5 5],'ScaleFactor',1,'Color','blue');
    % title('FB');
    % 
    % figure; imshow(img2Resized); hold on;
    % plot(flowLK2, 'DecimationFactor',[5 5],'ScaleFactor',1,'Color','blue');
    % title('LK');
    % 
    % figure; imshow(img2Resized); hold on;
    % plot(flowRaftML2, 'DecimationFactor',[5 5],'ScaleFactor',1,'Color','blue');
    % title('RAFT Matlab');
    % 
    figure; imshow(img2Resized); hold on;
    plot(flowRaftPy_300, 'DecimationFactor',[5 5],'ScaleFactor',1,'Color','blue');
    title('Raft 300');

    figure; imshow(img2Resized); hold on;
    plot(flowRaftPy_600, 'DecimationFactor',[5 5],'ScaleFactor',1,'Color','blue');
    title('Raft 600');

    figure; imshow(img2Resized); hold on;
    plot(flowRaftPy_900, 'DecimationFactor',[5 5],'ScaleFactor',1,'Color','blue');
    title('Raft 900');

    figure; imshow(img2Resized); hold on;
    plot(flowRaftPy_2000, 'DecimationFactor',[5 5],'ScaleFactor',1,'Color','blue');
    title('Raft 1600');
    
    drawnow
end