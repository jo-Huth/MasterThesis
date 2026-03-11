function create_synthetic_images(createSynFiles, syntheticTranslation, config)
    createSynthImages = config.createSynthImages;
    parfor idxPair = 1:config.numberCreateSynImages
        
        timestamp = regexp(createSynFiles(idxPair).name, '(\d{14})', 'tokens', 'once');
        
        imgPath = fullfile(createSynthImages, createSynFiles(idxPair).name);
        
        try
            img = imread(imgPath);
        catch ME
            error('Failed to load images: %s', ME.message);
        end
        %% Normalize Images
        IMG_H = 512; IMG_W = 512;
        
        % Resize to RAFT input size (no masking, no normalization)
        imgResized = imresize(img, [IMG_H, IMG_W]);
        % create synthetic flow
        syntheticImageSize = [512 512]; % [H W]
        
        % Load ground truth 
        synthetic_gt(imgResized, syntheticImageSize, syntheticTranslation, config, timestamp{1});
    
    end
end