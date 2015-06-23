function DecoupledNet_inference(config)
%% start DecoupledNet inference
fprintf('start DecoupledNet inference [%s]\n', config.model_name);

%% initialization
load(config.cmap);
init_VOC2012_TEST;

% initialize caffe
addpath(fullfile(config.Path.CNN.caffe_root, 'matlab/caffe'));
fprintf('initializing caffe..\n');
if caffe('is_initialized')
    caffe('reset')
end
caffe('init', config.Path.CNN.model_proto, config.Path.CNN.model_data)
caffe('set_device', config.gpuNum);
caffe('set_mode_gpu');
caffe('set_phase_test');
fprintf('done\n');

%% initialize paths
save_res_dir = sprintf('%s/%s',config.save_root ,config.model_name);
save_res_path = [save_res_dir '/%s.png'];

%% create directory
if config.write_file
    if ~exist(save_res_dir), mkdir(save_res_dir), end
end

fprintf('start generating result\n');
fprintf('caffe model: %s\n', config.Path.CNN.model_proto);
fprintf('caffe weight: %s\n', config.Path.CNN.model_data);

%% read VOC2012 TEST image set
ids=textread(sprintf(VOCopts.seg.imgsetpath, config.imageset), '%s');

for i=1:length(ids)
    fprintf('progress: %d/%d [%s]...', i, length(ids), ids{i});  
    tic;
    
    % read image
    I=imread(sprintf(VOCopts.imgpath,ids{i}));
    
    im_sz = max(size(I,1),size(I,2));
    caffe_im = padarray(I,[im_sz - size(I,1), im_sz - size(I,2)],'post');
    caffe_im = preprocess_image(caffe_im, config.im_sz);
    label = single(zeros([1,1,20]));
    
    tic;
    cnn_output = caffe('forward', {caffe_im;label});
    fprintf('[0:%f]',toc);    
    cls_score = cnn_output{1};
    cls_score_max = cnn_output{2};
    seg_score = cnn_output{3};
        
    score_map = zeros([config.im_sz,config.im_sz, 21]);
    
    %% compute bkg prob
    label = single(cls_score .* (cls_score > 0.5));    
    tic;
    cnn_output = caffe('forward', {caffe_im;label});
    fprintf('[0:%f]',toc);    
    cls_score = cnn_output{1};
    cls_score_max = cnn_output{2};
    seg_score = cnn_output{3};
            
    softmax_score = exp(seg_score - repmat(max(seg_score, [], 3), [1,1,size(seg_score,3)]));
    softmax_score = softmax_score ./ repmat(sum(softmax_score, 3), [1,1, size(softmax_score,3)]);
    
    score_map(:,:,1) = softmax_score(:,:,1);   
        
    for j = 1:20
        if cls_score(j) > config.thres
            label = zeros([1,1,20]);
            label(j) = cls_score(j);
            label = single(label);
            tic;
            cnn_output = caffe('forward', {caffe_im;label});
            fprintf('[%d:%f]',j,toc);
            seg_score = cnn_output{3};
            
            softmax_score = exp(seg_score - repmat(max(seg_score, [], 3), [1,1,size(seg_score,3)]));
            softmax_score = softmax_score ./ repmat(sum(softmax_score, 3), [1,1, size(softmax_score,3)]);

            score_map(:,:,j+1) = softmax_score(:,:,2);   
        end    
    end
    
    cidx = [1; find(cls_score > config.thres)+1];
    num_c = (sum(cls_score > config.thres)+1);
    score_map(:,:,cidx) = score_map(:,:,cidx) + repmat((single(sum(score_map,3)<=0)), [1, 1, num_c]);
        
    norm_score_map = score_map;
    norm_score_map = single(norm_score_map ./ repmat(sum(score_map, 3), [1,1,size(score_map, 3)]));
    
    resize_score_map = imresize(norm_score_map, [im_sz, im_sz]);    
    
    resize_score_map = permute(resize_score_map, [2,1,3]);
    cropped_score_map = single(resize_score_map(1:size(I,1),1:size(I,2),:));
        
    [~, segmask] = max(cropped_score_map, [], 3);
    segmask = uint8(segmask-1);
                
    if config.write_file
        imwrite(segmask,cmap,sprintf(save_res_path, ids{i}));        
    else
        for j = 1:20
            fprintf('%s: %d\n', VOCopts.classes{j}, cls_score(j)>config.thres);
        end
        subplot(1,2,1);
        imshow(I);
        subplot(1,2,2);
        resulting_seg_im = reshape(cmap(int32(segmask)+1,:),[size(segmask,1),size(segmask,2),3]);
        imshow(resulting_seg_im);
        waitforbuttonpress;
    end
    fprintf(' done\n');
end

%%function end
end
