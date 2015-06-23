clear all; close all; clc;

%% startup
startup;
config.imageset = 'test';
config.cmap= './voc_gt_cmap.mat';       
config.gpuNum = 0;                      % gpu id
config.Path.CNN.caffe_root = './caffe'; % caffe root path
config.save_root = './results';         % result will be save in this directory

%% configuration
config.write_file = 1;
config.thres = 0.5;
config.im_sz = 320;

%% DecoupledNet Full annotations
config.model_name = 'DecoupledNet_Full_anno';
config.Path.CNN.script_path = './DecoupledNet_Full_anno';
config.Path.CNN.model_data = [config.Path.CNN.script_path '/DecoupledNet_Full_anno_inference.caffemodel'];
config.Path.CNN.model_proto = [config.Path.CNN.script_path '/DecoupledNet_Full_anno_inference_deploy.prototxt'];

DecoupledNet_inference(config);

%% DecoupledNet 25 annotations
config.model_name = 'DecoupledNet_25_anno';
config.Path.CNN.script_path = './DecoupledNet_25_anno';
config.Path.CNN.model_data = [config.Path.CNN.script_path '/DecoupledNet_25_anno_inference.caffemodel'];
config.Path.CNN.model_proto = [config.Path.CNN.script_path '/DecoupledNet_25_anno_inference_deploy.prototxt'];

DecoupledNet_inference(config);

%% DecoupledNet 10 annotations
config.model_name = 'DecoupledNet_10_anno';
config.Path.CNN.script_path = './DecoupledNet_10_anno';
config.Path.CNN.model_data = [config.Path.CNN.script_path '/DecoupledNet_10_anno_inference.caffemodel'];
config.Path.CNN.model_proto = [config.Path.CNN.script_path '/DecoupledNet_10_anno_inference_deploy.prototxt'];

DecoupledNet_inference(config);

%% DecoupledNet 5 annotations
config.model_name = 'DecoupledNet_5_anno';
config.Path.CNN.script_path = './DecoupledNet_5_anno';
config.Path.CNN.model_data = [config.Path.CNN.script_path '/DecoupledNet_5_anno_inference.caffemodel'];
config.Path.CNN.model_proto = [config.Path.CNN.script_path '/DecoupledNet_5_anno_inference_deploy.prototxt'];

DecoupledNet_inference(config);





