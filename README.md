## DecoupledNet: Decoupled Deep Neural Network for Semi-supervised Semantic Segmentation 

Created by Seunghoon Hong, Hyeonwoo Noh and Bohyung Han at POSTECH

Acknowledgements: Thanks to Yangqing Jia and the BVLC team for creating Caffe.

### Introduction

DecoupledNet is semantic segmentation system which using heterogeneous annotations.
From pre-trained classification network, DecoupledNet fine-tune the segmentation network with very small amount of segmentation annotations and obtains excellent results on semantic segmentation task.

Detailed description of the system will be provided by our technical report [arXiv tech report] http://arxiv.org/abs/1506.04924 

### Citation

If you're using this code in a publication, please cite our papers.

    @article{hong2015decoupled,
      title={Decoupled Deep Neural Network for Semi-supervised Semantic Segmentation},
      author={Hong, Seunghoon and Noh, Hyeonwoo and Han, Bohyung},
      journal={arXiv preprint arXiv:1506.04924},
      year={2015}
    }

### Pre-trained Model

If you need model definition and pre-trained model only, you can download them from following location:
  0. caffe for DecoupledNet: https://github.com/HyeonwooNoh/caffe
  0. DecoupledNet [Full annotation] : 
    0. [prototxt] (http://cvlab.postech.ac.kr/research/decouplednet/model/DecoupledNet_Full_anno/DecoupledNet_Full_anno_inference_deploy.prototxt)
    0. [caffemodel] (http://cvlab.postech.ac.kr/research/decouplednet/model/DecoupledNet_Full_anno/DecoupledNet_Full_anno_inference.caffemodel)
  0. DecoupledNet [25 annotations] : 
    0. [prototxt] (http://cvlab.postech.ac.kr/research/decouplednet/model/DecoupledNet_25_anno/DecoupledNet_25_anno_inference_deploy.prototxt)
    0. [caffemodel] (http://cvlab.postech.ac.kr/research/decouplednet/model/DecoupledNet_25_anno/DecoupledNet_25_anno_inference.caffemodel)
  0. DecoupledNet [10 annotations] : 
    0. [prototxt] (http://cvlab.postech.ac.kr/research/decouplednet/model/DecoupledNet_10_anno/DecoupledNet_10_anno_inference_deploy.prototxt)
    0. [caffemodel] (http://cvlab.postech.ac.kr/research/decouplednet/model/DecoupledNet_10_anno/DecoupledNet_10_anno_inference.caffemodel)
  0. DecoupledNet [5 annotations] : 
    0. [prototxt] (http://cvlab.postech.ac.kr/research/decouplednet/model/DecoupledNet_5_anno/DecoupledNet_5_anno_inference_deploy.prototxt)
    0. [caffemodel] (http://cvlab.postech.ac.kr/research/decouplednet/model/DecoupledNet_5_anno/DecoupledNet_5_anno_inference.caffemodel)

### Licence

This software is being made available for research purpose only.
Check LICENSE file for details.

### System Requirements

This software is tested on Ubuntu 14.04 LTS (64bit).

**Prerequisites** 
  0. MATLAB (tested with 2014b on 64-bit Linux)
  0. prerequisites for caffe(http://caffe.berkeleyvision.org/installation.html#prequequisites)

### Installing DecoupledNet

**By running "setup.sh" you can download all the necessary file for training and inference including:**
  0. caffe: you need modified version of caffe which support DeconvNet - https://github.com/HyeonwooNoh/caffe.git
  0. data: data used for training
  0. model: caffemodel of trained DecoupledNet and caffemodel of pre-trained classification network

### Training DecoupledNet

Training scripts are included in *./training/* directory

**To train DecoupledNet with various setting, you can run following scripts**
  0. 001_convert_classification_network_to_fp_bp_network.sh: 
    * converting classification network to make forward-backward propagation possible (this converted model is prerequisite for DecoupledNet training)
  0. 002_train_seg_Full_anno.sh: 
    * training DecoupledNet with full segmentation annotations
  0. 003_train_seg_25_anno.sh: 
    * training DecoupledNet with 25 segmentation annotations per class
  0. 004_train_seg_10_anno.sh: 
    * training DecoupledNet with 10 segmentation annotations per class
  0. 005_train_seg_5_anno.sh: 
    * training DecoupledNet with 5 segmentation annotations per class

### Inference EDeconvNet+CRF

Run *run_demo.m* to reproduce EDeconvNet+CRF results on VOC2012 test data.

**This script will generated EDeconvNet+CRF results through following steps:**
  0. run FCN-8s and cache the score [cache\_FCN8s\_results.m]
  0. generate DeconvNet score and apply ensemble with FCN-8s score, post processing with densecrf [generate\_EDeconvNet\_CRF\_results.m]

*EDeconvNet+CRF obtains 72.5 mean I/U on PASCAL VOC 2012 Test*

**External dependencies [can be downloaded by running "setup.sh" script]**
  0. FCN-8s model and weight file [https://github.com/BVLC/caffe/wiki/Model-Zoo]
  0. densecrf with matlab wrapper [https://github.com/johannesu/meanfield-matlab.git]
  0. cached proposal bounding boxes extracted with edgebox object proposal [https://github.com/pdollar/edges] 






 
