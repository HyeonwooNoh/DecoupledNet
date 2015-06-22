**This directory contains finetuned classification model with forward-backward propagation layers**
Since we use additional deconvolution layer to imitate back-propagation, parameters of convolution layers have to be copied to corresponding deconvolution layers. We employ net-surgery to copy the parameters and resulting caffemodel is contained in this directory. 
