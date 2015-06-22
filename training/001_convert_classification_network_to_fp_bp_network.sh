##########################################################################
# convert classification network to forward-backward propagation network #
##########################################################################

cd 001_convert_classification_network_to_fp_bp_network

## create simulinks

# caffe
ln -s ../../caffe
# pre-trained model (VGG16 finetuned on voc)
ln -s ../../model/voc_finetuning_result
# directory to save resulting fp_bp_model
ln -s ../../model/fp_bp_model

## generate fp_bp model
python gen_fp_bp_model.py

## copy and rename converted model
cp ./fp_bp_model.caffemodel ./fp_bp_model/fp_bp_model.caffemodel
