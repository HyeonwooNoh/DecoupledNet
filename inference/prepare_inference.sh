############################
# prepare to run inference #
############################

## create simulinks
# caffe
ln -s ../caffe
ln -s ../model/DecoupledNet_Full_anno
ln -s ../model/DecoupledNet_25_anno
ln -s ../model/DecoupledNet_10_anno
ln -s ../model/DecoupledNet_5_anno

# download necessary data for inference
cd data
# VOC2012 test data
wget http://cvlab.postech.ac.kr/research/decouplednet/data/VOC2012_TEST.tar.gz
tar -zxvf VOC2012_TEST.tar.gz
rm -rf VOC2012_TEST.tar.gz
cd ..

