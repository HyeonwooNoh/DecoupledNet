# download and extract data necessary for training
cd VOC2012

# download and extract image and segmentation annotations necessary for training
wget http://cvlab.postech.ac.kr/research/decouplednet/data/VOC2012_SEG_AUG.tar.gz
tar -zxvf VOC2012_SEG_AUG.tar.gz
rm -rf VOC2012_SEG_AUG.tar.gz

cd ..

# download image sets for segmentation training
cd imagesets

wget http://cvlab.postech.ac.kr/research/decouplednet/data/seg_imgset_Full.tar.gz
tar -zxvf seg_imgset_Full.tar.gz
rm -rf seg_imgset_Full.tar.gz

wget http://cvlab.postech.ac.kr/research/decouplednet/data/seg_imgset_25.tar.gz
tar -zxvf seg_imgset_25.tar.gz
rm -rf seg_imgset_25.tar.gz

wget http://cvlab.postech.ac.kr/research/decouplednet/data/seg_imgset_10.tar.gz
tar -zxvf seg_imgset_10.tar.gz
rm -rf seg_imgset_10.tar.gz

wget http://cvlab.postech.ac.kr/research/decouplednet/data/seg_imgset_5.tar.gz
tar -zxvf seg_imgset_5.tar.gz
rm -rf seg_imgset_5.tar.gz

cd ..
