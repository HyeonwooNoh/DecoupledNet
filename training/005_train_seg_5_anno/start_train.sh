LOGDIR=./training_log
CAFFE=./caffe/build/tools/caffe
SOLVER=./solver.prototxt
WEIGHTS=./fp_bp_model/fp_bp_model.caffemodel

GLOG_log_dir=$LOGDIR $CAFFE train -solver $SOLVER -weights $WEIGHTS -gpu 0

