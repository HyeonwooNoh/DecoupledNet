

######### path configuration #########

caffe_root = './caffe'

script_path = '.'
    
caffe_model = script_path + '/fp_bp_network.prototxt'    
caffe_weight = script_path + '/voc_finetuning_result/vgg16_conv_fintuned_on_voc_with_avg_pool.caffemodel'
save_weight = script_path + '/fp_bp_model.caffemodel'

######################################



import numpy as np
import sys
sys.path.insert(0, caffe_root + '/python')
import caffe


# initialize caffe
net = caffe.Net(caffe_model, caffe_weight)
net.set_phase_test()
net.set_mode_gpu()
net.set_device(0)


# ## copy conv parameter to deconv
print net.params.keys()


# ### cls-score-voc
print net.params['cls-score-voc'][0].data.shape # weight
print net.params['cls-score-voc'][1].data.shape # bias

print net.params['cls-score-voc-bp'][0].data.shape # weight
print net.params['cls-score-voc-bp'][1].data.shape # bias

W = net.params['cls-score-voc'][0].data
print W.shape
W_bp = np.transpose(W,[1,0,2,3])
print W_bp.shape

net.params['cls-score-voc-bp'][0].data[...] = W_bp
net.params['cls-score-voc-bp'][1].data[...] = 0


# ### fc7-bp
print net.params['fc7'][0].data.shape # weight
print net.params['fc7'][1].data.shape # bias

print net.params['fc7-bp'][0].data.shape # weight
print net.params['fc7-bp'][1].data.shape # bias

W = net.params['fc7'][0].data
print W.shape

net.params['fc7-bp'][0].data[...] = W
net.params['fc7-bp'][1].data[...] = 0


# ### fc6-bp
print net.params['fc6'][0].data.shape # weight
print net.params['fc6'][1].data.shape # bias

print net.params['fc6-bp'][0].data.shape # weight
print net.params['fc6-bp'][1].data.shape # bias

W = net.params['fc6'][0].data
print W.shape

net.params['fc6-bp'][0].data[...] = W
net.params['fc6-bp'][1].data[...] = 0


# ### save
net.save(save_weight)

