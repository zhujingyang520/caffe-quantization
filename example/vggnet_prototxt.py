##########################################################
# This file exports the basic interface for generation of
# VGGNet-16, runner-up of ILSVRC14
##########################################################

# setup and import modules
import sys
caffe_root = '../../'
sys.path.insert(0, caffe_root+'python')
import caffe
from caffe import layers as L, params as P

# shorthand for learning parameter of weights & biases
# lr_mult: local learning rate multiplier; decay_mult: local decay multiplier
weight_param = dict(lr_mult=2, decay_mult=1)
bias_param = dict(lr_mult=1, decay_mult=0)
learned_param = [weight_param, bias_param]

# frozen param: learning rate multiplier = 0
frozen_param = [dict(lr_mult=0, decay_mult=0)]*2

# helper function for common layer generation
def conv(bottom, ks=3, nout=64, stride=1, pad=1, param=learned_param,
        weight_filler=dict(type='gaussian', std=0.01),
        bias_filler=dict(type='constant', value=0.1)):
    """CONV"""
    conv = L.Convolution(bottom, kernel_size=ks, stride=stride,
            num_output=nout, pad=pad, param=param,
            weight_filler=weight_filler, bias_filler=bias_filler)
    return conv

def conv_relu(bottom, ks=3, nout=64, stride=1, pad=1, param=learned_param,
        weight_filler=dict(type='gaussian', std=0.01),
        bias_filler=dict(type='constant', value=0.1)):
    """CONV + ReLU"""
    conv = L.Convolution(bottom, kernel_size=ks, stride=stride,
            num_output=nout, pad=pad, param=param,
            weight_filler=weight_filler, bias_filler=bias_filler)
    return conv, L.ReLU(conv, in_place=False)

def fc(bottom, nout, param=learned_param,
        weight_filler=dict(type='gaussian', std=0.01),
        bias_filler=dict(type='constant', value=0.1)):
    """FC"""
    fc = L.InnerProduct(bottom, num_output=nout, param=param,
            weight_filler=weight_filler, bias_filler=bias_filler)
    return fc

def fc_relu(bottom, nout, param=learned_param,
        weight_filler=dict(type='gaussian', std=0.01),
        bias_filler=dict(type='constant', value=0.1)):
    """FC + ReLU"""
    fc = L.InnerProduct(bottom, num_output=nout, param=param,
            weight_filler=weight_filler, bias_filler=bias_filler)
    return fc, L.ReLU(fc, in_place=False)

def accuracy_top1_top5(bottom, label):
    """Accuracy Top1 & Top5"""
    accuracy_top1 = L.Accuracy(bottom, label, include=dict(phase=caffe.TEST),
            accuracy_param=dict(top_k=1))
    accuracy_top5 = L.Accuracy(bottom, label, include=dict(phase=caffe.TEST),
            accuracy_param=dict(top_k=5))
    return accuracy_top1, accuracy_top5

def max_pool(bottom, ks=2, stride=2):
    """MAX Pooling"""
    return L.Pooling(bottom, pool=P.Pooling.MAX, kernel_size=ks, stride=stride)


################################################
# Helper function to generate VGGNet16 prototxt
################################################
def vggnet_16_proto(data, label=None, train=True, num_classes=1000,
        classifier_name='fc8', learn_all=False, filename='vggnet16.prototxt'):
    """Generates prototxt of VGGNet16"""
    n = caffe.NetSpec()
    n.data = data # 1st input layer
    param = learned_param if learn_all else frozen_param # use learnable parameter

    # set 1: 2 CONVs with output neuron no. = 64
    n.conv1_1, n.relu1_1 = conv_relu(n.data, nout=64, param=param)
    n.conv1_2, n.relu1_2 = conv_relu(n.relu1_1, nout=64, param=param)
    n.pool1 = max_pool(n.relu1_2)

    # set 2: 2 CONVs with output neuron no. = 128
    n.conv2_1, n.relu2_1 = conv_relu(n.pool1, nout=128, param=param)
    n.conv2_2, n.relu2_2 = conv_relu(n.relu2_1, nout=128, param=param)
    n.pool2 = max_pool(n.relu2_2)

    # set 3: 3 CONVs with output neuron no. = 256
    n.conv3_1, n.relu3_1 = conv_relu(n.pool2, nout=256, param=param)
    n.conv3_2, n.relu3_2 = conv_relu(n.relu3_1, nout=256, param=param)
    n.conv3_3, n.relu3_3 = conv_relu(n.relu3_2, nout=256, param=param)
    n.pool3 = max_pool(n.relu3_3)

    # set 4: 3 CONVs with output neuron no. = 512
    n.conv4_1, n.relu4_1 = conv_relu(n.pool3, nout=512, param=param)
    n.conv4_2, n.relu4_2 = conv_relu(n.relu4_1, nout=512, param=param)
    n.conv4_3, n.relu4_3 = conv_relu(n.relu4_2, nout=512, param=param)
    n.pool4 = max_pool(n.relu4_3)

    # set 5: 3 CONVs with output neuron no. = 512
    n.conv5_1, n.relu5_1 = conv_relu(n.pool4, nout=512, param=param)
    n.conv5_2, n.relu5_2 = conv_relu(n.relu5_1, nout=512, param=param)
    n.conv5_3, n.relu5_3 = conv_relu(n.relu5_2, nout=512, param=param)
    n.pool5 = max_pool(n.relu5_3)

    # FC6 with output neuron no. = 4096
    n.fc6, n.relu6 = fc_relu(n.pool5, nout=4096, param=param)
    if train:
        n.drop6 = fc7input = L.Dropout(n.relu6, in_place=False,
                dropout_param=dict(dropout_ratio=0.5))
    else:
        fc7input = n.relu6 # dropout has no impact during inference

    # FC7 with output neuron no. = 4096
    n.fc7, n.relu7 = fc_relu(fc7input, nout=4096, param=param)
    if train:
        n.drop7 = fc8input = L.Dropout(n.relu7, in_place=False,
                dropout_param=dict(dropout_ratio=0.5))
    else:
        fc8input = n.relu7 # dropout has no impact during inference

    # FC8 with output neuron no. = 1000; always learn fc8
    # (param = learned_param)
    fc8 = L.InnerProduct(fc8input, num_output=num_classes, param=learned_param)
    # give fc8 the name specified by argument `classifier_name`
    n.__setattr__(classifier_name, fc8)

    # add the accuracy loss layer
    if not train:
        n.probs = L.Softmax(fc8)
    if label is not None:
        n.label = label
        n.acc_top1, n.acc_top5 = accuracy_top1_top5(fc8, n.label)

    # write the net to prototxt file
    with open(filename, 'w') as f:
        f.write(str(n.to_proto()))


#############################################################
# Helper function to generate fixed point VGGNet-16 prototxt
#############################################################
def quantized_vggnet_16_proto(data, blobs_quantization_params, label=None, train=True,
        num_classes=1000, classifier_name='fc8', learn_all=False, filename='vggnet16.prototxt'):
    """Returns a NetSpec specified Quantized VGGNet-16"""
    param = learned_param if learn_all else frozen_param # use learnable parameter

    n = caffe.NetSpec()
    n.data = data                   # 1st input layer
    n.quantized_data = L.Quantization(n.data, quantization_param=blobs_quantization_params['data'])

    # set 1: 2 CONVs with output neuron no. = 64
    #n.conv1_1, n.relu1_1 = conv_relu(n.data, nout=64, param=param)
    n.conv1_1 = conv(n.quantized_data, nout=64, param=param)
    n.quantized_conv1_1 = L.Quantization(n.conv1_1, quantization_param=blobs_quantization_params['conv1_1'])
    n.relu1_1 = L.ReLU(n.quantized_conv1_1, in_place=False)
    n.quantized_relu1_1 = L.Quantization(n.relu1_1, quantization_param=blobs_quantization_params['relu1_1'])
    #n.conv1_2, n.relu1_2 = conv_relu(n.relu1_1, nout=64, param=param)
    n.conv1_2 = conv(n.quantized_relu1_1, nout=64, param=param)
    n.quantized_conv1_2 = L.Quantization(n.conv1_2, quantization_param=blobs_quantization_params['conv1_2'])
    n.relu1_2 = L.ReLU(n.quantized_conv1_2, in_place=False)
    n.quantized_relu1_2 = L.Quantization(n.relu1_2, quantization_param=blobs_quantization_params['relu1_2'])
    #n.pool1 = max_pool(n.relu1_2)
    n.pool1 = max_pool(n.quantized_relu1_2)
    n.quantized_pool1 = L.Quantization(n.pool1, quantization_param=blobs_quantization_params['pool1'])

    # set 2: 2 CONVs with output neuron no. = 128
    #n.conv2_1, n.relu2_1 = conv_relu(n.pool1, nout=128, param=param)
    n.conv2_1 = conv(n.quantized_pool1, nout=128, param=param)
    n.quantized_conv2_1 = L.Quantization(n.conv2_1, quantization_param=blobs_quantization_params['conv2_1'])
    n.relu2_1 = L.ReLU(n.quantized_conv2_1, in_place=False)
    n.quantized_relu2_1 = L.Quantization(n.relu2_1, quantization_param=blobs_quantization_params['relu2_1'])
    #n.conv2_2, n.relu2_2 = conv_relu(n.relu2_1, nout=128, param=param)
    n.conv2_2 = conv(n.quantized_relu2_1, nout=128, param=param)
    n.quantized_conv2_2 = L.Quantization(n.conv2_2, quantization_param=blobs_quantization_params['conv2_2'])
    n.relu2_2 = L.ReLU(n.quantized_conv2_2, in_place=False)
    n.quantized_relu2_2 = L.Quantization(n.relu2_2, quantization_param=blobs_quantization_params['relu2_2'])
    #n.pool2 = max_pool(n.relu2_2)
    n.pool2 = max_pool(n.quantized_relu2_2)
    n.quantized_pool2 = L.Quantization(n.pool2, quantization_param=blobs_quantization_params['pool2'])

    # set 3: 3 CONVs with output neuron no. = 256
    #n.conv3_1, n.relu3_1 = conv_relu(n.pool2, nout=256, param=param)
    n.conv3_1 = conv(n.quantized_pool2, nout=256, param=param)
    n.quantized_conv3_1 = L.Quantization(n.conv3_1, quantization_param=blobs_quantization_params['conv3_1'])
    n.relu3_1 = L.ReLU(n.quantized_conv3_1, in_place=False)
    n.quantized_relu3_1 = L.Quantization(n.relu3_1, quantization_param=blobs_quantization_params['relu3_1'])
    #n.conv3_2, n.relu3_2 = conv_relu(n.relu3_1, nout=256, param=param)
    n.conv3_2 = conv(n.quantized_relu3_1, nout=256, param=param)
    n.quantized_conv3_2 = L.Quantization(n.conv3_2, quantization_param=blobs_quantization_params['conv3_2'])
    n.relu3_2 = L.ReLU(n.quantized_conv3_2, in_place=False)
    n.quantized_relu3_2 = L.Quantization(n.relu3_2, quantization_param=blobs_quantization_params['relu3_2'])
    #n.conv3_3, n.relu3_3 = conv_relu(n.relu3_2, nout=256, param=param)
    n.conv3_3 = conv(n.quantized_relu3_2, nout=256, param=param)
    n.quantized_conv3_3 = L.Quantization(n.conv3_3, quantization_param=blobs_quantization_params['conv3_3'])
    n.relu3_3 = L.ReLU(n.quantized_conv3_3, in_place=False)
    n.quantized_relu3_3 = L.Quantization(n.relu3_3, quantization_param=blobs_quantization_params['relu3_3'])
    #n.pool3 = max_pool(n.relu3_3)
    n.pool3 = max_pool(n.quantized_relu3_3)
    n.quantized_pool3 = L.Quantization(n.pool3, quantization_param=blobs_quantization_params['pool3'])

    # set 4: 3 CONVs with output neuron no. = 512
    #n.conv4_1, n.relu4_1 = conv_relu(n.pool3, nout=512, param=param)
    n.conv4_1 = conv(n.quantized_pool3, nout=512, param=param)
    n.quantized_conv4_1 = L.Quantization(n.conv4_1, quantization_param=blobs_quantization_params['conv4_1'])
    n.relu4_1 = L.ReLU(n.quantized_conv4_1, in_place=False)
    n.quantized_relu4_1 = L.Quantization(n.relu4_1, quantization_param=blobs_quantization_params['relu4_1'])
    #n.conv4_2, n.relu4_2 = conv_relu(n.relu4_1, nout=512, param=param)
    n.conv4_2 = conv(n.quantized_relu4_1, nout=512, param=param)
    n.quantized_conv4_2 = L.Quantization(n.conv4_2, quantization_param=blobs_quantization_params['conv4_2'])
    n.relu4_2 = L.ReLU(n.quantized_conv4_2, in_place=False)
    n.quantized_relu4_2 = L.Quantization(n.relu4_2, quantization_param=blobs_quantization_params['relu4_2'])
    #n.conv4_3, n.relu4_3 = conv_relu(n.relu4_2, nout=512, param=param)
    n.conv4_3 = conv(n.quantized_relu4_2, nout=512, param=param)
    n.quantized_conv4_3 = L.Quantization(n.conv4_3, quantization_param=blobs_quantization_params['conv4_3'])
    n.relu4_3 = L.ReLU(n.quantized_conv4_3, in_place=False)
    n.quantized_relu4_3 = L.Quantization(n.relu4_3, quantization_param=blobs_quantization_params['relu4_3'])
    #n.pool4 = max_pool(n.relu4_3)
    n.pool4 = max_pool(n.quantized_relu4_3)
    n.quantized_pool4 = L.Quantization(n.pool4, quantization_param=blobs_quantization_params['pool4'])

    # set 5: 3 CONVs with output neuron no. = 512
    #n.conv5_1, n.relu5_1 = conv_relu(n.pool4, nout=512, param=param)
    n.conv5_1 = conv(n.quantized_pool4, nout=512, param=param)
    n.quantized_conv5_1 = L.Quantization(n.conv5_1, quantization_param=blobs_quantization_params['conv5_1'])
    n.relu5_1 = L.ReLU(n.quantized_conv5_1, in_place=False)
    n.quantized_relu5_1 = L.Quantization(n.relu5_1, quantization_param=blobs_quantization_params['relu5_1'])
    #n.conv5_2, n.relu5_2 = conv_relu(n.relu5_1, nout=512, param=param)
    n.conv5_2 = conv(n.quantized_relu5_1, nout=512, param=param)
    n.quantized_conv5_2 = L.Quantization(n.conv5_2, quantization_param=blobs_quantization_params['conv5_2'])
    n.relu5_2 = L.ReLU(n.quantized_conv5_2, in_place=False)
    n.quantized_relu5_2 = L.Quantization(n.relu5_2, quantization_param=blobs_quantization_params['relu5_2'])
    #n.conv5_3, n.relu5_3 = conv_relu(n.relu5_2, nout=512, param=param)
    n.conv5_3 = conv(n.quantized_relu5_2, nout=512, param=param)
    n.quantized_conv5_3 = L.Quantization(n.conv5_3, quantization_param=blobs_quantization_params['conv5_3'])
    n.relu5_3 = L.ReLU(n.quantized_conv5_3, in_place=False)
    n.quantized_relu5_3 = L.Quantization(n.relu5_3, quantization_param=blobs_quantization_params['relu5_3'])
    #n.pool5 = max_pool(n.relu5_3)
    n.pool5 = max_pool(n.quantized_relu5_3)
    n.quantized_pool5 = L.Quantization(n.pool5, quantization_param=blobs_quantization_params['pool5'])

    # FC6 with output neuron no. = 4096
    #n.fc6, n.relu6 = fc_relu(n.pool5, nout=4096, param=param)
    n.fc6 = fc(n.quantized_pool5, nout=4096, param=param)
    n.quantized_fc6 = L.Quantization(n.fc6, quantization_param=blobs_quantization_params['fc6'])
    n.relu6 = L.ReLU(n.quantized_fc6, in_place=False)
    n.quantized_relu6 = L.Quantization(n.relu6, quantization_param=blobs_quantization_params['relu6'])

    if train:
        n.drop6 = L.Dropout(n.quantized_relu6, in_place=False, dropout_param=dict(dropout_ratio=0.5))
        fc7input = n.quantized_drop6 = L.Quantization(n.drop6, quantization_param=
                blobs_quantization_params['drop6'])
    else:
        fc7input = n.quantized_relu6 # dropout has no impact during inference

    # FC7 with output neuron no. = 4096
    #n.fc7, n.relu7 = fc_relu(fc7input, nout=4096, param=param)
    n.fc7 = fc(fc7input, nout=4096, param=param)
    n.quantized_fc7 = L.Quantization(n.fc7, quantization_param=blobs_quantization_params['fc7'])
    n.relu7 = L.ReLU(n.quantized_fc7, in_place=False)
    n.quantized_relu7 = L.Quantization(n.relu7, quantization_param=blobs_quantization_params['relu7'])
    if train:
        n.drop7 = L.Dropout(n.quantized_relu7, in_place=False, dropout_param=dict(dropout_ratio=0.5))
        fc8input = n.quantized_drop7 = L.Quantization(n.drop7, quantization_param=
                blobs_quantization_params['drop7'])
    else:
        fc8input = n.quantized_relu7 # dropout has no impact during inference

    # FC8 with output neuron no. = 1000; always learn fc8
    # (param = learned_param)
    fc8 = L.InnerProduct(fc8input, num_output=num_classes, param=learned_param)
    # give fc8 the name specified by argument `classifier_name`
    n.__setattr__(classifier_name, fc8)
    n.quantized_fc8 = L.Quantization(fc8, quantization_param=blobs_quantization_params['fc8'])

    # add the accuracy loss layer
    if not train:
        n.probs = L.Softmax(n.quantized_fc8)
    if label is not None:
        n.label = label
        n.acc_top1, n.acc_top5 = accuracy_top1_top5(n.quantized_fc8, n.label)

    # write the net to prototxt file
    with open(filename, 'w') as f:
        f.write(str(n.to_proto()))

def convert_to_quantization_param(blobs_bit_width, blobs_range, round_method='FLOOR',
        round_strategy='CONSERVATIVE'):
    '''Helper function to convert the aggregate the blobs bit width, blobs range,
       round method, and round strategy together to the desired quantization parameter.

       Parameters
       -----------
       - blobs_bit_width: dictionary containing each layer's bit width
       - blobs_range: dictionary containing each layer's range
       - round_method, round_strategy: here we assumes a uniform method or strategy for
         each internel blobs
    '''

    # round method
    if round_method.upper() == 'FLOOR':
        quantization_param = dict(round_method=P.Quantization.FLOOR)
    elif round_method.upper() == 'CEIL':
        quantization_param = dict(round_method=P.Quantization.CEIL)
    elif round_method.upper() == 'ROUND':
        quantization_param = dict(round_method=P.Quantization.ROUND)
    elif round_method.upper() == 'TRUNC':
        quantization_param = dict(round_method=P.Quantization.TRUNC)
    else:
        raise TypeError('undefined rounding method: %s' % (round_method,))

    # round strategy
    if round_strategy.upper() == 'CONSERVATIVE':
        quantization_param['round_strategy'] = P.Quantization.CONSERVATIVE
    elif round_strategy.upper() == 'NEUTRAL':
        quantization_param['round_strategy'] = P.Quantization.NEUTRAL
    elif round_strategy.upper() == 'AGGRESSIVE':
        quantization_param['round_strategy'] = P.Quantization.AGGRESSIVE
    else:
        raise TypeError('undefined rounding strategy: %s' % (round_strategy,))

    blobs_quantization_params = {}
    for k, v in blobs_bit_width.iteritems():
        blobs_quantization_params[k] = quantization_param.copy()

        # bit width for quantization layer k
        blobs_quantization_params[k]['bit_width'] = v

        # data range for quantization layer k, make sure blobs range has key k
        blobs_quantization_params[k]['range'] = blobs_range[k]

    return blobs_quantization_params
