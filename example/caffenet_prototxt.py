##########################################################
# This file exports the basic interface for generation of
# CaffeNet (original version and fixed point version)
##########################################################

# setup basic modules
import sys
caffe_root = '../../'
sys.path.insert(0, caffe_root+'python')
import caffe
from caffe import layers as L, params as P

# shorthand for learning parameter of weights & biases
# lr_mult: local learning rate multiplier; decay_mult: local decay multiplier
weight_param = {'lr_mult': 1, 'decay_mult': 1}
bias_param = {'lr_mult': 2, 'decay_mult': 0}
learned_param = [weight_param, bias_param]

# frozen parameter: learning rate multiplier = decay multiplier = 0
frozen_param = [{'lr_mult': 0, 'decay_mult': 0}] * 2

# helper function for common layer generation
def conv(bottom, ks, nout, stride=1, pad=0, group=1, param=learned_param,
        weight_filler={'type': 'gaussian', 'std': 0.01},
        bias_filler={'type': 'constant', 'std': 0.1}):
    conv = L.Convolution(bottom, kernel_size=ks, stride=stride,
                         num_output=nout, pad=pad, group=group, param=param,
                         weight_filler=weight_filler, bias_filler=bias_filler)
    return conv

def conv_relu(bottom, ks, nout, stride=1, pad=0, group=1, param=learned_param,
             weight_filler={'type': 'gaussian', 'std': 0.01},
             bias_filler={'type': 'constant', 'value': 0.1}):
    '''CONV + ReLU'''
    conv = L.Convolution(bottom, kernel_size=ks, stride=stride,
                         num_output=nout, pad=pad, group=group, param=param,
                         weight_filler=weight_filler, bias_filler=bias_filler)
    return conv, L.ReLU(conv, in_place=False)

def fc(bottom, nout, param=learned_param,
            weight_filler={'type': 'gaussian', 'std': 0.005},
           bias_filler={'type': 'constant', 'value': 0.1}):
    fc = L.InnerProduct(bottom, num_output=nout, param=param,
                       weight_filler=weight_filler, bias_filler=bias_filler)
    return fc

def fc_relu(bottom, nout, param=learned_param,
            weight_filler={'type': 'gaussian', 'std': 0.005},
           bias_filler={'type': 'constant', 'value': 0.1}):
    '''FC + ReLU'''
    fc = L.InnerProduct(bottom, num_output=nout, param=param,
                       weight_filler=weight_filler, bias_filler=bias_filler)
    return fc, L.ReLU(fc, in_place=False)

def max_pool(bottom, ks, stride=1):
    '''MAX Pooling'''
    return L.Pooling(bottom, pool=P.Pooling.MAX, kernel_size=ks, stride=stride)

def accuracy_top1_top5(bottom, label):
    accuracy_top1 = L.Accuracy(bottom, label, accuracy_param=dict(top_k=1))
    accuracy_top5 = L.Accuracy(bottom, label, accuracy_param=dict(top_k=5))
    return accuracy_top1, accuracy_top5

def caffenet(data, label=None, train=True, num_classes=1000, classifier_name='fc8',
        learn_all=False, filename='caffenet.prototxt'):
    '''Returns a NetSpec specifying CaffeNet'''
    n = caffe.NetSpec()
    n.data = data # 1st input data layer
    param = learned_param if learn_all else frozen_param # use learnable parameter
    n.conv1, n.relu1 = conv_relu(n.data, ks=11, nout=96, stride=4, param=param)
    n.pool1 = max_pool(n.relu1, ks=3, stride=2)
    n.norm1 = L.LRN(n.pool1, local_size=5, alpha=1e-4, beta=0.75)
    n.conv2, n.relu2 = conv_relu(n.norm1, ks=5, nout=256, pad=2, group=2, param=param)
    n.pool2 = max_pool(n.relu2, ks=3, stride=2)
    n.norm2 = L.LRN(n.pool2, local_size=5, alpha=1e-4, beta=0.75)
    n.conv3, n.relu3 = conv_relu(n.norm2, ks=3, nout=384, pad=1, param=param)
    n.conv4, n.relu4 = conv_relu(n.relu3, ks=3, nout=384, pad=1, group=2, param=param)
    n.conv5, n.relu5 = conv_relu(n.relu4, ks=3, nout=256, pad=1, group=2, param=param)
    n.pool5 = max_pool(n.relu5, ks=3, stride=2)
    n.fc6, n.relu6 = fc_relu(n.pool5, nout=4096, param=param)
    if train:
        n.drop6 = fc7input = L.Dropout(n.relu6, in_place=False) # dropout exists in train,
                                                                # can remove
    else:
        fc7input = n.relu6
    n.fc7, n.relu7 = fc_relu(fc7input, nout=4096, param=param)
    if train:
        n.drop7 = fc8input = L.Dropout(n.relu7, in_place=False)
    else:
        fc8input = n.relu7
    # always learn fc8 (param=learned_param)
    fc8 = L.InnerProduct(fc8input, num_output=num_classes, param=learned_param)
    # give fc8 the name specified by argument `classifier_name`
    n.__setattr__(classifier_name, fc8)
    # add the accuracy, loss layer
    if not train:
        n.probs = L.Softmax(fc8)
    if label is not None:
        n.label = label
        n.loss = L.SoftmaxWithLoss(fc8, n.label)
        n.acc_top1, n.acc_top5 = accuracy_top1_top5(fc8, n.label)
    # write the net to the prototxt file
    with open(filename, 'w') as f:
        f.write(str(n.to_proto()))

def quantized_caffenet(data, blobs_quantization_params, label=None, train=True, num_classes=1000,
            classifier_name='fc8', learn_all=False, filename='caffenet.prototxt',
            round_method='FLOOR'):
    '''Returns a NetSpec specifying Quantized CaffeNet'''
    n = caffe.NetSpec()
    n.data = data # 1st input data layer
    n.quantized_data = L.Quantization(n.data, quantization_param=blobs_quantization_params['data'])
    param = learned_param if learn_all else frozen_param # use learnable parameter

    #n.conv1, n.relu1 = conv_relu(n.quantized_data, ks=11, nout=96, stride=4, param=param)
    n.conv1 = conv(n.quantized_data, ks=11, nout=96, stride=4, param=param)
    n.quantized_conv1 = L.Quantization(n.conv1, quantization_param=blobs_quantization_params['conv1'])
    n.relu1 = L.ReLU(n.quantized_conv1, in_place=False)
    n.quantized_relu1 = L.Quantization(n.relu1, quantization_param=blobs_quantization_params['relu1'])
    n.pool1 = max_pool(n.quantized_relu1, ks=3, stride=2)
    n.quantized_pool1 = L.Quantization(n.pool1, quantization_param=blobs_quantization_params['pool1'])
    n.norm1 = L.LRN(n.quantized_pool1, local_size=5, alpha=1e-4, beta=0.75)
    n.quantized_norm1 = L.Quantization(n.norm1, quantization_param=blobs_quantization_params['norm1'])

    #n.conv2, n.relu2 = conv_relu(n.quantized_norm1, ks=5, nout=256, pad=2, group=2, param=param)
    n.conv2 = conv(n.quantized_norm1, ks=5, nout=256, pad=2, group=2, param=param)
    n.quantized_conv2 = L.Quantization(n.conv2, quantization_param=blobs_quantization_params['conv2'])
    n.relu2 = L.ReLU(n.quantized_conv2, in_place=False)
    n.quantized_relu2 = L.Quantization(n.relu2, quantization_param=blobs_quantization_params['relu2'])
    n.pool2 = max_pool(n.quantized_relu2, ks=3, stride=2)
    n.quantized_pool2 = L.Quantization(n.pool2, quantization_param=blobs_quantization_params['pool2'])
    n.norm2 = L.LRN(n.quantized_pool2, local_size=5, alpha=1e-4, beta=0.75)
    n.quantized_norm2 = L.Quantization(n.norm2, quantization_param=blobs_quantization_params['norm2'])

    #n.conv3, n.relu3 = conv_relu(n.quantized_norm2, ks=3, nout=384, pad=1, param=param)
    n.conv3 = conv(n.quantized_norm2, ks=3, nout=384, pad=1, param=param)
    n.quantized_conv3 = L.Quantization(n.conv3, quantization_param=blobs_quantization_params['conv3'])
    n.relu3 = L.ReLU(n.quantized_conv3, in_place=False)
    n.quantized_relu3 = L.Quantization(n.relu3, quantization_param=blobs_quantization_params['relu3'])

    #n.conv4, n.relu4 = conv_relu(n.quantized_relu3, ks=3, nout=384, pad=1, group=2, param=param)
    n.conv4 = conv(n.quantized_relu3, ks=3, nout=384, pad=1, group=2, param=param)
    n.quantized_conv4 = L.Quantization(n.conv4, quantization_param=blobs_quantization_params['conv4'])
    n.relu4 = L.ReLU(n.quantized_conv4, in_place=False)
    n.quantized_relu4 = L.Quantization(n.relu4, quantization_param=blobs_quantization_params['relu4'])

    #n.conv5, n.relu5 = conv_relu(n.quantized_relu4, ks=3, nout=256, pad=1, group=2, param=param)
    n.conv5 = conv(n.quantized_relu4, ks=3, nout=256, pad=1, group=2, param=param)
    n.quantized_conv5 = L.Quantization(n.conv5, quantization_param=blobs_quantization_params['conv5'])
    n.relu5 = L.ReLU(n.quantized_conv5, in_place=False)
    n.quantized_relu5 = L.Quantization(n.relu5, quantization_param=blobs_quantization_params['relu5'])
    n.pool5 = max_pool(n.quantized_relu5, ks=3, stride=2)
    n.quantized_pool5 = L.Quantization(n.pool5, quantization_param=blobs_quantization_params['pool5'])

    #n.fc6, n.relu6 = fc_relu(n.pool5, nout=4096, param=param)
    n.fc6 = fc(n.quantized_pool5, nout=4096, param=param)
    n.quantized_fc6 = L.Quantization(n.fc6, quantization_param=blobs_quantization_params['fc6'])
    n.relu6 = L.ReLU(n.quantized_fc6, in_place=False)
    n.quantized_relu6 = L.Quantization(n.relu6, quantization_param=blobs_quantization_params['relu6'])
    if train:
        n.drop6 = L.Dropout(n.quantized_relu6, in_place=False) # dropout exists in train,
                                                                # can remove
        fc7input = n.quantized_drop6 = L.Quantization(n.drop6, quantization_param=blobs_quantization_params['drop6'])
    else:
        fc7input = n.quantized_relu6

    #n.fc7, n.relu7 = fc_relu(fc7input, nout=4096, param=param)
    n.fc7 = fc(fc7input, nout=4096, param=param)
    n.quantized_fc7 = L.Quantization(n.fc7, quantization_param=blobs_quantization_params['fc7'])
    n.relu7 = L.ReLU(n.quantized_fc7, in_place=False)
    n.quantized_relu7 = L.Quantization(n.relu7, quantization_param=blobs_quantization_params['relu7'])
    if train:
        n.drop7 = L.Dropout(n.quantized_relu7, in_place=False)
        fc8input = n.quantized_drop7 = L.Quantization(n.drop7, quantization_param=blobs_quantization_params['drop7'])
    else:
        fc8input = n.quantized_relu7
    # always learn fc8 (param=learned_param)
    fc8 = L.InnerProduct(fc8input, num_output=num_classes, param=learned_param)
    n.quantized_fc8 = L.Quantization(fc8, quantization_param=blobs_quantization_params['fc8'])
    # give fc8 the name specified by argument `classifier_name`
    n.__setattr__(classifier_name, fc8)
    # add the accuracy, loss layer
    if not train:
        n.probs = L.Softmax(n.quantized_fc8)
    if label is not None:
        n.label = label
        n.loss = L.SoftmaxWithLoss(n.quantized_fc8, n.label)
        n.acc_top1, n.acc_top5 = accuracy_top1_top5(n.quantized_fc8, n.label)
    # write the net to the prototxt file
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
