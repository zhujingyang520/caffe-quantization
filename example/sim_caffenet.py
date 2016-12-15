#################################################
# This module exports functions for running the
# simulation of CaffeNet on the specfied dataset
#################################################

import sys
# point to the caffe root, modify the following path if not
caffe_root = '../../'
sys.path.insert(0, caffe_root+'python')
import caffe
from caffe import layers as L, params as P
from caffenet_prototxt import caffenet, quantized_caffenet, convert_to_quantization_param
import tempfile, os
import numpy as np
import math
from fixed_point import FixedPoint

# global variable: caffenet pretrained weights
caffenet_weights = caffe_root + '/models/bvlc_reference_caffenet/bvlc_reference_caffenet.caffemodel'

def sim_floating_point_caffenet(LMDB_filename, batch_size=10, iterations=10, verbose=False):
    '''Simlulate the floating point CaffeNet on the specified LMDB dataset,
       where the batch size of running CaffeNet and the total iterations are
       specified by the arguments.
    '''
    # step 1. create the data & label layer
    # create input data layer for caffenet accuracy
    mean_file = caffe_root + 'data/ilsvrc12/imagenet_mean.binaryproto'
    ilsvrc_data, ilsvrc_label = L.Data(batch_size=batch_size, backend=P.Data.LMDB,
            source=LMDB_filename, ntop=2, transform_param={'crop_size': 227,
            'mirror': False, 'mean_file': mean_file})

    # step 2. generate the caffenet prototxt definition to a temporary file
    f = tempfile.NamedTemporaryFile(delete=False)
    #print 'Generate Prototxt File to %s' % (f.name,)
    f.close()
    # generate the prototxt
    caffenet(ilsvrc_data, label=ilsvrc_label, train=False, num_classes=1000,
            classifier_name='fc8', learn_all=False, filename=f.name)

    # step 3. create the net
    floating_point_caffenet = None
    floating_point_caffenet = caffe.Net(f.name, caffenet_weights, caffe.TEST)

    # step 4. do feedforward and record the range & accuracy
    accuracy = np.zeros((2, iterations)) # 2: top1 & top5
    blobs_range, blobs_range_ = {}, {}
    for blob_name in floating_point_caffenet.blobs:
        blobs_range[blob_name] = np.zeros(2)
        blobs_range_[blob_name] = np.zeros((2, iterations))

    print 'Floating point CaffNet starts inference...'
    for i in range(iterations):
        floating_point_caffenet.forward()
        if verbose:
            print 'Batch %d/%d: Top-1 Acc: %.2f%%, Top-5 Acc: %.2f%%' % (i, iterations,
                    floating_point_caffenet.blobs['acc_top1'].data*100.,
                    floating_point_caffenet.blobs['acc_top5'].data*100.)
        elif i % 10 == 0:
            print 'Batch %d/%d: Top-1 Acc: %.2f%%, Top-5 Acc: %.2f%%' % (i, iterations,
                    floating_point_caffenet.blobs['acc_top1'].data*100.,
                    floating_point_caffenet.blobs['acc_top5'].data*100.)
        accuracy[0, i] = floating_point_caffenet.blobs['acc_top1'].data
        accuracy[1, i] = floating_point_caffenet.blobs['acc_top5'].data
        for blob_name in floating_point_caffenet.blobs:
            blobs_range_[blob_name][0, i] = floating_point_caffenet.blobs[blob_name].\
                    data.min()
            blobs_range_[blob_name][1, i] = floating_point_caffenet.blobs[blob_name].\
                    data.max()

    # calculate the final blobs range (max & min values)
    for blob_name in blobs_range:
        blobs_range[blob_name] = [blobs_range_[blob_name][0].min(), blobs_range_[blob_name][1].max()]


    # step 5. record the weights & biases range
    weights_range, biases_range = {}, {}
    for param_name, param in floating_point_caffenet.params.items():
       weights_range[param_name] = np.zeros(2) # 2: min & max
       biases_range[param_name] = np.zeros(2) # 2: min & max
       weights_range[param_name][0] = param[0].data.min()
       weights_range[param_name][1] = param[0].data.max()
       biases_range[param_name][0] = param[1].data.min()
       biases_range[param_name][1] = param[1].data.max()

    # obtain the kernel name, used for further evaluation of floating on different kernels
    kernels_name = floating_point_caffenet.params.keys()

    # step 6. clean the network and file
    del floating_point_caffenet
    os.unlink(f.name)

    return accuracy, blobs_range, weights_range, biases_range, kernels_name

def sim_fixed_point_caffenet(LMDB_filename, bit_width, blobs_range,
        weights_range, biases_range, batch_size=10, iterations=10, round_method='FLOOR',
        round_strategy='CONSERVATIVE', verbose=False):
    '''simluate the fixed point caffenet. It will automatically determine
       the fraction bit width based on the range of activations, weights & biases.
       The integer bit width is determined by the specified bit width.
       - bit_width: a dictionary contains the bitwidth of `weights`, `biases`, and
         `blobs` respectively.
    '''
    # step 1. create data layer & label layer
    mean_file = caffe_root + 'data/ilsvrc12/imagenet_mean.binaryproto'
    ilsvrc_data, ilsvrc_label = L.Data(batch_size=batch_size, backend=P.Data.LMDB,
            source=LMDB_filename, ntop=2, transform_param={'crop_size': 227,
            'mirror': False, 'mean_file': mean_file})

    # step 2. generates the quantization parameters
    blobs_quantization_params = quantization_param_wrapper(bit_width['blobs'],
            blobs_range, round_method=round_method, round_strategy=round_strategy)
    weights_quantization_params = quantization_param_wrapper(bit_width['weights'],
            weights_range, round_method=round_method, round_strategy=round_strategy)
    biases_quantization_params = quantization_param_wrapper(bit_width['biases'],
            biases_range, round_method=round_method, round_strategy=round_strategy)

    # step 3. generate the caffenet prototxt definition to a temporary file
    f = tempfile.NamedTemporaryFile(delete=False)
    #print 'Generate Prototxt File to %s' % (f.name,)
    f.close()
    # generate the prototxt
    quantized_caffenet(ilsvrc_data, blobs_quantization_params, label=ilsvrc_label,
            train=False, num_classes=1000, classifier_name='fc8',
            learn_all=False, filename=f.name)

    # step 4. create the net
    fixed_point_caffenet = None
    fixed_point_caffenet = caffe.Net(f.name, caffenet_weights, caffe.TEST)

    # step 5. quantize the weights & biases based on the quantization parameters
    for k, v in fixed_point_caffenet.params.items():
        # 1. quantize the weights
        WFixedPoint = FixedPoint(weights_quantization_params[k]['range'],
                weights_quantization_params[k]['bit_width'], round_method=round_method,
                round_strategy=round_strategy)
        v[0].data[...] = WFixedPoint.quantize(v[0].data)

        # 2. quantize the biases
        BFixedPoint = FixedPoint(biases_quantization_params[k]['range'],
                biases_quantization_params[k]['bit_width'], round_method=round_method,
                round_strategy=round_strategy)
        v[1].data[...] = BFixedPoint.quantize(v[1].data)

    # step 6. feedforward the neural network
    accuracy = np.zeros((2, iterations)) # 2: top1 & top5
    print 'Fixed point CaffeNet starts inference...'
    for i in range(iterations):
        fixed_point_caffenet.forward()
        if verbose:
            print 'Batch %d/%d: Top-1 Acc: %.2f%%, Top-5 Acc: %.2f%%' % (i, iterations,
                    fixed_point_caffenet.blobs['acc_top1'].data*100.,
                    fixed_point_caffenet.blobs['acc_top5'].data*100.)
        elif i % 10 == 0:
            print 'Batch %d/%d: Top-1 Acc: %.2f%%, Top-5 Acc: %.2f%%' % (i, iterations,
                    fixed_point_caffenet.blobs['acc_top1'].data*100.,
                    fixed_point_caffenet.blobs['acc_top5'].data*100.)

        accuracy[0, i] = fixed_point_caffenet.blobs['acc_top1'].data
        accuracy[1, i] = fixed_point_caffenet.blobs['acc_top5'].data


    # step 7. clean the network and file
    del fixed_point_caffenet
    os.unlink(f.name)

    return accuracy


def quantization_param_wrapper(bit_width, data_range, round_method='FLOOR',
        round_strategy='CONSERVATIVE'):
    '''wrapper function to get the quantization parameters. Mainly supports for
       singleton bit width.
    '''
    dict_bit_width = {}
    if type(bit_width) is int:
        for k in data_range:
            dict_bit_width[k] = bit_width
    elif type(bit_width) is dict:
        dict_bit_width = bit_width
    else:
        raise TypeError('unexpected data type of bitwidth')

    return convert_to_quantization_param(dict_bit_width, data_range,
            round_method=round_method, round_strategy=round_strategy)


################################################################################
# deprecated function, never use
################################################################################
def determine_quantization_params(bit_width, data_range, round_method='FLOOR'):
    '''helper function to determine the quantization parameters: integer, fraction, and
       rounding method
       - bit_width: integer or dictionary to support a fine granularity bit width
       - data_range: dictionary for each layer's range
    '''
    data_quantization = {}
    for k, v in data_range.items():
        data_quantization[k] = np.zeros(2, dtype=np.int) # 2 quantization: I + F
        max_abs_range = np.max(np.abs(v))
        assert max_abs_range > 0, 'unexpected constant 0 range'
        # integer: determined by the max absolute range @ index 0
        data_quantization[k][0] = math.ceil(math.log(max_abs_range, 2))

        # fraction: determined by the bit width & fraction width @ index 1
        if type(bit_width) is int:
            data_quantization[k][1] = bit_width - 1 - (data_quantization[k][0] > 0) * \
                    data_quantization[k][0]
        elif type(bit_width) is dict:
            data_quantization[k][1] = bit_width[k] - 1 - (data_quantization[k][0] > 0) * \
                    data_quantization[k][0]
            #print 'set BW of %s to be %d' % (k, bit_width[k])
        else:
            raise TypeError("unexpected data type of bitwidth")

        assert data_quantization[k][1] >= 0, 'not enough bit width'

    # normalized to the quantization parameter format
    return convert_to_quantization_param(data_quantization, round_method=round_method)
