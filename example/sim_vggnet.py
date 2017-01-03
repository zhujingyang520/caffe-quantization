###################################################
# This module exports functions for running the
# simulation of VGGNet-16 on the specified dataset
###################################################

import sys, tempfile, os
# point to the caffe root, modify the following path if not
caffe_root = '../../'
sys.path.insert(0, caffe_root+'python')
import caffe
import numpy as np
from caffe import layers as L, params as P
from vggnet_prototxt import vggnet_16_proto, quantized_vggnet_16_proto, \
        convert_to_quantization_param
from fixed_point import FixedPoint
import collections

# global variable: vggnet-16 pretrained weights
vggnet_16_weights = caffe_root + '/models/vggnet/VGG_ILSVRC_16_layers.caffemodel'

## Helper function to run the simulation for floating point vggnet-16
def sim_floating_point_vggnet_16(LMDB_filename, batch_size=10, iterations=10, verbose=False):
    """Simulates the floating point VGGNet-16 on the specified LMDB dataset,
       where the batch size of running VGGNet-16 and the total iterations are
       specified by the input arguments.
    """
    # step 1. create the data & label layer
    mean_value = [104, 117, 123] # per-channel mean value [B, G, R]
    ilsvrc_data, ilsvrc_label = L.Data(batch_size=batch_size, backend=P.Data.LMDB,
            source=LMDB_filename, ntop=2,
            transform_param={'crop_size': 224, 'mirror': False, 'mean_value': mean_value})

    # step 2. generate the vggnet-16 prototxt definition to a temporary file
    f = tempfile.NamedTemporaryFile(delete=False)
    f.close()
    # write to the prototxt
    vggnet_16_proto(ilsvrc_data, label=ilsvrc_label, train=False, classifier_name='fc8',
            learn_all=False, filename=f.name)

    # step 3. create the net
    floating_point_vggnet_16 = None
    floating_point_vggnet_16 = caffe.Net(f.name, vggnet_16_weights, caffe.TEST)

    # step 4. do feedforward and record the range & accuracy
    accuracy = np.zeros((2, iterations))
    blobs_range, blobs_range_ = {}, {}
    for blob_name in floating_point_vggnet_16.blobs:
        blobs_range[blob_name] = np.zeros(2)
        blobs_range_[blob_name] = np.zeros((2, iterations)) # range for each iteration

    print 'Floating point VGGNet-16 starts inference...'
    for i in range(iterations):
        floating_point_vggnet_16.forward()
        if verbose:
            print 'Batch %d/%d: Top-1 Acc: %.2f%%, Top-5 Acc: %.2f%%' % (i, iterations,
                    floating_point_vggnet_16.blobs['acc_top1'].data*100.,
                    floating_point_vggnet_16.blobs['acc_top5'].data*100.)
        elif i % 10 == 0:
            print 'Batch %d/%d: Top-1 Acc: %.2f%%, Top-5 Acc: %.2f%%' % (i, iterations,
                    floating_point_vggnet_16.blobs['acc_top1'].data*100.,
                    floating_point_vggnet_16.blobs['acc_top5'].data*100.)

        # record the accuracy
        accuracy[0, i] = floating_point_vggnet_16.blobs['acc_top1'].data
        accuracy[1, i] = floating_point_vggnet_16.blobs['acc_top5'].data
        # record the blobs (internal feature maps) range
        for blob_name in floating_point_vggnet_16.blobs:
            blobs_range_[blob_name][0, i] = floating_point_vggnet_16.blobs[blob_name].\
                    data.min()
            blobs_range_[blob_name][1, i] = floating_point_vggnet_16.blobs[blob_name].\
                    data.max()

    # record the final blobs range (max & min over all iterations)
    for blob_name in floating_point_vggnet_16.blobs:
        blobs_range[blob_name] = [blobs_range_[blob_name][0].min(),
                blobs_range_[blob_name][1].max()]

    # step 5. record kernels range
    weights_range, biases_range = {}, {}
    for param_name, param in floating_point_vggnet_16.params.iteritems():
        weights_range[param_name] = np.zeros(2)
        biases_range[param_name] = np.zeros(2)
        weights_range[param_name][0] = param[0].data.min()
        weights_range[param_name][1] = param[0].data.max()
        biases_range[param_name][0] = param[1].data.min()
        biases_range[param_name][1] = param[1].data.max()
    # fetch the kernel names & blob names
    kernels_name = floating_point_vggnet_16.params.keys()
    blobs_name = floating_point_vggnet_16.blobs.keys()

    # step 6. clean the network and file
    del floating_point_vggnet_16
    os.unlink(f.name)

    return accuracy, blobs_range, weights_range, biases_range, kernels_name, blobs_name

def sim_fixed_point_vggnet_16(LMDB_filename, bit_width, blobs_range,
        weights_range, biases_range, batch_size=10, iterations=10, round_method='FLOOR',
        round_strategy='CONSERVATIVE', verbose=False):
    """simulates the fixed point vggnet16. It will automatically determine the fraction
    bit with based on the range of the activations, weights & biases.
    - bit_width: a dictionary contains the bit widths of `weights`, `biases`, and `blobs`
      respectively.
    """
    # step 1. create the data & label layer
    mean_value = [104, 117, 123] # per-channel mean value [B, G, R]
    ilsvrc_data, ilsvrc_label = L.Data(batch_size=batch_size, backend=P.Data.LMDB,
            source=LMDB_filename, ntop=2,
            transform_param={'crop_size': 224, 'mirror': False, 'mean_value': mean_value})

    # step 2. generates the quantization parameters
    blobs_quantization_params = quantization_param_wrapper(bit_width['blobs'],
            blobs_range, round_method=round_method, round_strategy=round_strategy)
    weights_quantization_params = quantization_param_wrapper(bit_width['weights'],
            weights_range, round_method=round_method, round_strategy=round_strategy)
    biases_quantization_params = quantization_param_wrapper(bit_width['biases'],
            biases_range, round_method=round_method, round_strategy=round_strategy)

    # step 3. generate the vggnet16 prototxt definition to a temporary file
    f = tempfile.NamedTemporaryFile(delete=False)
    f.close()
    # generate the prototxt
    quantized_vggnet_16_proto(ilsvrc_data, blobs_quantization_params, label=ilsvrc_label,
            train=False, num_classes=1000, classifier_name='fc8', learn_all=False,
            filename=f.name)

    # step 4. create the net
    fixed_point_vggnet_16 = None
    fixed_point_vggnet_16 = caffe.Net(f.name, vggnet_16_weights, caffe.TEST)

    # step 5. quantize the weights & biases based on the quantization parameters
    for k, v in fixed_point_vggnet_16.params.iteritems():
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
    accuracy = np.zeros((2, iterations))
    print 'Fxied point VGGNet16 starts inference...'
    for i in range(iterations):
        fixed_point_vggnet_16.forward()
        if verbose:
            print 'Batch %d/%d: Top-1 Acc: %.2f%%, Top-5 Acc: %.2f%%' % (i, iterations,
                    fixed_point_vggnet_16.blobs['acc_top1'].data*100.,
                    fixed_point_vggnet_16.blobs['acc_top5'].data*100.)
        elif i % 10 == 0:
            print 'Batch %d/%d: Top-1 Acc: %.2f%%, Top-5 Acc: %.2f%%' % (i, iterations,
                    fixed_point_vggnet_16.blobs['acc_top1'].data*100.,
                    fixed_point_vggnet_16.blobs['acc_top5'].data*100.)
        accuracy[0, i] = fixed_point_vggnet_16.blobs['acc_top1'].data
        accuracy[1, i] = fixed_point_vggnet_16.blobs['acc_top5'].data

    # step 7. clean the network and file
    del fixed_point_vggnet_16
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
    elif type(bit_width) is dict or type(bit_width) is collections.OrderedDict:
        dict_bit_width = bit_width
    else:
        raise TypeError('unexpected data type of bitwidth')

    return convert_to_quantization_param(dict_bit_width, data_range,
            round_method=round_method, round_strategy=round_strategy)
