# caffe-quantization

Added quantization layer into caffe (support a coarse level fixed point simulation). The working flow
is to insert the newly-built `QuantizationLayer` between computation data path. It will round the 
output of the previous layer using the specified round method and strategy under a pre-determined 
bit width.

## Installation
Only 3 files needed to be added to the original Caffe framework (https://github.com/BVLC/caffe).

- `quantization_layer.hpp`: header file for `QuantizationLayer`, add to `$CAFFE_ROOT/include/caffe/layers`
- `QuantizationLayer.cpp`: source file for `QuantizationLayer`, add to `$CAFFE_ROOT/src/caffe/layers`
- `caffe.proto`: Added id = 147 `QuantizationParameter` for `QuantizationLayer`, replace the original one
  at `$CAFFE_ROOT/src/caffe/proto`

Re-Compile the Caffe framework and PyCaffe/MatCaffe if you use the Python/Matlab Interface.

## Usage
Three IPython Notebooks are provided to demonstrate how to use PyCaffe for manipulating the 
`QuantizationLayer`.
