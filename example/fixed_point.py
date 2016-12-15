import numpy as np

## deprecated version of fixed_point
def fixed_point(x, integer_width, fraction_width, round_method):
    '''fixed_point routine: changes the given input x into the specified
    quantization scheme representation. An equivalent implementation in
    quantization_layer class'''

    #print 'int: %d, frac: %d, round: %s' % (integer_width, fraction_width, round_method)

    scaling_factor = 2. ** (fraction_width +    # support for negative integer width
            (integer_width < 0) * abs(integer_width))
    max_region = 2. ** (integer_width * (integer_width > 0) + fraction_width)
    if round_method.upper() == 'ROUND':
        input_data_rounded = np.round(x * scaling_factor)
    elif round_method.upper() == 'FLOOR':
        input_data_rounded = np.floor(x * scaling_factor)
    elif round_method.upper() == 'CEIL':
        input_data_rounded = np.ceil(x * scaling_factor)
    elif round_method.upper() == 'TRUNC':
        input_data_rounded = np.trunc(x * scaling_factor)
    else:
        raise TypeError('undefined round method: %s' % (round_method,))

    return np.minimum(np.maximum(-max_region, input_data_rounded), max_region-1)/scaling_factor

## General fixed point simulation
class FixedPoint(object):
    '''
    FixedPoint exports a class to automatically analyze the range of input data.
    '''
    def __init__(self, input_range, N, round_method='FLOOR', round_strategy='NEUTRAL'):
        '''
        Constructor to pass the range [min, max] for the current layer

        Parameters
        -----------
        - input_range: a list with the form [min, max]
        - N: the total bitwidth (includes sign bit) for current layer representation
        - round_method: valid rounding method {'FLOOR', 'ROUND', 'CEIL', 'TRUNC'}
        - round_strategy: valid rounding strategy {'CONSERVATIVE', 'NEUTRAL', 'AGGRESSIVE'}
        '''
        # sanity check
        assert N > 0, 'non-positive bitwidth %d' % (N,)
        assert len(input_range), 'unexpected range specified'
        assert input_range[0] <= input_range[1], 'unexpected min & max values for the range'

        self.input_range = input_range
        self.N = N
        self.round_method = round_method
        self.round_strategy = round_strategy

    def __analyze_scaling_factor(self):
        '''
        Analyze the scaling factor for the given input range and bitwidth
        '''
        # A smart encoding is adopted here, if the input data are uni-polarity
        # (all pos/neg), we use unsigned encoding, otherwise, we adopts 2s complement
        if self.input_range[0] >= 0:
            # non-negative N-bit range: [0, 2^N-1]
            self.min_value, self.max_value = 0., 2.**self.N-1
        elif self.input_range[1] <= 0:
            # non-positive N-bit range: [-(2^N-1), 0]
            self.min_value, self.max_value = -2.**self.N+1, 0
        else:
            # 2s complement [-2^{N-1}, 2^{N-1}-1]
            self.min_value, self.max_value = -(2.**(self.N-1)), (2.**(self.N-1))-1

        # analyze the scaling factor based on range and N-bit representation range
        neg_scaling_factor = np.log2(self.min_value/self.input_range[0]) \
                if self.input_range[0] < 0 else float('inf')
        pos_scaling_factor = np.log2(self.max_value/self.input_range[1]) \
                if self.input_range[1] > 0 else float('inf')
        # for binary number, the scaling factor should be only power of 2
        # we adopts different rounding strategies
        if self.round_strategy.upper() == 'CONSERVATIVE':
            self.scaling_factor = 2**np.floor(min(pos_scaling_factor, neg_scaling_factor))
        elif self.round_strategy.upper() == 'NEUTRAL':
            self.scaling_factor = 2**np.round(min(pos_scaling_factor, neg_scaling_factor))
        elif self.round_strategy.upper() == 'AGGRESSIVE':
            self.scaling_factor = 2**np.ceil(min(pos_scaling_factor, neg_scaling_factor))
        else:
            raise TypeError('undefined rounding strategy %s' % (self.round_strategy,))

    def quantize(self, in_values):
        '''
        Quantize the input values using N-bit

        Parameters
        -----------
        - in_values: original inputs, the numpy array of arbitary shape

        Return
        -------
        - out_values: quantized results with the same shape as in_values
        '''
        self.__analyze_scaling_factor()     # analyze the scaling factor first

        if self.round_method.upper() == 'ROUND':
            in_values_scaling = np.round(in_values * self.scaling_factor)
        elif self.round_method.upper() == 'FLOOR':
            in_values_scaling = np.floor(in_values * self.scaling_factor)
        elif self.round_method.upper() == 'CEIL':
            in_values_scaling = np.ceil(in_values * self.scaling_factor)
        elif self.round_method.upper() == 'TRUNC':
            in_values_scaling = np.trunc(in_values * self.scaling_factor)
        else:
            raise TypeError('undefined round method %s' % (self.round_method,))

        return np.minimum(np.maximum(self.min_value, in_values_scaling), self.max_value) \
                / self.scaling_factor
