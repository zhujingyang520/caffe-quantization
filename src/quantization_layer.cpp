#include <algorithm>
#include <vector>
#include <limits>

#include "caffe/layers/quantization_layer.hpp"

namespace caffe {

template <typename Dtype>
void QuantizationLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
	      const vector<Blob<Dtype>*>& top) {
	// parses the parameters from *.prototxt and quick sanity check
	bit_width_ = this->layer_param_.quantization_param().bit_width();
	CHECK_GT(bit_width_, 0) << type() << " Layer has unexpected negative bit width";

	round_method_ = this->layer_param_.quantization_param().round_method();
	round_strategy_ = this->layer_param_.quantization_param().round_strategy();

	// read range
	CHECK_EQ(this->layer_param_.quantization_param().range_size(), 2) <<
			type() << " Layer has unexpected number of range";
	range_low_ = std::min(this->layer_param_.quantization_param().range(0),
			this->layer_param_.quantization_param().range(1));
	range_high_ = std::max(this->layer_param_.quantization_param().range(0),
			this->layer_param_.quantization_param().range(1));
}

template <typename Dtype>
void QuantizationLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
		const vector<Blob<Dtype>*>& top) {
	const Dtype* bottom_data = bottom[0]->cpu_data();
	Dtype* top_data = top[0]->mutable_cpu_data();
	const int count = bottom[0]->count();

	// analyze the scaling factor, accompanied with min/max range of data
	double scaling_factor, min_value, max_value;
	analyze_scaling_factor(scaling_factor, min_value, max_value);

	// apply quantization element-wise
	for (int i = 0; i < count; ++i) {
		top_data[i] = fixed_point(bottom_data[i], scaling_factor,
				min_value, max_value);
	}
}

template <typename Dtype>
void QuantizationLayer<Dtype>::analyze_scaling_factor(double& scaling_factor,
		double& min_value, double& max_value) const {
	// smart choosing between 2s complement encoding or unsigned encoding
	if (range_low_ >= 0.0) {
		// non-negative input range with unsigned range [0, 2^N-1]
		min_value = 0.0;
		max_value = pow(2.0, bit_width_) - 1.0;
	} else if (range_high_ <= 0.0) {
		// non-positive input range with unsigned range [-2^N+1, 0]
		min_value = -pow(2.0, bit_width_) + 1.0;
		max_value = 0.0;
	} else {
		// N-bit 2s complement can represent the integer between -2^(N-1)
		// to 2^(N-1)-1
		min_value = -pow(2.0, bit_width_-1);
		max_value = pow(2.0, bit_width_-1) - 1.0;
	}

	// analyze the scaling factor based on min(max)value and range
	// scaling factor should be power of 2
	double neg_scaling_factor = (range_low_ < 0) ? log2(min_value/range_low_) :
			std::numeric_limits<double>::infinity();
	double pos_scaling_factor = (range_high_ > 0) ? log2(max_value/range_high_) :
			std::numeric_limits<double>::infinity();

	switch (round_strategy_) {
	case QuantizationParameter_RoundStrategy_CONSERVATIVE:
		scaling_factor = pow(2.0, floor(std::min(neg_scaling_factor, pos_scaling_factor)));
		break;
	case QuantizationParameter_RoundStrategy_NEUTRAL:
		scaling_factor = pow(2.0, round(std::min(neg_scaling_factor, pos_scaling_factor)));
		break;
	case QuantizationParameter_RoundStrategy_AGGRESSIVE:
		scaling_factor = pow(2.0, ceil(std::min(neg_scaling_factor, pos_scaling_factor)));
		break;
	default:
		LOG(FATAL) << "Unknown round strategy.";
	}

}

template <typename Dtype>
Dtype QuantizationLayer<Dtype>::fixed_point(const Dtype& input_data,
		const double& scaling_factor, const double& min_value,
		const double& max_value) const {
	// rounded results of input data
	double input_data_rounded;

	switch (round_method_) {
	case QuantizationParameter_RoundMethod_ROUND:
		input_data_rounded = round(input_data * (Dtype)scaling_factor);
		break;
	case QuantizationParameter_RoundMethod_FLOOR:
		input_data_rounded = floor(input_data * (Dtype)scaling_factor);
		break;
	case QuantizationParameter_RoundMethod_CEIL:
		input_data_rounded = ceil(input_data * (Dtype)scaling_factor);
		break;
	case QuantizationParameter_RoundMethod_TRUNC:
		input_data_rounded = trunc(input_data * (Dtype)scaling_factor);
		break;
	default:
		LOG(FATAL) << "Unknown round method.";
	}

	return std::min(std::max(input_data_rounded, min_value), max_value) /
			(Dtype)scaling_factor;
}

#ifdef CPU_ONLY
//STUB_GPU(QuantizationLayer);
#endif

INSTANTIATE_CLASS(QuantizationLayer);
REGISTER_LAYER_CLASS(Quantization);

}	// namespace caffe
