#ifndef CAFFE_QUANTIZATION_LAYER_HPP_
#define CAFFE_QUANTIZATION_LAYER_HPP_

#include <vector>

#include "caffe/blob.hpp"
#include "caffe/layer.hpp"
#include "caffe/proto/caffe.pb.h"

#include "caffe/layers/neuron_layer.hpp"

namespace caffe {

/**
 * @brief The quantization layer emulates the quantization results
 *		  of input data given by the specified total quantization
 *		  bit width.
 */
template <typename Dtype>
class QuantizationLayer : public NeuronLayer<Dtype> {
public:
	/**
	* @param param provides QuantizationParameter quantization_param,
	*     with QuantizationLayer configuration, including
	*     - bit_width: total bit width of quantization
	*     - range: a list containing the range of input data
	*     - round_method: including FLOOR, CEIL, ROUND, and TRUNC
	*/
	explicit QuantizationLayer(const LayerParameter& param)
		: NeuronLayer<Dtype>(param) {}
	virtual ~QuantizationLayer() {}

	// parse the parameters from the prototxt
	virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom,
	      const vector<Blob<Dtype>*>& top);

	virtual inline const char* type() const { return "Quantization"; }

protected:
	/**
	* @param bottom input Blob vector (length 1)
	*   -# @f$ (N \times C \times H \times W) @f$
	*      the inputs @f$ x @f$
	* @param top output Blob vector (length 1)
	*   -# @f$ (N \times C \times H \times W) @f$
	*      the computed outputs @f$
	*        y = quantization of input blob x
	*/
	virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
			const vector<Blob<Dtype>*>& top);

	/* No GPU Implementation, call Forward CPU by default */

	/* No Backward Implementation, QuantizationLayer can only be used in
	 * the forward pass
	 */
	virtual void Backward_cpu(const vector<Blob<Dtype>*>& top,
			const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
		if (propagate_down[0]) {
			NOT_IMPLEMENTED;
		}
	}

private:
	// instance variables
	int bit_width_;										// total bit width of quantization scheme
	QuantizationParameter_RoundMethod round_method_;	// round method of quantization scheme
	float range_low_, range_high_;						// low / high range
	QuantizationParameter_RoundStrategy round_strategy_;// round strategy of quantization scheme

private:
	/**
	 * Fixed point quantization using a pre-determined scaling factor
	 * integer and fraction width.
	 * @param input_data
	 * 		- input data to be quantized
	 * @param scaling_factor
	 * 		- scaling factor for quantization
	 * @param min/max_value
	 * 		- min or max value can be represented by the encoding scheme
	 */
	Dtype fixed_point(const Dtype& input_data, const double& scaling_factor,
			const double& min_value, const double& max_value) const;

	/**
	 * private method to automatically calculate the scaling factor for fixed point
	 * quantization
	 * @ param scaling_factor
	 * 		- scaling factor based on the data range
	 * @ param min/max_value
	 * 		- min or max value can be represented by the encoding scheme
	 */
	void analyze_scaling_factor(double& scaling_factor, double& min_value,
			double& max_value) const;
};

}		// namespace caffe

#endif /* CAFFE_QUANTIZATION_LAYER_HPP_ */
