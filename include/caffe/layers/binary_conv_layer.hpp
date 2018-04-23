#ifndef CAFFE_BINARY_CONV_LAYER_HPP_
#define CAFFE_BINARY_CONV_LAYER_HPP_

#include <vector>

#include "caffe/blob.hpp"
#include "caffe/layer.hpp"
#include "caffe/proto/caffe.pb.h"

#include "caffe/layers/base_conv_layer.hpp"

namespace caffe {

#define sign(x) ((x)>=0?1:-1)

/**
 * @brief binary convlution layer
 */
template <typename Dtype>
class BinaryConvolutionLayer : public BaseConvolutionLayer<Dtype> {
 public:
  explicit BinaryConvolutionLayer(const LayerParameter& param)
      : BaseConvolutionLayer<Dtype>(param) {}
  virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual inline const char* type() const { return "BinaryConvolution"; }

 protected:
  virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Forward_gpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Backward_cpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);
  virtual void Backward_gpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);
  virtual inline bool reverse_dimensions() { return false; }
  virtual void compute_output_shape();
  virtual void binarizeCPUTo(const Blob<Dtype>* weights, Blob<Dtype>* wb);
#ifndef CPU_ONLY
  virtual void binarizeGPUTo(const Blob<Dtype>* weights, Blob<Dtype>* wb);
#endif

 private:
  Blob<Dtype> binary_w_;
  Blob<Dtype> w_buffer_;

  Blob<Dtype> meancenter_;
  Blob<Dtype> A_;

	Blob<Dtype> multiplier_;
};

}  // namespace caffe

#endif  // CAFFE_BINARY_CONV_LAYER_HPP_
