#ifndef CAFFE_CONV_DW_LAYER_HPP_
#define CAFFE_CONV_DW_LAYER_HPP_

#include <vector>
#include "caffe/blob.hpp"
#include "caffe/layer.hpp"
#include "caffe/proto/caffe.pb.h"

namespace caffe {

template <typename Dtype>
class ConvolutionDepthwiseLayer : public Layer<Dtype> {
 public:
  explicit ConvolutionDepthwiseLayer(const LayerParameter& param)
      : Layer<Dtype>(param) {}
  virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual inline int ExactNumBottomBlobs() const { return 1; }
  virtual inline int ExactNumTopBlobs() const { return 1; }
  virtual inline const char* type() const { return "ConvolutionDepthwise"; }
 protected:
  virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Forward_gpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Backward_cpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);
  virtual void Backward_gpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);
  int kernel_h_;
  int kernel_w_;
  int stride_h_;
  int stride_w_;
  int pad_h_;
  int pad_w_;
  int dilation_h_;
  int dilation_w_;
  Blob<Dtype> weight_buffer_;
  Blob<Dtype> weight_multiplier_;
  Blob<Dtype> bias_buffer_;
  Blob<Dtype> bias_multiplier_;
};

}  // namespace caffe

#endif  // CAFFE_CONV_DW_LAYER_HPP_
