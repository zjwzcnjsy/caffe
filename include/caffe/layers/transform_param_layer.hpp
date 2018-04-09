#ifndef CAFFE_TRANSFORM_PARAM_LAYER_HPP_
#define CAFFE_TRANSFORM_PARAM_LAYER_HPP_

#include <vector>

#include "caffe/blob.hpp"
#include "caffe/layer.hpp"
#include "caffe/proto/caffe.pb.h"

namespace caffe {

template <typename Dtype>
void bestFit(const int num_landmark, const Dtype* dest, const Dtype* src, Dtype *transformed_src, Dtype *T);

/**
 * @brief DAN网络中求当前形状变换到初始形状的变换参数。
 *
 * @param bottom input Blob vector (length 1)
 *   -# @f$ (N \times num_landmark*2) @f$
 *      当前形状
 * @param top output Blob vector (length 1)
 *   -# @f$ (1 \times num_landmark*2) @f$
 *      初始形状
 */
template <typename Dtype>
class TransformParamLayer : public Layer<Dtype> {
 public:
  explicit TransformParamLayer(const LayerParameter& param)
      : Layer<Dtype>(param) {}
  virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Reshape(const vector<Blob<Dtype>*>& bottom,
	  const vector<Blob<Dtype>*>& top);

  virtual inline const char* type() const { return "TransformParam"; }
  virtual inline int ExactNumBottomBlobs() const { return 2; }
  virtual inline int ExactNumTopBlobs() const { return 1; }

 protected:
  /// @copydoc AbsValLayer
  virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Forward_gpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Backward_cpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);
  virtual void Backward_gpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);

  int num_landmark_;
  Blob<Dtype> tmp_landmark_;
};

}  // namespace caffe

#endif  // CAFFE_TRANSFORM_PARAM_LAYER_HPP_
