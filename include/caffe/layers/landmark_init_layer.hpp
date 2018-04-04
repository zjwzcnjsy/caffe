#ifndef CAFFE_LANDMARK_INIT_LAYER_HPP_
#define CAFFE_LANDMARK_INIT_LAYER_HPP_

#include <vector>

#include "caffe/blob.hpp"
#include "caffe/layer.hpp"
#include "caffe/proto/caffe.pb.h"

namespace caffe {

/**
 * @brief DAN网络中第一个阶段的shape更新，回归的增量与平均shape相加。
 *
 * @param bottom input Blob vector (length 1)
 *   -# @f$ (N \times num_landmark*2) @f$
 *      回归的形状增量
 * @param top output Blob vector (length 1)
 *   -# @f$ (N \times num_landmark*2) @f$
 *      初始形状
 */
template <typename Dtype>
class LandmarkInitLayer : public Layer<Dtype> {
 public:
  explicit LandmarkInitLayer(const LayerParameter& param)
      : Layer<Dtype>(param) {}
  virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Reshape(const vector<Blob<Dtype>*>& bottom,
	  const vector<Blob<Dtype>*>& top);

  virtual inline const char* type() const { return "LandmarkInit"; }
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
};

}  // namespace caffe

#endif  // CAFFE_LANDMARK_INIT_LAYER_HPP_
