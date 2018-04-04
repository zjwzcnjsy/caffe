#ifndef CAFFE_LANDMARK_IMAGE_LAYER_HPP_
#define CAFFE_LANDMARK_IMAGE_LAYER_HPP_

#include <vector>

#include "caffe/blob.hpp"
#include "caffe/layer.hpp"
#include "caffe/proto/caffe.pb.h"

namespace caffe {

/**
 * @brief DAN网络中根据变换后的当前形状生成HeatMap图像。
 *
 * @param bottom input Blob vector (length 1)
 *   -# @f$ (N \times num_landmark*2) @f$
 *      当前变换后的形状
 * @param top output Blob vector (length 1)
 *   -# @f$ (N \times channels \times height \times width) @f$
 *      生成的Heatmap图像
 */
template <typename Dtype>
class LandmarkImageLayer : public Layer<Dtype> {
 public:
  explicit LandmarkImageLayer(const LayerParameter& param)
      : Layer<Dtype>(param) {}
  virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Reshape(const vector<Blob<Dtype>*>& bottom,
	  const vector<Blob<Dtype>*>& top);

  virtual inline const char* type() const { return "LandmarkImage"; }
  virtual inline int ExactNumBottomBlobs() const { return 1; }
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
  int img_height_;
  int img_width_;
  int landmark_patch_size_;

  Blob<Dtype> tmp_landmark_images_;
};

}  // namespace caffe

#endif  // CAFFE_LANDMARK_IMAGE_LAYER_HPP_
