#ifndef CAFFE_AFFINE_TRANSFORM_LAYER_HPP_
#define CAFFE_AFFINE_TRANSFORM_LAYER_HPP_

#include <vector>

#include "caffe/blob.hpp"
#include "caffe/layer.hpp"
#include "caffe/proto/caffe.pb.h"

namespace caffe {

/**
 * @brief DAN网络中根据上一阶段的变换参数对输入图像做变换。
 *
 * @param bottom input Blob vector (length 1)
 *   -# @f$ (N \times channels \times height \times width) @f$
 *      输入图像
 * @param bottom input Blob vector (length 1)
 *   -# @f$ (N \times 6) @f$
 *      输入变换参数
 * @param top output Blob vector (length 1)
 *   -# @f$ (N \times channels \times height \times width) @f$
 *      输出变换的图像
 */
template <typename Dtype>
class AffineTransformLayer : public Layer<Dtype> {
 public:
  explicit AffineTransformLayer(const LayerParameter& param)
      : Layer<Dtype>(param) {}
  virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Reshape(const vector<Blob<Dtype>*>& bottom,
	  const vector<Blob<Dtype>*>& top);

  virtual inline const char* type() const { return "AffineTransform"; }
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

  int num_;
  int channels_;
  int height_;
  int width_;
  int out_img_height_;
  int out_img_width_;

  Blob<Dtype> tmp_transform_param_;
};

}  // namespace caffe

#endif  // CAFFE_AFFINE_TRANSFORM_LAYER_HPP_
