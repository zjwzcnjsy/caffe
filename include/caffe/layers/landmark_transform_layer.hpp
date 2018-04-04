#ifndef CAFFE_LANDMARK_TRANSFORM_LAYER_HPP_
#define CAFFE_LANDMARK_TRANSFORM_LAYER_HPP_

#include <vector>

#include "caffe/blob.hpp"
#include "caffe/layer.hpp"
#include "caffe/proto/caffe.pb.h"

namespace caffe {

/**
 * @brief DAN�����и��ݱ任�����Ե�ǰ��״���б任��
 *	����ṹ����ʱ����ָ��inverse��������ʾ�Ƿ������任��Ĭ��ΪFalse
 *
 * @param bottom input Blob vector (length 1)
 *   -# @f$ (N \times num_landmark*2) @f$
 *      ��ǰ��״
 * @param top output Blob vector (length 1)
 *   -# @f$ (N \times 6) @f$
 *      �任����
 */
template <typename Dtype>
class LandmarkTransformLayer : public Layer<Dtype> {
 public:
  explicit LandmarkTransformLayer(const LayerParameter& param)
      : Layer<Dtype>(param) {}
  virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Reshape(const vector<Blob<Dtype>*>& bottom,
	  const vector<Blob<Dtype>*>& top);

  virtual inline const char* type() const { return "LandmarkTransform"; }
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
  bool inverse_;
  Blob<Dtype> transform_param_;
};

}  // namespace caffe

#endif  // CAFFE_LANDMARK_TRANSFORM_LAYER_HPP_
