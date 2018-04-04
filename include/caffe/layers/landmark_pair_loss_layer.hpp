#ifndef CAFFE_LANDMARK_PAIR_LOSS_LAYER_HPP_
#define CAFFE_LANDMARK_PAIR_LOSS_LAYER_HPP_

#include <vector>

#include "caffe/blob.hpp"
#include "caffe/layer.hpp"
#include "caffe/proto/caffe.pb.h"
#include "caffe/layers/loss_layer.hpp"

namespace caffe {

/**
 * @brief DAN�����йؼ���ĵ�����Ĵ��۲㡣
 *
 * @param bottom input Blob vector (length 1)
 *   -# @f$ (N \times num_landmark*2) @f$
 *      Ԥ����״
 * @param bottom input Blob vector (length 1)
 *   -# @f$ (N \times num_landmark*2) @f$
 *      ��ǰ��״
 * @param top output Blob vector (length 1)
 *   -# @f$ (1) @f$
 *      ����ֵ
 */
template <typename Dtype>
class LandmarkPairLossLayer : public LossLayer<Dtype> {
 public:
  explicit LandmarkPairLossLayer(const LayerParameter& param)
      : LossLayer<Dtype>(param) {}
  virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Reshape(const vector<Blob<Dtype>*>& bottom,
	  const vector<Blob<Dtype>*>& top);

  virtual inline const char* type() const { return "LandmarkPairLoss"; }
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
  Blob<Dtype> tmp_diff;
  Blob<Dtype> tmp_dist;
  Blob<Dtype> tmp_eye_dist;
};

}  // namespace caffe

#endif  // CAFFE_LANDMARK_PAIR_LOSS_LAYER_HPP_
