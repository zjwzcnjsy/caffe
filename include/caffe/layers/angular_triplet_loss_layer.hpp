#ifndef CAFFE_ANGULAR_TRIPLET_LOSS_LAYER_HPP_
#define CAFFE_ANGULAR_TRIPLET_LOSS_LAYER_HPP_

#include <vector>

#include "caffe/blob.hpp"
#include "caffe/layer.hpp"
#include "caffe/proto/caffe.pb.h"

namespace caffe {

/**
 * @brief https://github.com/KaleidoZhouYN/Angular-Triplet-Loss
 *
 * TODO(dox): thorough documentation for Forward, Backward, and proto params.
 */
template <typename Dtype>
class AngularTripletLossLayer : public Layer<Dtype> {
 public:
  explicit AngularTripletLossLayer(const LayerParameter& param)
      : Layer<Dtype>(param) {}
  virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);

  virtual inline const char* type() const { return "AngularTripletLoss"; }
  virtual inline int ExactNumBottomBlobs() const { return 2; }
  virtual inline int MaxTopBlobs() const { return 2; }

 protected:
  virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Forward_gpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Backward_cpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);
  virtual void Backward_gpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);

  int M_;
  int K_;
  int N_;
  
 AngularTripletLossParameter_MarginType type_;

  // common variables
  Blob<Dtype> x_norm_;
  Blob<Dtype> cos_theta_;
  Blob<Dtype> sign_0_; // sign_0 = sign(cos_theta)
  // for DOUBLE type
  Blob<Dtype> cos_theta_quadratic_;
  // for TRIPLE type
  Blob<Dtype> sign_1_; // sign_1 = sign(abs(cos_theta) - 0.5)
  Blob<Dtype> sign_2_; // sign_2 = sign_0 * (1 + sign_1) - 2
  Blob<Dtype> cos_theta_cubic_;
  // for QUADRA type
  Blob<Dtype> sign_3_; // sign_3 = sign_0 * sign(2 * cos_theta_quadratic_ - 1)
  Blob<Dtype> sign_4_; // sign_4 = 2 * sign_0 + sign_3 - 3
  Blob<Dtype> cos_theta_quartic_;
  
  Blob<Dtype> margin_top_data_; 

  int iter_;
  Dtype lambda_;
  //bool triplet_flag; 

};

}  // namespace caffe

#endif  // CAFFE_ANGULAR_TRIPLET_LOSS_LAYER_HPP_
