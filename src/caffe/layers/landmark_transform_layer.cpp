#include <vector>

#include "caffe/layers/landmark_transform_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

template <typename Dtype>
void LandmarkTransformLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  CHECK_EQ(bottom[0]->num(), bottom[1]->num()) << "the number of shape miss match.";
  CHECK_EQ(bottom[1]->channels(), 6) << "must has 6 transform param";
  num_landmark_ = bottom[0]->channels() / 2;
  const LandmarkTransformParameter& landmark_transform_param = this->layer_param_.landmark_transform_param();
  inverse_ = landmark_transform_param.inverse();
  transform_param_.ReshapeLike(*bottom[1]);
}

template <typename Dtype>
void LandmarkTransformLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
	const vector<Blob<Dtype>*>& top) {
	top[0]->ReshapeLike(*bottom[0]);
}

template <typename Dtype>
void LandmarkTransformLayer<Dtype>::Forward_cpu(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  const int num = bottom[0]->num();
  const int dim = bottom[0]->count(1);
  const Dtype* bottom_data = bottom[0]->cpu_data();
  const Dtype* transform_param_data = bottom[1]->cpu_data();
  Dtype* tmp_transform_param_data = transform_param_.mutable_cpu_data();
  Dtype* top_data = top[0]->mutable_cpu_data();
  for (int i = 0; i < num; ++i) {
	  caffe_copy(6, transform_param_data, tmp_transform_param_data);
	  if (inverse_)
	  {
		  caffe_2p2_matrix_inv(tmp_transform_param_data, tmp_transform_param_data);
		  Dtype *A = tmp_transform_param_data;
		  Dtype *t = tmp_transform_param_data + 4;
		  Dtype t0 = t[0];
		  Dtype t1 = t[1];
		  t[0] = -(t0 * A[0] + t1 * A[2]);
		  t[1] = -(t0 * A[1] + t1 * A[3]);
	  }
	  Dtype *t = tmp_transform_param_data + 4;
	  for (int j = 0; j < num_landmark_; ++j) {
		  top_data[2 * j + 0] = t[0];
		  top_data[2 * j + 1] = t[1];
	  }
	  caffe_cpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans,
		  num_landmark_, 2, 2, (Dtype)1.,
		  bottom_data, tmp_transform_param_data, (Dtype)1., top_data);
	  bottom_data += dim;
	  top_data += dim;
	  transform_param_data += 6;
	  tmp_transform_param_data += 6;
  }
}

template <typename Dtype>
void LandmarkTransformLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
  if (propagate_down[0]) {
	  const int num = bottom[0]->num();
	  const int dim = bottom[0]->count(1);
	  Dtype* bottom_diff = bottom[0]->mutable_cpu_diff();
	  const Dtype* tmp_transform_param_data = transform_param_.cpu_data();
	  const Dtype* top_diff = top[0]->cpu_diff();
	  for (int i = 0; i < num; ++i) {
		  caffe_cpu_gemm<Dtype>(CblasNoTrans, CblasTrans,
			  num_landmark_, 2, 2, (Dtype)1.,
			  top_diff, tmp_transform_param_data, (Dtype)0., bottom_diff);
		  bottom_diff += dim;
		  top_diff += dim;
		  tmp_transform_param_data += 6;
	  }
  }
}

#ifdef CPU_ONLY
STUB_GPU(LandmarkTransformLayer);
#endif

INSTANTIATE_CLASS(LandmarkTransformLayer);
REGISTER_LAYER_CLASS(LandmarkTransform);

}  // namespace caffe
