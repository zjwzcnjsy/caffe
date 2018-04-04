#include <vector>

#include "caffe/layers/landmark_init_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

template <typename Dtype>
void LandmarkInitLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  CHECK_EQ(bottom[0]->channels(), bottom[1]->channels()) << "the number of landmark miss match.";
}

template <typename Dtype>
void LandmarkInitLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
	const vector<Blob<Dtype>*>& top) {
	top[0]->ReshapeLike(*bottom[0]);
}

template <typename Dtype>
void LandmarkInitLayer<Dtype>::Forward_cpu(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  const int num = bottom[0]->num();
  const int dim = bottom[0]->count(1);
  const Dtype* bottom_data = bottom[0]->cpu_data();
  const Dtype* initShape_data = bottom[1]->cpu_data();
  Dtype* top_data = top[0]->mutable_cpu_data();
  for (int i = 0; i < num; ++i) {
	  caffe_add(dim, bottom_data, initShape_data, top_data);
	  bottom_data += dim;
	  top_data += dim;
  }
}

template <typename Dtype>
void LandmarkInitLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
  if (propagate_down[0]) {
	  caffe_copy(bottom[0]->count(), top[0]->cpu_diff(), bottom[0]->mutable_cpu_diff());
  }
}

#ifdef CPU_ONLY
STUB_GPU(LandmarkInitLayer);
#endif

INSTANTIATE_CLASS(LandmarkInitLayer);
REGISTER_LAYER_CLASS(LandmarkInit);

}  // namespace caffe
