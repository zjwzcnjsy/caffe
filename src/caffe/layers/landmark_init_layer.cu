#include <vector>

#include "caffe/layers/landmark_init_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

template <typename Dtype>
void LandmarkInitLayer<Dtype>::Forward_gpu(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
	const int num = bottom[0]->num();
	const int dim = bottom[0]->count(1);
	const Dtype* bottom_data = bottom[0]->gpu_data();
	const Dtype* initShape_data = bottom[1]->gpu_data();
	Dtype* top_data = top[0]->mutable_gpu_data();
	for (int i = 0; i < num; ++i) {
		caffe_gpu_add(dim, bottom_data, initShape_data, top_data);
		bottom_data += dim;
		top_data += dim;
	}
}

template <typename Dtype>
void LandmarkInitLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
	if (propagate_down[0]) {
		caffe_gpu_memcpy(bottom[0]->count(), top[0]->gpu_diff(), bottom[0]->mutable_gpu_diff());
	}
}

INSTANTIATE_LAYER_GPU_FUNCS(LandmarkInitLayer);


}  // namespace caffe
