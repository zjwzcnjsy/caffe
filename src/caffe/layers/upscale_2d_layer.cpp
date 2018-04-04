#include <vector>

#include "caffe/layers/upscale_2d_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

template <typename Dtype>
void Upscale2DLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
}

template <typename Dtype>
void Upscale2DLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
	const vector<Blob<Dtype>*>& top) {
	vector<int> top_shape = bottom[0]->shape();
	top_shape[2] *= 2;
	top_shape[3] *= 2;
	top[0]->Reshape(top_shape);
}

template <typename Dtype>
void Upscale2DLayer<Dtype>::Forward_cpu(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  const int num = top[0]->num();
  const int channels = top[0]->channels();
  const int height = top[0]->height();
  const int width = top[0]->width();
  const Dtype* bottom_data = bottom[0]->cpu_data();
  Dtype* top_data = top[0]->mutable_cpu_data();
  for (int n = 0; n < num; ++n) {
	  for (int c = 0; c < channels; ++c) {
		  for (int h = 0; h < height; ++h) {
			  for (int w = 0; w < width; ++w) {
				  *top_data++ = bottom_data[bottom[0]->offset(n, c, h / 2, w / 2)];
			  }
		  }
	  }
  }
}

template <typename Dtype>
void Upscale2DLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
  if (propagate_down[0]) {
	  const int num = bottom[0]->num();
	  const int channels = bottom[0]->channels();
	  const int height = bottom[0]->height();
	  const int width = bottom[0]->width();
	  Dtype* bottom_diff = bottom[0]->mutable_cpu_diff();
	  const Dtype* top_diff = top[0]->cpu_diff();
	  for (int i = 0; i < num; ++i) {
		  for (int j = 0; j < channels; ++j) {
			  for (int h = 0; h < 2 * height; ++h) {
				  for (int w = 0; w < 2 * width; ++w) {
					  bottom_diff[bottom[0]->offset(i, j, h / 2, w / 2)] += *top_diff++;
				  }
			  }
		  }
	  }
  }
}

#ifdef CPU_ONLY
STUB_GPU(Upscale2DLayer);
#endif

INSTANTIATE_CLASS(Upscale2DLayer);
REGISTER_LAYER_CLASS(Upscale2D);

}  // namespace caffe
