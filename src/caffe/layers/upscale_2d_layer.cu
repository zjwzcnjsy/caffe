#include <vector>

#include "caffe/layers/upscale_2d_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

template <typename Dtype>
__global__ void upscale_2d_forward_kernel(const int nthreads, const int num, const int channels,
	const int height, const int width, const Dtype *bottom_data, Dtype* top_data) {
	CUDA_KERNEL_LOOP(index, nthreads) {
		const int w = index % width;
		const int h = (index / width) % height;
		const int c = (index / width / height) % channels;
		const int n = index / width / height / channels;
		top_data[index] = bottom_data[((n*channels + c)*height / 2 + h / 2)*width / 2 + w / 2];
	}
}

template <typename Dtype>
__global__ void upscale_2d_backward_kernel(const int nthreads, const int num, const int channels,
	const int height, const int width, Dtype *bottom_data, const Dtype* top_data) {
	CUDA_KERNEL_LOOP(index, nthreads) {
		const int w = index % width;
		const int h = (index / width) % height;
		const int c = (index / width / height) % channels;
		const int n = index / width / height / channels;
		bottom_data[((n*channels + c)*height / 2 + h / 2)*width / 2 + w / 2] += top_data[index];
	}
}

template <typename Dtype>
void Upscale2DLayer<Dtype>::Forward_gpu(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
	upscale_2d_forward_kernel<Dtype> << <CAFFE_GET_BLOCKS(top[0]->count()), CAFFE_CUDA_NUM_THREADS >> >(
		top[0]->count(), top[0]->num(), top[0]->channels(), top[0]->height(), top[0]->width(), bottom[0]->gpu_data(), top[0]->mutable_gpu_data());
}

template <typename Dtype>
void Upscale2DLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
	if (propagate_down[0]) {
		upscale_2d_backward_kernel<Dtype> << <CAFFE_GET_BLOCKS(top[0]->count()), CAFFE_CUDA_NUM_THREADS >> >(
			top[0]->count(), top[0]->num(), top[0]->channels(), top[0]->height(), top[0]->width(), bottom[0]->mutable_gpu_diff(), top[0]->gpu_diff());
	}
}

INSTANTIATE_LAYER_GPU_FUNCS(Upscale2DLayer);


}  // namespace caffe
