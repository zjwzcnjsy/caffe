#include <vector>

#include "caffe/layers/landmark_transform_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

template <typename Dtype>
__global__ void transform_param_inv_kernel(const int n, const Dtype* x, Dtype* y) {
	CUDA_KERNEL_LOOP(index, n) {
		const Dtype *A = x + index * 6;
		const Dtype *t = A + 4;
		Dtype *A2 = y + index * 6;
		Dtype *t2 = A2 + 4;
		double det = A[0] * A[3] - A[1] * A[2];
		if (det != 0.) {
			det = 1. / det;
			double t0, t1;
			t0 = A[0] * det;
			t1 = A[3] * det;
			A2[3] = static_cast<Dtype>(t0);
			A2[0] = static_cast<Dtype>(t1);
			t0 = -A[1] * det;
			t1 = -A[2] * det;
			A2[1] = static_cast<Dtype>(t0);
			A2[2] = static_cast<Dtype>(t1);
		}
		Dtype t0 = t[0];
		Dtype t1 = t[1];
		t2[0] = -(t0 * A2[0] + t1 * A2[2]);
		t2[1] = -(t0 * A2[1] + t1 * A2[3]);
	}
}

template <typename Dtype>
__global__ void top_data_add_t_kernel(const int n, const int num_landmark, const Dtype* transform_param, Dtype* top_data) {
	CUDA_KERNEL_LOOP(index, n) {
		const Dtype *A = transform_param + index * 6;
		const Dtype *t = A + 4;
		for (int j = 0; j < num_landmark; ++j) {
			top_data[index*num_landmark * 2 + 2 * j + 0] = t[0];
			top_data[index*num_landmark * 2 + 2 * j + 1] = t[1];
		}
	}
}

template <typename Dtype>
void LandmarkTransformLayer<Dtype>::Forward_gpu(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
	const int num = bottom[0]->num();
	const int dim = bottom[0]->count(1);
	const Dtype* bottom_data = bottom[0]->gpu_data();
	const Dtype* transform_param_data = bottom[1]->gpu_data();
	Dtype* top_data = top[0]->mutable_gpu_data();
	if (inverse_) {
		transform_param_inv_kernel<Dtype> << <CAFFE_GET_BLOCKS(bottom[1]->num()), CAFFE_CUDA_NUM_THREADS >> >(
			bottom[1]->num(), bottom[1]->gpu_data(), transform_param_.mutable_gpu_data());
	}
	else {
		caffe_copy(bottom[1]->count(), bottom[1]->gpu_data(), transform_param_.mutable_gpu_data());
	}
	const Dtype* tmp_transform_param_data = transform_param_.gpu_data();
	top_data_add_t_kernel<Dtype> << <CAFFE_GET_BLOCKS(bottom[1]->num()), CAFFE_CUDA_NUM_THREADS >> >(
		bottom[1]->num(), num_landmark_, transform_param_.gpu_data(), top_data);
	for (int i = 0; i < num; ++i) {
		caffe_gpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans,
			num_landmark_, 2, 2, (Dtype)1.,
			bottom_data, tmp_transform_param_data, (Dtype)1., top_data);
		bottom_data += dim;
		top_data += dim;
		tmp_transform_param_data += 6;
	}
}

template <typename Dtype>
void LandmarkTransformLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
	if (propagate_down[0]) {
		const int num = bottom[0]->num();
		const int dim = bottom[0]->count(1);
		Dtype* bottom_diff = bottom[0]->mutable_gpu_diff();
		const Dtype* tmp_transform_param_data = transform_param_.gpu_data();
		const Dtype* top_diff = top[0]->gpu_diff();
		for (int i = 0; i < num; ++i) {
			caffe_gpu_gemm<Dtype>(CblasNoTrans, CblasTrans,
				num_landmark_, 2, 2, (Dtype)1.,
				top_diff, tmp_transform_param_data, (Dtype)0., bottom_diff);
			bottom_diff += dim;
			top_diff += dim;
			tmp_transform_param_data += 6;
		}
	}
}

INSTANTIATE_LAYER_GPU_FUNCS(LandmarkTransformLayer);


}  // namespace caffe
