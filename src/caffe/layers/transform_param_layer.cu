#include <vector>

#include "caffe/layers/transform_param_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

template <typename Dtype>
__device__ Dtype dot_kernel(const int n, const Dtype *x, const Dtype *y) {
	Dtype ret = Dtype(0);
	for (size_t i = 0; i < n; i++)
	{
		ret += x[i] * y[i];
	}
	return ret;
}

template <typename Dtype>
__device__ Dtype l2_norm_kernel(const int n, const Dtype *x) {
	Dtype ret = dot_kernel<Dtype>(n, x, x);
	return sqrtf(ret);
}

template <typename Dtype>
__device__ void bestFit_kernel(const int num_landmark, const Dtype* dest, const Dtype* src, Dtype *transformed_src, Dtype *T,
	Dtype *srcVec, Dtype *dstVec) {
	Dtype dstMean_x = 0, dstMean_y = 0, srcMean_x = 0, srcMean_y = 0;
	for (size_t i = 0; i < num_landmark; i++)
	{
		dstMean_x += dest[2 * i + 0];
		dstMean_y += dest[2 * i + 1];
		srcMean_x += src[2 * i + 0];
		srcMean_y += src[2 * i + 1];
	}
	dstMean_x /= static_cast<Dtype>(num_landmark);
	dstMean_y /= static_cast<Dtype>(num_landmark);
	srcMean_x /= static_cast<Dtype>(num_landmark);
	srcMean_y /= static_cast<Dtype>(num_landmark);

	for (size_t i = 0; i < num_landmark; i++)
	{
		srcVec[2 * i + 0] = src[2 * i + 0] - srcMean_x;
		srcVec[2 * i + 1] = src[2 * i + 1] - srcMean_y;
		dstVec[2 * i + 0] = dest[2 * i + 0] - dstMean_x;
		dstVec[2 * i + 1] = dest[2 * i + 1] - dstMean_y;
	}
	Dtype srcVecL2Norm = l2_norm_kernel<Dtype>(2 * num_landmark, srcVec);
	Dtype a = dot_kernel<Dtype>(2 * num_landmark, srcVec, dstVec) / (srcVecL2Norm*srcVecL2Norm);
	Dtype b = Dtype(0);
	for (size_t i = 0; i < num_landmark; i++)
	{
		b += srcVec[2 * i] * dstVec[2 * i + 1] - srcVec[2 * i + 1] * dstVec[2 * i];
	}
	b /= srcVecL2Norm*srcVecL2Norm;
	T[0] = a;
	T[1] = b;
	T[2] = -b;
	T[3] = a;
	T[4] = dstMean_x - (srcMean_x*T[0] + srcMean_y*T[2]);
	T[5] = dstMean_y - (srcMean_x*T[1] + srcMean_y*T[3]);
	if (transformed_src != NULL) {
		for (size_t i = 0; i < num_landmark; i++) {
			transformed_src[2 * i + 0] = dstMean_x;
			transformed_src[2 * i + 1] = dstMean_y;
		}
		for (size_t i = 0; i < num_landmark; i++) {
			transformed_src[2 * i + 0] += srcVec[2 * i + 0] * T[0] + srcVec[2 * i + 1] * T[2];
			transformed_src[2 * i + 1] += srcVec[2 * i + 0] * T[1] + srcVec[2 * i + 1] * T[3];
		}
	}
}
template <typename Dtype>
__global__ void transform_param_forward_kernel(const int nThreads, const int num_landmark, 
	const Dtype* src, const Dtype *dst, Dtype* transform_param, Dtype* tmp_srcVec, Dtype* tmp_destVec) {
	CUDA_KERNEL_LOOP(index, nThreads) {
		bestFit_kernel<Dtype>(num_landmark, dst, src + index * num_landmark * 2, NULL, transform_param + index * 6, 
			tmp_srcVec + +index * num_landmark * 2, tmp_destVec + +index * num_landmark * 2);
	}
}

template <typename Dtype>
void TransformParamLayer<Dtype>::Forward_gpu(
	const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
	const int num = bottom[0]->num();
	const int dim = bottom[0]->count(1);
	const Dtype* bottom_data = bottom[0]->gpu_data();
	const Dtype* mean_shape_data = bottom[1]->gpu_data();
	Dtype* top_data = top[0]->mutable_gpu_data();
	transform_param_forward_kernel<Dtype> <<<CAFFE_GET_BLOCKS(num), CAFFE_CUDA_NUM_THREADS>>>(
		num, num_landmark_, bottom_data, mean_shape_data, top_data, tmp_landmark_.mutable_gpu_data(), tmp_landmark_.mutable_gpu_diff());
}

template <typename Dtype>
void TransformParamLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
	if (propagate_down[0]) {
		// nothing to do
	}
}

INSTANTIATE_LAYER_GPU_FUNCS(TransformParamLayer);


}  // namespace caffe
