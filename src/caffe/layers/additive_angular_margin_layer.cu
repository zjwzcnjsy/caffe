#include <vector>
#include <cfloat>

#include "caffe/blob.hpp"
#include "caffe/common.hpp"
#include "caffe/filler.hpp"
#include "caffe/layer.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/layers/additive_angular_margin_layer.hpp"

namespace caffe {
	template <typename Dtype>
	__global__ void AdditiveAngularMarginForward(const int n, const int dim, const Dtype* label,
		Dtype* top_data, Dtype margin, Dtype cos_margin, Dtype sin_margin) {
		CUDA_KERNEL_LOOP(index, n) {
			int gt = static_cast<int>(label[index]);
			float cos_theta = top_data[index * dim + gt];
			top_data[index * dim + gt] = cos_theta * cos_margin - sqrtf(1 - cos_theta*cos_theta)*sin_margin;
		}
	}

	template <typename Dtype>
	__global__ void AdditiveAngularMarginBackward(const int n, const int dim, const Dtype* label,
		const Dtype* top_data, const Dtype* top_diff, Dtype* bottom_diff,
		Dtype margin, Dtype cos_margin, Dtype sin_margin) {
		CUDA_KERNEL_LOOP(index, n) {
			int gt = static_cast<int>(label[index]);
			float cos_theta = top_data[index * dim + gt];
			float sin_theta = sqrtf(1 - cos_theta*cos_theta);
			bottom_diff[index * dim + gt] *= cos_margin + cos_theta*sin_margin / sin_theta;
		}
	}

	template <typename Dtype>
	void AdditiveAngularMarginLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
		const vector<Blob<Dtype>*>& top) {
		const Dtype* bottom_data = bottom[0]->gpu_data();
		const Dtype* label_data = bottom[1]->gpu_data();
		Dtype* top_data = top[0]->mutable_gpu_data();

		int num = bottom[0]->num();
		int count = bottom[0]->count();
		int dim = count / num;

		if (top[0] != bottom[0]) caffe_copy(count, bottom_data, top_data);

		// NOLINT_NEXT_LINE(whitespace/operators)
		AdditiveAngularMarginForward<Dtype> << <CAFFE_GET_BLOCKS(num), CAFFE_CUDA_NUM_THREADS >> > (
			num, dim, label_data, top_data, margin_, cos_margin_, sin_margin_);
		CUDA_POST_KERNEL_CHECK;
	}

	template <typename Dtype>
	void AdditiveAngularMarginLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
		const vector<bool>& propagate_down,
		const vector<Blob<Dtype>*>& bottom) {
		if (top[0] != bottom[0] && propagate_down[0]) {
			const Dtype* top_diff = top[0]->gpu_diff();
			Dtype* bottom_diff = bottom[0]->mutable_gpu_diff();
			const Dtype* top_data = top[0]->gpu_data();
			const Dtype* label_data = bottom[1]->gpu_data();

			int num = bottom[0]->num();
			int count = bottom[0]->count();
			int dim = count / num;
			caffe_copy(count, top_diff, bottom_diff);

			// NOLINT_NEXT_LINE(whitespace/operators)
			AdditiveAngularMarginBackward<Dtype> << <CAFFE_GET_BLOCKS(num), CAFFE_CUDA_NUM_THREADS >> > (
				num, dim, label_data, top_data, top_diff, bottom_diff, margin_, cos_margin_, sin_margin_);
			CUDA_POST_KERNEL_CHECK;
		}
	}

	INSTANTIATE_LAYER_GPU_FUNCS(AdditiveAngularMarginLayer);

}  // namespace caffe
