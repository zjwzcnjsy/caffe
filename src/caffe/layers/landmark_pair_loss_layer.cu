#include <vector>

#include "caffe/layers/landmark_pair_loss_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

template <typename Dtype>
__global__ void calc_eye_dist_68_kernel(const int N, const int num_landmark, const Dtype *gtLandmarks, Dtype *dist) {
	CUDA_KERNEL_LOOP(index, N) {
		const Dtype *gtLandmark = gtLandmarks + index * num_landmark * 2;
		Dtype left_eye_x = 0, left_eye_y = 0;
		Dtype right_eye_x = 0, right_eye_y = 0;
		for (int i = 36; i < 42; ++i) {
			left_eye_x += gtLandmark[2 * i + 0];
			left_eye_y += gtLandmark[2 * i + 1];
		}
		left_eye_x /= 6;
		left_eye_y /= 6;
		for (int i = 42; i < 48; ++i) {
			right_eye_x += gtLandmark[2 * i + 0];
			right_eye_y += gtLandmark[2 * i + 1];
		}
		right_eye_x /= 6;
		right_eye_y /= 6;
		Dtype dx = left_eye_x - right_eye_x;
		Dtype dy = left_eye_y - right_eye_y;
		dist[index] = sqrt(dx*dx + dy*dy);
	}
}

template <typename Dtype>
__global__ void calc_landmark_dist_kernel(const int N, const int num_landmark, const Dtype *diff, Dtype *dist) {
	CUDA_KERNEL_LOOP(index, N) {
		int iBatch = index / num_landmark;
		int iLandmark = index % num_landmark;
		const Dtype *diff2 = diff + iBatch * num_landmark * 2 + iLandmark * 2;
		
		dist[index] = sqrt(diff2[0] + diff2[1]);
	}
}

template <typename Dtype>
__global__ void calc_single_landmark_error_kernel(const int N, const int num_landmark, const Dtype *diff, Dtype *dist) {
	CUDA_KERNEL_LOOP(index, N) {
		int iBatch = index / num_landmark;
		int iLandmark = index % num_landmark;
		Dtype *diff2 = diff + iBatch * num_landmark * 2 + iLandmark * 2;

		dist[index] = sqrt(diff2[0] + diff2[1]);
	}
}

template <typename Dtype>
void LandmarkPairLossLayer<Dtype>::Forward_gpu(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
	const int num = bottom[0]->num();
	const int landmark_dim = 2 * num_landmark_;
	
	// 求预测landmark与真实landmark的的差和差的平方
	caffe_gpu_sub(bottom[0]->count(), bottom[0]->gpu_data(), bottom[1]->gpu_data(), tmp_diff.mutable_gpu_data());
	caffe_gpu_mul(tmp_diff.count(), tmp_diff.gpu_data(), tmp_diff.gpu_data(), tmp_diff.mutable_gpu_diff());
	// 求预测landmark与真实landmark的距离
	calc_landmark_dist_kernel<Dtype> << <CAFFE_GET_BLOCKS(num*num_landmark_), CAFFE_CUDA_NUM_THREADS >> >(
		num*num_landmark_, num_landmark_, tmp_diff.gpu_diff(), tmp_dist.mutable_gpu_data());
	
	// 求真实landmark的眼睛间的距离
	calc_eye_dist_68_kernel<Dtype> <<<CAFFE_GET_BLOCKS(num), CAFFE_CUDA_NUM_THREADS >>>(
		num, num_landmark_, bottom[1]->gpu_data(), tmp_eye_dist.mutable_gpu_data());

	const Dtype *tmp_dist_data = tmp_dist.gpu_data();
	const Dtype *tmp_eye_dist_data = tmp_eye_dist.cpu_data();
	Dtype error = Dtype(0.0);
	for (int i = 0; i < num; ++i) {
		Dtype sum_dist = Dtype(0.0);
		caffe_gpu_asum(num_landmark_, tmp_dist_data + i*num_landmark_, &sum_dist);
		sum_dist /= static_cast<Dtype>(num_landmark_);
		error += sum_dist / tmp_eye_dist_data[i];
	}
	top[0]->mutable_cpu_data()[0] = error / static_cast<Dtype>(num);
}

template <typename Dtype>
void LandmarkPairLossLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
	if (propagate_down[0]) {
		const int num = bottom[0]->num();
		const int landmark_dim = 2 * num_landmark_;
		const Dtype *tmp_diff_data = tmp_diff.gpu_data();
		const Dtype *tmp_dist_data = tmp_dist.gpu_data();
		const Dtype *tmp_eye_dist_data = tmp_eye_dist.cpu_data();
		const Dtype *top_diff = top[0]->cpu_diff();
		Dtype *bottom_diff = bottom[0]->mutable_gpu_diff();
		for (int i = 0; i < bottom[0]->num(); ++i) {
			Dtype alpha = top_diff[0] / (num*tmp_eye_dist_data[i] * num_landmark_);
			caffe_gpu_div(num_landmark_, tmp_diff_data, tmp_dist_data, bottom_diff);
			caffe_gpu_div(num_landmark_, tmp_diff_data + num_landmark_, tmp_dist_data, bottom_diff + num_landmark_);
			caffe_gpu_scal(landmark_dim, alpha, bottom_diff);
			tmp_dist_data += num_landmark_;
			bottom_diff += landmark_dim;
		}
	}
}

INSTANTIATE_LAYER_GPU_FUNCS(LandmarkPairLossLayer);


}  // namespace caffe
