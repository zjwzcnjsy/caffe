#include <vector>

#include "caffe/layers/binary_conv_layer2.hpp"
#include "cuda_profiler_api.h"
#include "caffe/util/benchmark.hpp"
#include "caffe/util/binary_kernels.hpp"

#include <iostream>
using namespace std;

namespace caffe {

	template <typename Dtype>
	void BinaryConvolutionV2Layer<Dtype>::forward_gpu_gemm(const Dtype* input,
		const Dtype* weights, Dtype* output, bool skip_im2col) {
		const Dtype* col_buff = input;
		if (!is_1x1_) {
			if (!skip_im2col) {
				conv_im2col_gpu(input, col_buffer_.mutable_gpu_data());
			}
			col_buff = col_buffer_.gpu_data();
		}
		for (int g = 0; g < group_; ++g) {
			caffe_gpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, conv_out_channels_ /
				group_, conv_out_spatial_dim_, kernel_dim_,
				(Dtype)1., weights + weight_offset_ * g, col_buff + col_offset_ * g,
				(Dtype)0., output + output_offset_ * g);
		}
	}

	template <typename Dtype>
	void BinaryConvolutionV2Layer<Dtype>::forward_gpu_gemm_xnor(const Dtype* input,
		const Dtype* weights, const Dtype* A, Dtype* output, bool skip_im2col) {
		const Dtype* col_buff = input;
		if (!is_1x1_) {
			if (!skip_im2col) {
				conv_im2col_gpu(input, col_buffer_.mutable_gpu_data());
			}
			col_buff = col_buffer_.gpu_data();
		}
		const int dim = conv_out_channels_ / group_ * conv_out_spatial_dim_;
		for (int g = 0; g < group_; ++g) {
			xnor_gemm(weights, A, col_buff, output, uiA_.mutable_gpu_data(), uiB_.mutable_gpu_data(),
				conv_out_channels_ / group_, kernel_dim_, conv_out_spatial_dim_);
		}
	}

	template <typename Dtype>
	void BinaryConvolutionV2Layer<Dtype>::forward_gpu_bias(Dtype* output,
		const Dtype* bias) {
		caffe_gpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, num_output_,
			out_spatial_dim_, 1, (Dtype)1., bias, bias_multiplier_.gpu_data(),
			(Dtype)1., output);
	}

	template <typename Dtype>
	void BinaryConvolutionV2Layer<Dtype>::backward_gpu_gemm(const Dtype* output,
		const Dtype* weights, Dtype* input) {
		Dtype* col_buff = col_buffer_.mutable_gpu_data();
		if (is_1x1_) {
			col_buff = input;
		}
		for (int g = 0; g < group_; ++g) {
			caffe_gpu_gemm<Dtype>(CblasTrans, CblasNoTrans, kernel_dim_,
				conv_out_spatial_dim_, conv_out_channels_ / group_,
				(Dtype)1., weights + weight_offset_ * g, output + output_offset_ * g,
				(Dtype)0., col_buff + col_offset_ * g);
		}
		if (!is_1x1_) {
			conv_col2im_gpu(col_buff, input);
		}
	}

	template <typename Dtype>
	void BinaryConvolutionV2Layer<Dtype>::weight_gpu_gemm(const Dtype* input,
		const Dtype* output, Dtype* weights) {
		const Dtype* col_buff = input;
		if (!is_1x1_) {
			conv_im2col_gpu(input, col_buffer_.mutable_gpu_data());
			col_buff = col_buffer_.gpu_data();
		}
		for (int g = 0; g < group_; ++g) {
			caffe_gpu_gemm<Dtype>(CblasNoTrans, CblasTrans, conv_out_channels_ / group_,
				kernel_dim_, conv_out_spatial_dim_,
				(Dtype)1., output + output_offset_ * g, col_buff + col_offset_ * g,
				(Dtype)1., weights + weight_offset_ * g);
		}
	}

	template <typename Dtype>
	void BinaryConvolutionV2Layer<Dtype>::backward_gpu_bias(Dtype* bias,
		const Dtype* input) {
		caffe_gpu_gemv<Dtype>(CblasNoTrans, num_output_, out_spatial_dim_, 1.,
			input, bias_multiplier_.gpu_data(), 1., bias);
	}

	template <typename Dtype>
	__global__ void binarize(const int nThreads, const int kernel_dim,
		const Dtype* w, const Dtype* A, Dtype* w_bin, Dtype* w_sign) {
		CUDA_KERNEL_LOOP(index, nThreads) {
			int num = index / kernel_dim;
			Dtype v = sign(w[index]);
			w_sign[index] = v;
			w_bin[index] = v * A[num];
		}
	}

	template <typename Dtype>
	__global__ void permute_channel(const int nThreads,
		const int num, const int channels, const int height, const int width,
		const Dtype* weights, Dtype* buffer) {
		CUDA_KERNEL_LOOP(index, nThreads) {
			const int w = index % width;
			const int h = (index / width) % height;
			const int c = (index / width / height) % channels;
			const int n = index / width / height / channels;
			buffer[((n*height + h)*width + w)*channels + c] = weights[index];
		}
	}

	template <typename Dtype>
	__global__ void calc_meancenter(const int nThreads,
		const int num, const int channels, const int height, const int width,
		const Dtype* weights, Dtype* center) {
		CUDA_KERNEL_LOOP(index, nThreads) {
			const int w = index % width;
			const int h = (index / width) % height;
			const int n = index / width / height;
			Dtype sum = Dtype(0.0);
			for (int c = 0; c < channels; ++c) {
				sum += weights[((n*channels + c)*height + h)*width + w];
			}
			center[(n*height + h)*width + w] = sum / static_cast<Dtype>(channels);
		}
	}

	template <typename Dtype>
	__global__ void meancenter_remove_and_clamp(const int nThreads,
		const int num, const int channels, const int height, const int width,
		const Dtype* center, Dtype* weights, const Dtype minv, const Dtype maxv) {
		CUDA_KERNEL_LOOP(index, nThreads) {
			const int w = index % width;
			const int h = (index / width) % height;
			const int n = index / width / height / channels;
			Dtype v = weights[index] - center[(n*height + h)*width + w];
			if (v < minv) {
				v = minv;
			}
			else if (v > maxv) {
				v = maxv;
			}
			else {
				//nothing to do;
			}
			weights[index] = v;
		}
	}

	template <typename Dtype>
	__global__ void mc_remove_clamp_abs(const int nThreads,
		const int num, const int channels, const int height, const int width,
		const Dtype* center, Dtype* weights, Dtype* abs_w, const Dtype minv, const Dtype maxv) {
		CUDA_KERNEL_LOOP(index, nThreads) {
			const int w = index % width;
			const int h = (index / width) % height;
			const int n = index / width / height / channels;
			Dtype v = weights[index] - center[(n*height + h)*width + w];
			if (v < minv) {
				v = minv;
			}
			else if (v > maxv) {
				v = maxv;
			}
			else {
				//nothing to do;
			}
			weights[index] = v;
			abs_w[index] = std::abs(v);
		}
	}

	template <typename Dtype>
	__global__ void meancenter_remove(const int nThreads,
		const int num, const int channels, const int height, const int width,
		const Dtype* center, Dtype* weights) {
		CUDA_KERNEL_LOOP(index, nThreads) {
			const int w = index % width;
			const int h = (index / width) % height;
			const int n = index / width / height / channels;
			weights[index] -= center[(n*height + h)*width + w];
		}
	}

	template <typename Dtype>
	__global__ void clamp(const int nThreads, Dtype* weights, const Dtype minv, const Dtype maxv) {
		CUDA_KERNEL_LOOP(index, nThreads) {
			Dtype v = weights[index];
			if (v < minv) {
				v = minv;
			}
			else if (v > maxv) {
				v = maxv;
			}
			else {
				//nothing to do;
			}
			weights[index] = v;
		}
	}

	template <typename Dtype>
	void BinaryConvolutionV2Layer<Dtype>::binarizeGPUTo(Blob<Dtype>* weights) {
		CHECK_EQ(weights->count(), binary_w_.count());
		CHECK_EQ(weights->num(), A_.num());
		const int count = weights->count();
		const int num = weights->num();
		const int kernel_dim = weights->count(1);
		// compute A
		//caffe_gpu_abs(count, weights->gpu_data(), w_buffer_.mutable_gpu_diff());
		caffe_gpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, num,
			1, kernel_dim, Dtype(1. / kernel_dim), w_buffer_.gpu_diff(), multiplier_.gpu_data(),
			(Dtype)0., A_.mutable_gpu_data());
		// compute sign(w) and A*sign(w)
		binarize<Dtype> << <CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS >> >(
			count, kernel_dim, weights->gpu_data(), A_.gpu_data(), weights->mutable_gpu_data(), binary_w_.mutable_gpu_diff());
	}

	template <typename Dtype>
	void BinaryConvolutionV2Layer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
		const vector<Blob<Dtype>*>& top) {
		cudaProfilerStart();
		if (this->phase_ == TRAIN) {
			const int count = this->blobs_[0]->count();
			const int num = this->blobs_[0]->num();
			const int channels = this->blobs_[0]->channels();
			const int height = this->blobs_[0]->height();
			const int width = this->blobs_[0]->width();
			// compute mean
			if (height == 1 && width == 1) {
				caffe_gpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, num,
					1, channels, Dtype(1. / channels), this->blobs_[0]->gpu_data(), multiplier_.gpu_data(),
					(Dtype)0., meancenter_.mutable_gpu_data());
			}
			else {
				permute_channel<Dtype> << <CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS >> > (
					count, num, channels, height, width,
					this->blobs_[0]->gpu_data(), w_buffer_.mutable_gpu_data());
				caffe_gpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, num*height*width,
					1, channels, Dtype(1. / channels), w_buffer_.gpu_data(), multiplier_.gpu_data(),
					(Dtype)0., meancenter_.mutable_gpu_data());
				/*calc_meancenter<Dtype> << <CAFFE_GET_BLOCKS(count / channels), CAFFE_CUDA_NUM_THREADS >> >(
					count / channels, num, channels, height, width,
					this->blobs_[0]->gpu_data(), meancenter_.mutable_gpu_data());*/
			}

			// subtract mean and clip weight to [-1,1]
			mc_remove_clamp_abs<Dtype> << <CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS >> >(
				count, num, channels, height, width,
				meancenter_.gpu_data(), this->blobs_[0]->mutable_gpu_data(), w_buffer_.mutable_gpu_diff(), 
				Dtype(-1.0), Dtype(1.0));
			
			// store weight to buffer
			caffe_copy(count, this->blobs_[0]->gpu_data(), w_buffer_.mutable_gpu_data());

			// binary weight, this->blobs_[0]'s data hold A*sign(w), binary_w_'s diff hold sign(w)
			binarizeGPUTo(&(*this->blobs_[0]));
		}
		else {
			const int count = this->blobs_[0]->count();
			// store weight to buffer
			caffe_copy(count, this->blobs_[0]->gpu_data(), w_buffer_.mutable_gpu_data());

			const int num = this->blobs_[0]->num();
			const int channels = this->blobs_[0]->channels();
			const int height = this->blobs_[0]->height();
			const int width = this->blobs_[0]->width();
			// compute mean
			calc_meancenter<Dtype> << <CAFFE_GET_BLOCKS(count / channels), CAFFE_CUDA_NUM_THREADS >> >(
				count / channels, num, channels, height, width,
				this->blobs_[0]->gpu_data(), meancenter_.mutable_gpu_data());
			// subtract mean and clip weight to [-1,1]
			mc_remove_clamp_abs<Dtype> << <CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS >> >(
				count, num, channels, height, width,
				meancenter_.gpu_data(), this->blobs_[0]->mutable_gpu_data(), w_buffer_.mutable_gpu_diff(),
				Dtype(-1.0), Dtype(1.0));
			// binary weight, binary_w_'s data hold A*sign(w), binary_w_'s diff hold sign(w)
			binarizeGPUTo(&(*this->blobs_[0]));
		}

		/*const Dtype* w_sign_cpu = binary_w_.cpu_diff();
		cout << "w_sign:";
		for (int i = 0; i < binary_w_.count(); ++i) {
			cout << ", " << w_sign_cpu[i];
		}
		cout << endl;
		cout << "A_cpu:";
		const Dtype* A_cpu = A_.cpu_data();
		for (int i = 0; i < A_.count(); ++i) {
			cout << ", " << A_cpu[i];
		}
		cout << endl;*/

		const Dtype* weight = this->blobs_[0]->gpu_data();
		const Dtype* w_sign = binary_w_.gpu_diff();
		const Dtype* A = A_.gpu_data();
		for (int i = 0; i < bottom.size(); ++i) {
			const Dtype* bottom_data = bottom[i]->gpu_data();
			Dtype* top_data = top[i]->mutable_gpu_data();
			for (int n = 0; n < this->num_; ++n) {
				this->forward_gpu_gemm_xnor(bottom_data + n * this->bottom_dim_, w_sign, A,
					top_data + n * this->top_dim_);
				if (this->bias_term_) {
					const Dtype* bias = this->blobs_[1]->gpu_data();
					this->forward_gpu_bias(top_data + n * this->top_dim_, bias);
				}
			}
		}
		if (this->phase_ == TEST) {
			const int count = this->blobs_[0]->count();
			// restore weight
			caffe_copy(count, w_buffer_.gpu_data(), this->blobs_[0]->mutable_gpu_data());
		}

		/*const Dtype* data1 = top[0]->cpu_data();
		std::cout << "out1:";
		for (int i = 0; i < top[0]->count(); ++i) {
			std::cout << ", " << data1[i];
		}
		std::cout << std::endl;

		for (int i = 0; i < bottom.size(); ++i) {
			const Dtype* bottom_data = bottom[i]->gpu_data();
			Dtype* top_data = top[i]->mutable_gpu_data();
			for (int n = 0; n < this->num_; ++n) {
				this->forward_gpu_gemm(bottom_data + n * this->bottom_dim_, weight,
					top_data + n * this->top_dim_);
				if (this->bias_term_) {
					const Dtype* bias = this->blobs_[1]->gpu_data();
					this->forward_gpu_bias(top_data + n * this->top_dim_, bias);
				}
			}
		}

		const Dtype* data2 = top[0]->cpu_data();
		std::cout << "out2:";
		for (int i = 0; i < top[0]->count(); ++i) {
			std::cout << ", " << data2[i];
		}
		std::cout << std::endl;*/
		cudaProfilerStop();
	}

	template <typename Dtype>
	__global__ void A_backwark_kernel(const int nThreads, const int kernel_dim,
		const Dtype* w_sign, const Dtype* w_hat_grad, Dtype* A_grad) {
		CUDA_KERNEL_LOOP(index, nThreads) {
			Dtype sum = Dtype(0.);
			for (int i = 0; i < kernel_dim; ++i) {
				sum += w_sign[index*kernel_dim + i] * w_hat_grad[index*kernel_dim + i];
			}
			A_grad[index] = sum / static_cast<Dtype>(kernel_dim);
		}
	}

	template <typename Dtype>
	__global__ void meancenter_backwark_kernel(const int nThreads, const int num, const int channels, const int height, const int width,
		const Dtype* w_hat_grad, Dtype* meancenter_grad) {
		CUDA_KERNEL_LOOP(index, nThreads) {
			const int w = index % width;
			const int h = (index / width) % height;
			const int n = index / width / height;
			Dtype sum = Dtype(0.0);
			for (int c = 0; c < channels; ++c) {
				sum += w_hat_grad[((n*channels + c)*height + h)*width + w];
			}
			meancenter_grad[(n*height + h)*width + w] = sum / static_cast<Dtype>(channels);
		}
	}

	template <typename Dtype>
	__global__ void w_backwark_kernel(const int nThreads, const int num, const int channels, const int height, const int width,
		const Dtype* w, const Dtype* w_sign, const Dtype* w_hat_grad,
		const Dtype* A, const Dtype* A_grad, const Dtype* meancenter_grad, Dtype* w_grad) {
		CUDA_KERNEL_LOOP(index, nThreads) {
			const int w1 = index % width;
			const int h = (index / width) % height;
			const int n = index / width / height / channels;
			w_grad[index] = (w_sign[index] * A_grad[n] + w_hat_grad[index] * A[n] * (w[index] <= Dtype(1.) && w[index] >= Dtype(-1.)))
				* (Dtype(1.) - meancenter_grad[(n*height+h)*width+w1]);
		}
	}

	template <typename Dtype>
	void BinaryConvolutionV2Layer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
		const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
		const Dtype* weight = this->blobs_[0]->gpu_data();
		Dtype* weight_diff = this->blobs_[0]->mutable_gpu_diff();
		for (int i = 0; i < top.size(); ++i) {
			const Dtype* top_diff = top[i]->gpu_diff();
			// Bias gradient, if necessary.
			if (this->bias_term_ && this->param_propagate_down_[1]) {
				Dtype* bias_diff = this->blobs_[1]->mutable_gpu_diff();
				for (int n = 0; n < this->num_; ++n) {
					this->backward_gpu_bias(bias_diff, top_diff + n * this->top_dim_);
				}
			}
			if (this->param_propagate_down_[0] || propagate_down[i]) {
				const Dtype* bottom_data = bottom[i]->gpu_data();
				Dtype* bottom_diff = bottom[i]->mutable_gpu_diff();
				for (int n = 0; n < this->num_; ++n) {
					// gradient w.r.t. weight. Note that we will accumulate diffs.
					if (this->param_propagate_down_[0]) {
						this->weight_gpu_gemm(bottom_data + n * this->bottom_dim_,
							top_diff + n * this->top_dim_, weight_diff);
					}
					// gradient w.r.t. bottom data, if necessary.
					if (propagate_down[i]) {
						this->backward_gpu_gemm(top_diff + n * this->top_dim_, weight,
							bottom_diff + n * this->bottom_dim_);
					}
				}
			}
		}

		if (this->phase_ == TRAIN) {
			const int count = this->blobs_[0]->count();
			const int num = this->blobs_[0]->num();
			const int channels = this->blobs_[0]->channels();
			const int height = this->blobs_[0]->height();
			const int width = this->blobs_[0]->width();
			const int kernel_dim = this->blobs_[0]->count(1);
			// restore weight
			caffe_copy(count, w_buffer_.gpu_data(), this->blobs_[0]->mutable_gpu_data());
			// compute A grad
			caffe_gpu_mul(count, binary_w_.gpu_diff(), this->blobs_[0]->gpu_diff(), w_buffer_.mutable_gpu_diff());
			caffe_gpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, num,
				1, kernel_dim, Dtype(1. / kernel_dim), w_buffer_.gpu_diff(), multiplier_.gpu_data(),
				(Dtype)0., A_.mutable_gpu_diff());
			// compute meancenter grad
			if (height == 1 && width == 1) {
				caffe_gpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, num,
					1, channels, Dtype(1. / channels), this->blobs_[0]->gpu_diff(), multiplier_.gpu_data(),
					(Dtype)0., meancenter_.mutable_gpu_diff());
			}
			else {
				meancenter_backwark_kernel<Dtype> << <CAFFE_GET_BLOCKS(num*height*width), CAFFE_CUDA_NUM_THREADS >> >(
					num*height*width, num, channels, height, width,
					this->blobs_[0]->gpu_diff(), meancenter_.mutable_gpu_diff());
			}
			// compute w grad
			w_backwark_kernel<Dtype> << <CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS >> >(
				count, num, channels, height, width,
				this->blobs_[0]->gpu_data(), binary_w_.gpu_diff(), this->blobs_[0]->gpu_diff(),
				A_.gpu_data(), A_.gpu_diff(), meancenter_.gpu_diff(), this->blobs_[0]->mutable_gpu_diff());
		}
	}

	template void BinaryConvolutionV2Layer<float>::binarizeGPUTo(Blob<float>* weights);
	template void BinaryConvolutionV2Layer<double>::binarizeGPUTo(Blob<double>* weights);

	INSTANTIATE_LAYER_GPU_FUNCS(BinaryConvolutionV2Layer);

}  // namespace caffe
