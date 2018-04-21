#include <vector>

#include "caffe/layers/binary_conv_layer.hpp"

namespace caffe {

	template <typename Dtype>
	__global__ void binarize(const int nThreads, const int kernel_dim,
		const Dtype* w, const Dtype* A, Dtype* w_bin, Dtype* w_sign) {
		CUDA_KERNEL_LOOP(index, nThreads) {
			int num = index / kernel_dim;
			w_sign[index] = sign(w[index]);
			w_bin[index] = w_sign[index] * A[num];
		}
	}

	template <typename Dtype>
	__global__ void calc_A(const int nThreads, const int kernel_dim,
		const Dtype* w, Dtype* A) {
		CUDA_KERNEL_LOOP(index, nThreads) {
			Dtype sum = Dtype(0.0);
			for (int i = 0; i < kernel_dim; ++i) {
				sum += std::abs(w[index*kernel_dim + i]);
			}
			A[index] = sum / static_cast<Dtype>(kernel_dim);
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
	void BinaryConvolutionLayer<Dtype>::binarizeGPUTo(const Blob<Dtype>* weights, Blob<Dtype>* wb) {
		CHECK_EQ(weights->count(), wb->count());
		CHECK_EQ(weights->num(), A_.num());
		const int count = weights->count();
		const int num = weights->num();
		const int kernel_dim = weights->count(1);
		// compute A
		calc_A<Dtype> << <CAFFE_GET_BLOCKS(num), CAFFE_CUDA_NUM_THREADS >> >(
			num, kernel_dim, weights->gpu_data(), A_.mutable_gpu_data());
		// compute sign(w) and A*sign(w)
		binarize<Dtype> << <CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS >> >(
			count, kernel_dim, weights->gpu_data(), A_.gpu_data(), wb->mutable_gpu_data(), wb->mutable_gpu_diff());
	}

	template <typename Dtype>
	void BinaryConvolutionLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
		const vector<Blob<Dtype>*>& top) {
		if (this->phase_ == TRAIN) {
			const int count = this->blobs_[0]->count();
			const int num = this->blobs_[0]->num();
			const int channels = this->blobs_[0]->channels();
			const int height = this->blobs_[0]->height();
			const int width = this->blobs_[0]->width();
			// compute mean
			calc_meancenter<Dtype> << <CAFFE_GET_BLOCKS(count / channels), CAFFE_CUDA_NUM_THREADS >> >(
				count / channels, num, channels, height, width,
				this->blobs_[0]->gpu_data(), meancenter_.mutable_gpu_data());
			// subtract mean and clip weight to [-1,1]
			meancenter_remove_and_clamp<Dtype> << <CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS >> >(
				count, num, channels, height, width,
				meancenter_.gpu_data(), this->blobs_[0]->mutable_gpu_data(), Dtype(-1.0), Dtype(1.0));
			// binary weight, binary_w_'s data hold A*sign(w), binary_w_'s diff hold sign(w)
			binarizeGPUTo(&(*this->blobs_[0]), &binary_w_);
			// store weight to buffer
			caffe_copy(count, this->blobs_[0]->gpu_data(), w_buffer_.mutable_gpu_data());
			// copy binary weight to weight
			caffe_copy(count, binary_w_.gpu_data(), this->blobs_[0]->mutable_gpu_data());
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
			meancenter_remove_and_clamp<Dtype> << <CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS >> >(
				count, num, channels, height, width,
				meancenter_.gpu_data(), this->blobs_[0]->mutable_gpu_data(), Dtype(-1.0), Dtype(1.0));
			// binary weight, binary_w_'s data hold A*sign(w), binary_w_'s diff hold sign(w)
			binarizeGPUTo(&(*this->blobs_[0]), &binary_w_);
			// copy binary weight to weight
			caffe_copy(count, binary_w_.gpu_data(), this->blobs_[0]->mutable_gpu_data());
		}

		const Dtype* weight = this->blobs_[0]->gpu_data();
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
		if (this->phase_ == TEST) {
			const int count = this->blobs_[0]->count();
			// restore weight
			caffe_copy(count, binary_w_.gpu_data(), this->blobs_[0]->mutable_gpu_data());
		}
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
	void BinaryConvolutionLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
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
			A_backwark_kernel<Dtype> << <CAFFE_GET_BLOCKS(num), CAFFE_CUDA_NUM_THREADS >> >(
				num, kernel_dim, binary_w_.gpu_diff(), this->blobs_[0]->gpu_diff(), A_.mutable_gpu_diff());
			// compute meancenter grad
			meancenter_backwark_kernel<Dtype> << <CAFFE_GET_BLOCKS(num*height*width), CAFFE_CUDA_NUM_THREADS >> >(
				num*height*width, num, channels, height, width, 
				this->blobs_[0]->gpu_diff(), meancenter_.mutable_gpu_diff());
			// compute w grad
			w_backwark_kernel<Dtype> << <CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS >> >(
				count, num, channels, height, width,
				this->blobs_[0]->gpu_data(), binary_w_.gpu_diff(), this->blobs_[0]->gpu_diff(),
				A_.gpu_data(), A_.gpu_diff(), meancenter_.gpu_diff(), this->blobs_[0]->mutable_gpu_diff());
		}
	}

	template void BinaryConvolutionLayer<float>::binarizeGPUTo(const Blob<float>* weights, Blob<float>* wb);
	template void BinaryConvolutionLayer<double>::binarizeGPUTo(const Blob<double>* weights, Blob<double>* wb);

	INSTANTIATE_LAYER_GPU_FUNCS(BinaryConvolutionLayer);

}  // namespace caffe
