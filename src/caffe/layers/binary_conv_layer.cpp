#include <vector>

#include "caffe/layers/binary_conv_layer.hpp"

namespace caffe {

template <typename Dtype>
inline void calc_meancenter(const int num, const int channels, const int height, const int width,
	const Dtype* weights, Dtype* center, const Dtype* multiplier)
{
	const int kernel_dim = channels * height * width;
	const int kernel_spatial_dim = height * width;
	if (height == 1 && width == 1) {
		caffe_cpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, num,
			1, channels, Dtype(1. / channels), weights, multiplier,
			(Dtype)0., center);
		return;
	}

	for (int n = 0; n < num; ++n) {
		caffe_cpu_gemm<Dtype>(CblasTrans, CblasNoTrans, kernel_spatial_dim,
			1, channels, Dtype(1. / channels), weights + n * kernel_dim, multiplier,
			(Dtype)0., center + n * kernel_spatial_dim);
	}
}

template <typename Dtype>
inline void expand_meancenter(const int num, const int channels, const int height, const int width,
	const Dtype* center, Dtype* buffer)
{
	const int spatial_dim = height*width;
	for (int n = 0; n < num; ++n) {
		const Dtype *pc = center + n*spatial_dim;
		for (int c = 0; c < channels; ++c) {
			Dtype *pb = buffer + (n*channels+c)*spatial_dim;
			caffe_copy<Dtype>(spatial_dim, pc, pb);
		}
	}
}

template <typename Dtype>
void meancenter_remove_and_clamp(const int num, const int channels, const int height, const int width,
	Dtype* weights, const Dtype* center, const Dtype minv=Dtype(-1.), const Dtype maxv = Dtype(1.))
{
	const int spatial_dim = height * width;
	const int kernel_dim = channels * height * width;
	for (int n = 0; n < num; ++n) {
		for (int c = 0; c < channels; ++c) {
			const Dtype* spatial_center = center + n*spatial_dim;
			for (int h = 0; h < height; ++h) {
				for (int w = 0; w < width; ++w) {
					Dtype v = *weights - *spatial_center++;
					if (v < minv) {
						v = minv;
					}
					else if (v > maxv) {
						v = maxv;
					}
					else {
						//nothing to do;
					}
					*weights++ = v;
				}
			}
		}
	}
}

template <typename Dtype>
inline void clamp(const int count, Dtype* weights, const Dtype minv = Dtype(-1.), const Dtype maxv = Dtype(1.))
{
	for (int i = 0; i < count; ++i) {
		Dtype v = weights[i];
		if (v < minv) {
			v = minv;
		}
		else if (v > maxv) {
			v = maxv;
		}
		else {
			//nothing to do;
		}
		weights[i] = v;
	}
}

template <typename Dtype>
void BinaryConvolutionLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  BaseConvolutionLayer<Dtype>::LayerSetUp(bottom, top);
  binary_w_.ReshapeLike(*(this->blobs_[0]));
  w_buffer_.ReshapeLike(*(this->blobs_[0]));
	multiplier_.ReshapeLike(*(this->blobs_[0]));
	caffe_set(multiplier_.count(), Dtype(1.), multiplier_.mutable_cpu_data());
  vector<int> meancenter_shape = this->blobs_[0]->shape();
  meancenter_shape[1] = 1;
  meancenter_.Reshape(meancenter_shape);
  vector<int> A_shape(1, this->blobs_[0]->num());
  A_.Reshape(A_shape);
}

template <typename Dtype>
void BinaryConvolutionLayer<Dtype>::compute_output_shape() {
  const int* kernel_shape_data = this->kernel_shape_.cpu_data();
  const int* stride_data = this->stride_.cpu_data();
  const int* pad_data = this->pad_.cpu_data();
  const int* dilation_data = this->dilation_.cpu_data();
  this->output_shape_.clear();
  for (int i = 0; i < this->num_spatial_axes_; ++i) {
    // i + 1 to skip channel axis
    const int input_dim = this->input_shape(i + 1);
    const int kernel_extent = dilation_data[i] * (kernel_shape_data[i] - 1) + 1;
    const int output_dim = (input_dim + 2 * pad_data[i] - kernel_extent)
        / stride_data[i] + 1;
    this->output_shape_.push_back(output_dim);
  }
}

template <typename Dtype>
void BinaryConvolutionLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
	if (this->phase_ == TRAIN) {
		const int count = this->blobs_[0]->count();
		const int num = this->blobs_[0]->num();
		const int channels = this->blobs_[0]->channels();
		const int height = this->blobs_[0]->height();
		const int width = this->blobs_[0]->width();
		// compute mean
		calc_meancenter(num, channels, height, width,
			this->blobs_[0]->cpu_data(), meancenter_.mutable_cpu_data(), multiplier_.cpu_data());
		// expand mean(n,h,w) to mean(n,c,h,w)
		//expand_meancenter(num, channels, height, width, meancenter_.cpu_data(), w_buffer_.mutable_cpu_diff());
		// subtract mean and clip weight to [-1,1]
		//caffe_sub(count, this->blobs_[0]->cpu_data(), w_buffer_.cpu_diff(), this->blobs_[0]->mutable_cpu_data());
		//clamp(count, this->blobs_[0]->mutable_cpu_data());
		meancenter_remove_and_clamp(num, channels, height, width,
			this->blobs_[0]->mutable_cpu_data(), meancenter_.cpu_data(), Dtype(-1.0), Dtype(1.0));

		// binary weight, binary_w_'s data hold A*sign(w), binary_w_'s diff hold sign(w)
		binarizeCPUTo(&(*this->blobs_[0]), &binary_w_);
		// store weight to buffer
		caffe_copy(this->blobs_[0]->count(), this->blobs_[0]->cpu_data(), w_buffer_.mutable_cpu_data());
		// copy binary weight to weight
		caffe_copy(binary_w_.count(), binary_w_.cpu_data(), this->blobs_[0]->mutable_cpu_data());
	}
	else {
		const int count = this->blobs_[0]->count();
		// store weight to buffer
		caffe_copy(this->blobs_[0]->count(), this->blobs_[0]->cpu_data(), w_buffer_.mutable_cpu_data());
		
		const int num = this->blobs_[0]->num();
		const int channels = this->blobs_[0]->channels();
		const int height = this->blobs_[0]->height();
		const int width = this->blobs_[0]->width();
		// compute mean
		calc_meancenter(num, channels, height, width,
			this->blobs_[0]->cpu_data(), meancenter_.mutable_cpu_data(), multiplier_.cpu_data());
		// subtract mean and clip weight to [-1,1]
		meancenter_remove_and_clamp(num, channels, height, width,
			this->blobs_[0]->mutable_cpu_data(), meancenter_.cpu_data(), Dtype(-1.0), Dtype(1.0));

		// binary weight, binary_w_'s data hold A*sign(w), binary_w_'s diff hold sign(w)
		binarizeCPUTo(&(*this->blobs_[0]), &binary_w_);
		// copy binary weight to weight
		caffe_copy(binary_w_.count(), binary_w_.cpu_data(), this->blobs_[0]->mutable_cpu_data());
	}
  
  const Dtype* weight = this->blobs_[0]->cpu_data();
  for (int i = 0; i < bottom.size(); ++i) {
    const Dtype* bottom_data = bottom[i]->cpu_data();
    Dtype* top_data = top[i]->mutable_cpu_data();
    for (int n = 0; n < this->num_; ++n) {
      this->forward_cpu_gemm(bottom_data + n * this->bottom_dim_, weight,
          top_data + n * this->top_dim_);
      if (this->bias_term_) {
        const Dtype* bias = this->blobs_[1]->cpu_data();
        this->forward_cpu_bias(top_data + n * this->top_dim_, bias);
      }
    }
  }
	if (this->phase_ == TEST) {
		const int count = this->blobs_[0]->count();
		// restore weight
		caffe_copy(count, w_buffer_.cpu_data(), this->blobs_[0]->mutable_cpu_data());
	}
}

template <typename Dtype>
void calc_A_grad(const int num, const int kernel_dim,
	const Dtype* w_sign, const Dtype* w_hat_grad, Dtype* A_grad) {
	for (int n = 0; n < num; ++n) {
		A_grad[n] = caffe_cpu_dot(kernel_dim, w_sign + n*kernel_dim, w_hat_grad + n*kernel_dim) / static_cast<Dtype>(kernel_dim);
	}
}

template <typename Dtype>
void calc_w_grad(const int count, const int channels, const int kernel_dim,
	const Dtype* w, const Dtype* w_sign, const Dtype* w_hat_grad,
	const Dtype* A, const Dtype* A_grad, Dtype* w_grad) {
	for (int i = 0; i < count; ++i) {
		int num = i / kernel_dim;
		w_grad[i] = (w_sign[i] * A_grad[num] + w_hat_grad[i] * A[num] * (w[i] <= Dtype(1.) && w[i] >= Dtype(-1.)))
			* (Dtype(1.) - Dtype(1.) / static_cast<Dtype>(channels))*kernel_dim*Dtype(1e+9);
	}
}

template <typename Dtype>
void BinaryConvolutionLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
  const Dtype* weight = this->blobs_[0]->cpu_data();
  Dtype* weight_diff = this->blobs_[0]->mutable_cpu_diff();
  for (int i = 0; i < top.size(); ++i) {
    const Dtype* top_diff = top[i]->cpu_diff();
    const Dtype* bottom_data = bottom[i]->cpu_data();
    Dtype* bottom_diff = bottom[i]->mutable_cpu_diff();
    // Bias gradient, if necessary.
    if (this->bias_term_ && this->param_propagate_down_[1]) {
      Dtype* bias_diff = this->blobs_[1]->mutable_cpu_diff();
      for (int n = 0; n < this->num_; ++n) {
        this->backward_cpu_bias(bias_diff, top_diff + n * this->top_dim_);
      }
    }
    if (this->param_propagate_down_[0] || propagate_down[i]) {
      for (int n = 0; n < this->num_; ++n) {
        // gradient w.r.t. weight. Note that we will accumulate diffs.
        if (this->param_propagate_down_[0]) {
          this->weight_cpu_gemm(bottom_data + n * this->bottom_dim_,
              top_diff + n * this->top_dim_, weight_diff);
        }
        // gradient w.r.t. bottom data, if necessary.
        if (propagate_down[i]) {
          this->backward_cpu_gemm(top_diff + n * this->top_dim_, weight,
              bottom_diff + n * this->bottom_dim_);
        }
      }
    }
  }
	if (this->phase_ == TRAIN) {
		const int count = this->blobs_[0]->count();
		const int num = this->blobs_[0]->num();
		const int channels = this->blobs_[0]->channels();
		const int kernel_dim = this->blobs_[0]->count(1);
		// restore weight
		caffe_copy(count, w_buffer_.cpu_data(), this->blobs_[0]->mutable_cpu_data());
		// compute A grad
		caffe_mul(count, binary_w_.cpu_diff(), this->blobs_[0]->cpu_diff(), w_buffer_.mutable_cpu_diff());
		caffe_cpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, num,
			1, kernel_dim, Dtype(1. / kernel_dim), w_buffer_.cpu_diff(), multiplier_.cpu_data(),
			(Dtype)0., A_.mutable_cpu_diff());
		//calc_A_grad(num, kernel_dim, binary_w_.cpu_diff(), this->blobs_[0]->cpu_diff(), A_.mutable_cpu_diff());
		// compute w grad
		calc_w_grad(count, channels, kernel_dim,
			this->blobs_[0]->cpu_data(), binary_w_.cpu_diff(), this->blobs_[0]->cpu_diff(),
			A_.cpu_data(), A_.cpu_diff(), this->blobs_[0]->mutable_cpu_diff());
	}
}

template <typename Dtype>
void BinaryConvolutionLayer<Dtype>::binarizeCPUTo(const Blob<Dtype>* weights, Blob<Dtype>* wb) {
	const int count = weights->count();
	const int num = weights->num();
	const int kernel_dim = weights->count(1);
	// compute A
	caffe_abs(count, weights->cpu_data(), wb->mutable_cpu_data());
	caffe_cpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, num,
		1, kernel_dim, Dtype(1. / kernel_dim), wb->cpu_data(), multiplier_.cpu_data(),
		(Dtype)0., A_.mutable_cpu_data());

	const int weight_dim = weights->count(1);
	const Dtype *w = weights->cpu_data();
	const Dtype *A = A_.cpu_data();
	Dtype *binary_weight = wb->mutable_cpu_data();
	caffe_cpu_sign(count, w, binary_weight);
	for (int n = 0; n < num; ++n) {
		caffe_scal(kernel_dim, A[n], binary_weight + n * kernel_dim);
	}
	/*for (int index = 0; index < count; index++) {
		const int num = index / kernel_dim;
		binary_weight[index] = A[num] * sign(w[index]);
	}*/
}


#ifdef CPU_ONLY
STUB_GPU(BinaryConvolutionLayer);
#endif

INSTANTIATE_CLASS(BinaryConvolutionLayer);
REGISTER_LAYER_CLASS(BinaryConvolution);

}  // namespace caffe
