#include <vector>

#include "caffe/layers/binary_conv_layer.hpp"

namespace caffe {

template <typename Dtype>
void BinaryConvolutionLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  BaseConvolutionLayer<Dtype>::LayerSetUp(bottom, top);
  binary_w_.ReshapeLike(*(this->blobs_[0]));
  w_buffer_.ReshapeLike(*(this->blobs_[0]));
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
  // binary weight
  binarizeCPUTo(&(*this->blobs_[0]), &binary_w_);
  // store weight to buffer
  caffe_copy(this->blobs_[0]->count(), this->blobs_[0]->cpu_data(), w_buffer_.mutable_cpu_data());
  // copy binary weight to weight
  caffe_copy(binary_w_.count(), binary_w_.cpu_data(), this->blobs_[0]->mutable_cpu_data());
  
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
  // restore weight
  caffe_copy(w_buffer_.count(), w_buffer_.cpu_data(), this->blobs_[0]->mutable_cpu_data());
  // TODO: compute w grad
}

template <typename Dtype>
void BinaryConvolutionLayer<Dtype>::binarizeCPUTo(const Blob<Dtype>* weights, Blob<Dtype>* wb) {
	const int weight_dim = weights->count(1);
	Dtype * A = A_.mutable_cpu_data();
	for (int num = 0; num < weights->num(); num++) {
		A[num] = caffe_cpu_asum(weight_dim, weights->cpu_data() + num * weight_dim) / Dtype(weight_dim);
	}
	for (int index = 0; index < weights->count(); index++) {
		const int num = index / weight_dim;
		wb->mutable_cpu_data()[index] = A[num] * sign(weights->cpu_data()[index]);
	}
}


#ifdef CPU_ONLY
STUB_GPU(BinaryConvolutionLayer);
#endif

INSTANTIATE_CLASS(BinaryConvolutionLayer);
REGISTER_LAYER_CLASS(BinaryConvolution);

}  // namespace caffe
