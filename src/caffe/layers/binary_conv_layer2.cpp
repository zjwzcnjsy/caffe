#include <algorithm>
#include <vector>

#include "caffe/filler.hpp"
#include "caffe/layers/binary_conv_layer2.hpp"
#include "caffe/util/im2col.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

template <typename Dtype>
void BinaryConvolutionV2Layer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  // Configure the kernel size, padding, stride, and inputs.
	BinaryConvolutionParameter conv_param = this->layer_param_.binary_convolution_param();
  force_nd_im2col_ = conv_param.force_nd_im2col();
  channel_axis_ = bottom[0]->CanonicalAxisIndex(conv_param.axis());
  const int first_spatial_axis = channel_axis_ + 1;
  const int num_axes = bottom[0]->num_axes();
  num_spatial_axes_ = num_axes - first_spatial_axis;
  CHECK_GE(num_spatial_axes_, 0);
  vector<int> spatial_dim_blob_shape(1, std::max(num_spatial_axes_, 1));
  // Setup filter kernel dimensions (kernel_shape_).
  kernel_shape_.Reshape(spatial_dim_blob_shape);
  int* kernel_shape_data = kernel_shape_.mutable_cpu_data();
  if (conv_param.has_kernel_h() || conv_param.has_kernel_w()) {
    CHECK_EQ(num_spatial_axes_, 2)
        << "kernel_h & kernel_w can only be used for 2D convolution.";
    CHECK_EQ(0, conv_param.kernel_size_size())
        << "Either kernel_size or kernel_h/w should be specified; not both.";
    kernel_shape_data[0] = conv_param.kernel_h();
    kernel_shape_data[1] = conv_param.kernel_w();
  } else {
    const int num_kernel_dims = conv_param.kernel_size_size();
    CHECK(num_kernel_dims == 1 || num_kernel_dims == num_spatial_axes_)
        << "kernel_size must be specified once, or once per spatial dimension "
        << "(kernel_size specified " << num_kernel_dims << " times; "
        << num_spatial_axes_ << " spatial dims).";
      for (int i = 0; i < num_spatial_axes_; ++i) {
        kernel_shape_data[i] =
            conv_param.kernel_size((num_kernel_dims == 1) ? 0 : i);
      }
  }
  for (int i = 0; i < num_spatial_axes_; ++i) {
    CHECK_GT(kernel_shape_data[i], 0) << "Filter dimensions must be nonzero.";
  }
  // Setup stride dimensions (stride_).
  stride_.Reshape(spatial_dim_blob_shape);
  int* stride_data = stride_.mutable_cpu_data();
  if (conv_param.has_stride_h() || conv_param.has_stride_w()) {
    CHECK_EQ(num_spatial_axes_, 2)
        << "stride_h & stride_w can only be used for 2D convolution.";
    CHECK_EQ(0, conv_param.stride_size())
        << "Either stride or stride_h/w should be specified; not both.";
    stride_data[0] = conv_param.stride_h();
    stride_data[1] = conv_param.stride_w();
  } else {
    const int num_stride_dims = conv_param.stride_size();
    CHECK(num_stride_dims == 0 || num_stride_dims == 1 ||
          num_stride_dims == num_spatial_axes_)
        << "stride must be specified once, or once per spatial dimension "
        << "(stride specified " << num_stride_dims << " times; "
        << num_spatial_axes_ << " spatial dims).";
    const int kDefaultStride = 1;
    for (int i = 0; i < num_spatial_axes_; ++i) {
      stride_data[i] = (num_stride_dims == 0) ? kDefaultStride :
          conv_param.stride((num_stride_dims == 1) ? 0 : i);
      CHECK_GT(stride_data[i], 0) << "Stride dimensions must be nonzero.";
    }
  }
  // Setup pad dimensions (pad_).
  pad_.Reshape(spatial_dim_blob_shape);
  int* pad_data = pad_.mutable_cpu_data();
  if (conv_param.has_pad_h() || conv_param.has_pad_w()) {
    CHECK_EQ(num_spatial_axes_, 2)
        << "pad_h & pad_w can only be used for 2D convolution.";
    CHECK_EQ(0, conv_param.pad_size())
        << "Either pad or pad_h/w should be specified; not both.";
    pad_data[0] = conv_param.pad_h();
    pad_data[1] = conv_param.pad_w();
  } else {
    const int num_pad_dims = conv_param.pad_size();
    CHECK(num_pad_dims == 0 || num_pad_dims == 1 ||
          num_pad_dims == num_spatial_axes_)
        << "pad must be specified once, or once per spatial dimension "
        << "(pad specified " << num_pad_dims << " times; "
        << num_spatial_axes_ << " spatial dims).";
    const int kDefaultPad = 0;
    for (int i = 0; i < num_spatial_axes_; ++i) {
      pad_data[i] = (num_pad_dims == 0) ? kDefaultPad :
          conv_param.pad((num_pad_dims == 1) ? 0 : i);
    }
  }
  // Setup dilation dimensions (dilation_).
  dilation_.Reshape(spatial_dim_blob_shape);
  int* dilation_data = dilation_.mutable_cpu_data();
  const int num_dilation_dims = conv_param.dilation_size();
  CHECK(num_dilation_dims == 0 || num_dilation_dims == 1 ||
        num_dilation_dims == num_spatial_axes_)
      << "dilation must be specified once, or once per spatial dimension "
      << "(dilation specified " << num_dilation_dims << " times; "
      << num_spatial_axes_ << " spatial dims).";
  const int kDefaultDilation = 1;
  for (int i = 0; i < num_spatial_axes_; ++i) {
    dilation_data[i] = (num_dilation_dims == 0) ? kDefaultDilation :
                       conv_param.dilation((num_dilation_dims == 1) ? 0 : i);
  }
  // Special case: im2col is the identity for 1x1 convolution with stride 1
  // and no padding, so flag for skipping the buffer and transformation.
  is_1x1_ = true;
  for (int i = 0; i < num_spatial_axes_; ++i) {
    is_1x1_ &=
        kernel_shape_data[i] == 1 && stride_data[i] == 1 && pad_data[i] == 0;
    if (!is_1x1_) { break; }
  }
  // Configure output channels and groups.
  channels_ = bottom[0]->shape(channel_axis_);
  num_output_ = this->layer_param_.binary_convolution_param().num_output();
  CHECK_GT(num_output_, 0);
  group_ = this->layer_param_.binary_convolution_param().group();
  CHECK_EQ(channels_ % group_, 0);
  CHECK_EQ(num_output_ % group_, 0)
      << "Number of output should be multiples of group.";
  if (reverse_dimensions()) {
    conv_out_channels_ = channels_;
    conv_in_channels_ = num_output_;
  } else {
    conv_out_channels_ = num_output_;
    conv_in_channels_ = channels_;
  }
  // Handle the parameters: weights and biases.
  // - blobs_[0] holds the filter weights
  // - blobs_[1] holds the biases (optional)
  vector<int> weight_shape(2);
  weight_shape[0] = conv_out_channels_;
  weight_shape[1] = conv_in_channels_ / group_;
  for (int i = 0; i < num_spatial_axes_; ++i) {
    weight_shape.push_back(kernel_shape_data[i]);
  }
  bias_term_ = this->layer_param_.binary_convolution_param().bias_term();
  vector<int> bias_shape(bias_term_, num_output_);
  if (this->blobs_.size() > 0) {
    CHECK_EQ(1 + bias_term_, this->blobs_.size())
        << "Incorrect number of weight blobs.";
    if (weight_shape != this->blobs_[0]->shape()) {
      Blob<Dtype> weight_shaped_blob(weight_shape);
      LOG(FATAL) << "Incorrect weight shape: expected shape "
          << weight_shaped_blob.shape_string() << "; instead, shape was "
          << this->blobs_[0]->shape_string();
    }
    if (bias_term_ && bias_shape != this->blobs_[1]->shape()) {
      Blob<Dtype> bias_shaped_blob(bias_shape);
      LOG(FATAL) << "Incorrect bias shape: expected shape "
          << bias_shaped_blob.shape_string() << "; instead, shape was "
          << this->blobs_[1]->shape_string();
    }
    LOG(INFO) << "Skipping parameter initialization";
  } else {
    if (bias_term_) {
      this->blobs_.resize(2);
    } else {
      this->blobs_.resize(1);
    }
    // Initialize and fill the weights:
    // output channels x input channels per-group x kernel height x kernel width
    this->blobs_[0].reset(new Blob<Dtype>(weight_shape));
    shared_ptr<Filler<Dtype> > weight_filler(GetFiller<Dtype>(
        this->layer_param_.binary_convolution_param().weight_filler()));
    weight_filler->Fill(this->blobs_[0].get());
    // If necessary, initialize and fill the biases.
    if (bias_term_) {
      this->blobs_[1].reset(new Blob<Dtype>(bias_shape));
      shared_ptr<Filler<Dtype> > bias_filler(GetFiller<Dtype>(
          this->layer_param_.binary_convolution_param().bias_filler()));
      bias_filler->Fill(this->blobs_[1].get());
    }
  }
  kernel_dim_ = this->blobs_[0]->count(1);
  weight_offset_ = conv_out_channels_ * kernel_dim_ / group_;
  // Propagate gradients to the parameters (as directed by backward pass).
  this->param_propagate_down_.resize(this->blobs_.size(), true);
}

template <typename Dtype>
void BinaryConvolutionV2Layer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  const int first_spatial_axis = channel_axis_ + 1;
  CHECK_EQ(bottom[0]->num_axes(), first_spatial_axis + num_spatial_axes_)
      << "bottom num_axes may not change.";
  num_ = bottom[0]->count(0, channel_axis_);
  CHECK_EQ(bottom[0]->shape(channel_axis_), channels_)
      << "Input size incompatible with convolution kernel.";
  // TODO: generalize to handle inputs of different shapes.
  for (int bottom_id = 1; bottom_id < bottom.size(); ++bottom_id) {
    CHECK(bottom[0]->shape() == bottom[bottom_id]->shape())
        << "All inputs must have the same shape.";
  }
  // Shape the tops.
  bottom_shape_ = &bottom[0]->shape();
  compute_output_shape();
  vector<int> top_shape(bottom[0]->shape().begin(),
      bottom[0]->shape().begin() + channel_axis_);
  top_shape.push_back(num_output_);
  for (int i = 0; i < num_spatial_axes_; ++i) {
    top_shape.push_back(output_shape_[i]);
  }
  for (int top_id = 0; top_id < top.size(); ++top_id) {
    top[top_id]->Reshape(top_shape);
  }
  if (reverse_dimensions()) {
    conv_out_spatial_dim_ = bottom[0]->count(first_spatial_axis);
  } else {
    conv_out_spatial_dim_ = top[0]->count(first_spatial_axis);
  }
  col_offset_ = kernel_dim_ * conv_out_spatial_dim_;
  output_offset_ = conv_out_channels_ * conv_out_spatial_dim_ / group_;
  // Setup input dimensions (conv_input_shape_).
  vector<int> bottom_dim_blob_shape(1, num_spatial_axes_ + 1);
  conv_input_shape_.Reshape(bottom_dim_blob_shape);
  int* conv_input_shape_data = conv_input_shape_.mutable_cpu_data();
  for (int i = 0; i < num_spatial_axes_ + 1; ++i) {
    if (reverse_dimensions()) {
      conv_input_shape_data[i] = top[0]->shape(channel_axis_ + i);
    } else {
      conv_input_shape_data[i] = bottom[0]->shape(channel_axis_ + i);
    }
  }
  // The im2col result buffer will only hold one image at a time to avoid
  // overly large memory usage. In the special case of 1x1 convolution
  // it goes lazily unused to save memory.
  col_buffer_shape_.clear();
  col_buffer_shape_.push_back(kernel_dim_ * group_);
  for (int i = 0; i < num_spatial_axes_; ++i) {
    if (reverse_dimensions()) {
      col_buffer_shape_.push_back(input_shape(i + 1));
    } else {
      col_buffer_shape_.push_back(output_shape_[i]);
    }
  }
  col_buffer_.Reshape(col_buffer_shape_);
  bottom_dim_ = bottom[0]->count(channel_axis_);
  top_dim_ = top[0]->count(channel_axis_);
  num_kernels_im2col_ = conv_in_channels_ * conv_out_spatial_dim_;
  num_kernels_col2im_ = reverse_dimensions() ? top_dim_ : bottom_dim_;
  // Set up the all ones "bias multiplier" for adding biases by BLAS
  out_spatial_dim_ = top[0]->count(first_spatial_axis);
  if (bias_term_) {
    vector<int> bias_multiplier_shape(1, out_spatial_dim_);
    bias_multiplier_.Reshape(bias_multiplier_shape);
    caffe_set(bias_multiplier_.count(), Dtype(1),
        bias_multiplier_.mutable_cpu_data());
  }

	binary_w_.ReshapeLike(*(this->blobs_[0]));
	w_buffer_.ReshapeLike(*(this->blobs_[0]));
	multiplier_.ReshapeLike(*(this->blobs_[0]));
	caffe_set(multiplier_.count(), Dtype(1.), multiplier_.mutable_cpu_data());
	vector<int> meancenter_shape = this->blobs_[0]->shape();
	meancenter_shape[1] = 1;
	meancenter_.Reshape(meancenter_shape);
	vector<int> A_shape(1, this->blobs_[0]->num());
	A_.Reshape(A_shape);

	vector<int> uiA_shape(2);
	uiA_shape[0] = conv_out_channels_;
	uiA_shape[1] = kernel_dim_;
	uiA_.Reshape(uiA_shape);
	uiB_.Reshape(col_buffer_shape_);
}


template <typename Dtype>
void BinaryConvolutionV2Layer<Dtype>::compute_output_shape() {
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
void BinaryConvolutionV2Layer<Dtype>::forward_cpu_gemm(const Dtype* input,
    const Dtype* weights, Dtype* output, bool skip_im2col) {
  const Dtype* col_buff = input;
  if (!is_1x1_) {
    if (!skip_im2col) {
      conv_im2col_cpu(input, col_buffer_.mutable_cpu_data());
    }
    col_buff = col_buffer_.cpu_data();
  }
  for (int g = 0; g < group_; ++g) {
    caffe_cpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, conv_out_channels_ /
        group_, conv_out_spatial_dim_, kernel_dim_,
        (Dtype)1., weights + weight_offset_ * g, col_buff + col_offset_ * g,
        (Dtype)0., output + output_offset_ * g);
  }
}

template <typename Dtype>
void BinaryConvolutionV2Layer<Dtype>::forward_cpu_bias(Dtype* output,
    const Dtype* bias) {
  caffe_cpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, num_output_,
      out_spatial_dim_, 1, (Dtype)1., bias, bias_multiplier_.cpu_data(),
      (Dtype)1., output);
}

template <typename Dtype>
void BinaryConvolutionV2Layer<Dtype>::backward_cpu_gemm(const Dtype* output,
    const Dtype* weights, Dtype* input) {
  Dtype* col_buff = col_buffer_.mutable_cpu_data();
  if (is_1x1_) {
    col_buff = input;
  }
  for (int g = 0; g < group_; ++g) {
    caffe_cpu_gemm<Dtype>(CblasTrans, CblasNoTrans, kernel_dim_,
        conv_out_spatial_dim_, conv_out_channels_ / group_,
        (Dtype)1., weights + weight_offset_ * g, output + output_offset_ * g,
        (Dtype)0., col_buff + col_offset_ * g);
  }
  if (!is_1x1_) {
    conv_col2im_cpu(col_buff, input);
  }
}

template <typename Dtype>
void BinaryConvolutionV2Layer<Dtype>::weight_cpu_gemm(const Dtype* input,
    const Dtype* output, Dtype* weights) {
  const Dtype* col_buff = input;
  if (!is_1x1_) {
    conv_im2col_cpu(input, col_buffer_.mutable_cpu_data());
    col_buff = col_buffer_.cpu_data();
  }
  for (int g = 0; g < group_; ++g) {
    caffe_cpu_gemm<Dtype>(CblasNoTrans, CblasTrans, conv_out_channels_ / group_,
        kernel_dim_, conv_out_spatial_dim_,
        (Dtype)1., output + output_offset_ * g, col_buff + col_offset_ * g,
        (Dtype)1., weights + weight_offset_ * g);
  }
}

template <typename Dtype>
void BinaryConvolutionV2Layer<Dtype>::backward_cpu_bias(Dtype* bias,
    const Dtype* input) {
  caffe_cpu_gemv<Dtype>(CblasNoTrans, num_output_, out_spatial_dim_, 1.,
      input, bias_multiplier_.cpu_data(), 1., bias);
}

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
void BinaryConvolutionV2Layer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
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
void BinaryConvolutionV2Layer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
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
void BinaryConvolutionV2Layer<Dtype>::binarizeCPUTo(const Blob<Dtype>* weights, Blob<Dtype>* wb) {
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
STUB_GPU(BinaryConvolutionV2Layer);
#endif

INSTANTIATE_CLASS(BinaryConvolutionV2Layer);
REGISTER_LAYER_CLASS(BinaryConvolutionV2);

}  // namespace caffe
