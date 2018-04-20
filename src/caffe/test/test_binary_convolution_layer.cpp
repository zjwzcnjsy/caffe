#include <vector>

#include "gtest/gtest.h"

#include "caffe/blob.hpp"
#include "caffe/common.hpp"
#include "caffe/filler.hpp"
#include "caffe/layers/binary_conv_layer.hpp"

#include "caffe/test/test_caffe_main.hpp"
#include "caffe/test/test_gradient_check_util.hpp"

namespace caffe {

#define sign(x) ((x)>=0?1:-1)

template <typename Dtype>
void calc_meancenter(const int count, const int num, const int channels, const int height, const int width,
	const Dtype* weights, Dtype* center)
{
	for (int i = 0; i < num*height*width; ++i) {
		center[i] = Dtype(0.);
	}
	for (int n = 0; n < num; ++n) {
		for (int c = 0; c < channels; ++c) {
			for (int h = 0; h < height; ++h) {
				for (int w = 0; w < width; ++w) {
					center[(n*height + h)*width + w] += weights[((n*channels + c)*height + h)*width + w] / static_cast<Dtype>(channels);
				}
			}
		}
	}
}

template <typename Dtype>
void meancenter_remove_and_clamp(const int count, const int num, const int channels, const int height, const int width,
	const Dtype* weights, const Dtype* center, Dtype* out_w, const Dtype minv = Dtype(-1.), const Dtype maxv = Dtype(1.))
{
	for (int n = 0; n < num; ++n) {
		for (int c = 0; c < channels; ++c) {
			for (int h = 0; h < height; ++h) {
				for (int w = 0; w < width; ++w) {
					Dtype v = weights[((n*channels + c)*height + h)*width + w] - center[(n*height + h)*width + w];
					if (v < minv) {
						v = minv;
					}
					else if (v > maxv) {
						v = maxv;
					}
					else {
						//nothing to do;
					}
					out_w[((n*channels + c)*height + h)*width + w] = v;
				}
			}
		}
	}
}

template <typename Dtype>
void binarizeCPUTo(const Blob<Dtype>* weights, Blob<Dtype>* wb) {
	wb->ReshapeLike(*weights);
	Blob<Dtype> meancenter_, A_;
	vector<int> meancenter_shape = weights->shape();
	meancenter_shape[1] = 1;
	meancenter_.Reshape(meancenter_shape);
	vector<int> A_shape(1, weights->num());
	A_.Reshape(A_shape);

	const int count = weights->count();
	const int num = weights->num();
	const int channels = weights->channels();
	const int height = weights->height();
	const int width = weights->width();

	// compute mean
	calc_meancenter(count, num, channels, height, width,
		weights->cpu_data(), meancenter_.mutable_cpu_data());
	// subtract mean and clip weight to [-1,1]
	meancenter_remove_and_clamp(count, num, channels, height, width,
		weights->cpu_data(), meancenter_.cpu_data(), wb->mutable_cpu_data(), Dtype(-1.0), Dtype(1.0));

	const int weight_dim = channels*height*width;
	Dtype * A = A_.mutable_cpu_data();
	for (int i = 0; i < num; i++) {
		A[i] = caffe_cpu_asum(weight_dim, wb->cpu_data() + i * weight_dim) / Dtype(weight_dim);
	}
	for (int i = 0; i < count; i++) {
		const int n = i / weight_dim;
		wb->mutable_cpu_data()[i] = A[n] * sign(wb->cpu_data()[i]);
	}
}

template void binarizeCPUTo(const Blob<float>* weights, Blob<float>* wb);
template void binarizeCPUTo(const Blob<double>* weights, Blob<double>* wb);

// Reference convolution for checking results:
// accumulate through explicit loops over input, output, and filters.
template <typename Dtype>
void caffe_bin_conv(const Blob<Dtype>* in, ConvolutionParameter* conv_param,
    const Blob<Dtype>* weights,
    Blob<Dtype>* out) {
	Blob<Dtype> wb;
	binarizeCPUTo(weights, &wb);
  const bool has_depth = (out->num_axes() == 5);
  if (!has_depth) { CHECK_EQ(4, out->num_axes()); }
  // Kernel size, stride, and pad
  int kernel_h, kernel_w;
  if (conv_param->has_kernel_h() || conv_param->has_kernel_w()) {
    kernel_h = conv_param->kernel_h();
    kernel_w = conv_param->kernel_w();
  } else {
    kernel_h = kernel_w = conv_param->kernel_size(0);
  }
  int pad_h, pad_w;
  if (conv_param->has_pad_h() || conv_param->has_pad_w()) {
    pad_h = conv_param->pad_h();
    pad_w = conv_param->pad_w();
  } else {
    pad_h = pad_w = conv_param->pad_size() ? conv_param->pad(0) : 0;
  }
  int stride_h, stride_w;
  if (conv_param->has_stride_h() || conv_param->has_stride_w()) {
    stride_h = conv_param->stride_h();
    stride_w = conv_param->stride_w();
  } else {
    stride_h = stride_w = conv_param->stride_size() ? conv_param->stride(0) : 1;
  }
  int dilation_h, dilation_w;
  dilation_h = dilation_w = conv_param->dilation_size() ?
                            conv_param->dilation(0) : 1;
  int kernel_d, pad_d, stride_d, dilation_d;
  if (has_depth) {
    kernel_d = kernel_h;
    stride_d = stride_h;
    pad_d = pad_h;
    dilation_d = dilation_h;
  } else {
    kernel_d = stride_d = dilation_d = 1;
    pad_d = 0;
  }
  // Groups
  int groups = conv_param->group();
  int o_g = out->shape(1) / groups;
  int k_g = in->shape(1) / groups;
  int o_head, k_head;
  // Convolution
  vector<int> weight_offset(4 + has_depth);
  vector<int> in_offset(4 + has_depth);
  vector<int> out_offset(4 + has_depth);
  Dtype* out_data = out->mutable_cpu_data();
  for (int n = 0; n < out->shape(0); n++) {
    for (int g = 0; g < groups; g++) {
      o_head = o_g * g;
      k_head = k_g * g;
      for (int o = 0; o < o_g; o++) {
        for (int k = 0; k < k_g; k++) {
          for (int z = 0; z < (has_depth ? out->shape(2) : 1); z++) {
            for (int y = 0; y < out->shape(2 + has_depth); y++) {
              for (int x = 0; x < out->shape(3 + has_depth); x++) {
                for (int r = 0; r < kernel_d; r++) {
                  for (int p = 0; p < kernel_h; p++) {
                    for (int q = 0; q < kernel_w; q++) {
                      int in_z = z * stride_d - pad_d + r * dilation_d;
                      int in_y = y * stride_h - pad_h + p * dilation_h;
                      int in_x = x * stride_w - pad_w + q * dilation_w;
                      if (in_z >= 0 && in_z < (has_depth ? in->shape(2) : 1)
                          && in_y >= 0 && in_y < in->shape(2 + has_depth)
                          && in_x >= 0 && in_x < in->shape(3 + has_depth)) {
                        weight_offset[0] = o + o_head;
                        weight_offset[1] = k;
                        if (has_depth) { weight_offset[2] = r; }
                        weight_offset[2 + has_depth] = p;
                        weight_offset[3 + has_depth] = q;
                        in_offset[0] = n;
                        in_offset[1] = k + k_head;
                        if (has_depth) { in_offset[2] = in_z; }
                        in_offset[2 + has_depth] = in_y;
                        in_offset[3 + has_depth] = in_x;
                        out_offset[0] = n;
                        out_offset[1] = o + o_head;
                        if (has_depth) { out_offset[2] = z; }
                        out_offset[2 + has_depth] = y;
                        out_offset[3 + has_depth] = x;
                        out_data[out->offset(out_offset)] +=
                            in->data_at(in_offset)
                            * wb.data_at(weight_offset);
                      }
                    }
                  }
                }
              }
            }
          }
        }
      }
    }
  }
}

template void caffe_bin_conv(const Blob<float>* in,
    ConvolutionParameter* conv_param,
		const Blob<float>* weights,
    Blob<float>* out);
template void caffe_bin_conv(const Blob<double>* in,
    ConvolutionParameter* conv_param,
		const Blob<double>* weights,
    Blob<double>* out);

template <typename TypeParam>
class BinaryConvolutionLayerTest : public MultiDeviceTest<TypeParam> {
  typedef typename TypeParam::Dtype Dtype;

 protected:
	 BinaryConvolutionLayerTest()
      : blob_bottom_(new Blob<Dtype>(2, 3, 6, 4)),
        blob_bottom_2_(new Blob<Dtype>(2, 3, 6, 4)),
        blob_top_(new Blob<Dtype>()),
        blob_top_2_(new Blob<Dtype>()) {}
  virtual void SetUp() {
    // fill the values
    FillerParameter filler_param;
    filler_param.set_value(1.);
    GaussianFiller<Dtype> filler(filler_param);
    filler.Fill(this->blob_bottom_);
    filler.Fill(this->blob_bottom_2_);
    blob_bottom_vec_.push_back(blob_bottom_);
    blob_top_vec_.push_back(blob_top_);
  }

  virtual ~BinaryConvolutionLayerTest() {
    delete blob_bottom_;
    delete blob_bottom_2_;
    delete blob_top_;
    delete blob_top_2_;
  }

  virtual Blob<Dtype>* MakeReferenceTop(Blob<Dtype>* top) {
    this->ref_blob_top_.reset(new Blob<Dtype>());
    this->ref_blob_top_->ReshapeLike(*top);
    return this->ref_blob_top_.get();
  }

  Blob<Dtype>* const blob_bottom_;
  Blob<Dtype>* const blob_bottom_2_;
  Blob<Dtype>* const blob_top_;
  Blob<Dtype>* const blob_top_2_;
  shared_ptr<Blob<Dtype> > ref_blob_top_;
  vector<Blob<Dtype>*> blob_bottom_vec_;
  vector<Blob<Dtype>*> blob_top_vec_;
};

TYPED_TEST_CASE(BinaryConvolutionLayerTest, TestDtypesAndDevices);

TYPED_TEST(BinaryConvolutionLayerTest, TestSetup) {
  typedef typename TypeParam::Dtype Dtype;
  LayerParameter layer_param;
  ConvolutionParameter* convolution_param =
      layer_param.mutable_convolution_param();
	convolution_param->set_bias_term(false);
	convolution_param->add_kernel_size(3);
  convolution_param->add_stride(2);
  convolution_param->set_num_output(4);
  this->blob_bottom_vec_.push_back(this->blob_bottom_2_);
  this->blob_top_vec_.push_back(this->blob_top_2_);
  shared_ptr<Layer<Dtype> > layer(
      new BinaryConvolutionLayer<Dtype>(layer_param));
  layer->SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
  EXPECT_EQ(this->blob_top_->num(), 2);
  EXPECT_EQ(this->blob_top_->channels(), 4);
  EXPECT_EQ(this->blob_top_->height(), 2);
  EXPECT_EQ(this->blob_top_->width(), 1);
  EXPECT_EQ(this->blob_top_2_->num(), 2);
  EXPECT_EQ(this->blob_top_2_->channels(), 4);
  EXPECT_EQ(this->blob_top_2_->height(), 2);
  EXPECT_EQ(this->blob_top_2_->width(), 1);
  // setting group should not change the shape
  convolution_param->set_num_output(3);
  convolution_param->set_group(3);
  layer.reset(new BinaryConvolutionLayer<Dtype>(layer_param));
  layer->SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
  EXPECT_EQ(this->blob_top_->num(), 2);
  EXPECT_EQ(this->blob_top_->channels(), 3);
  EXPECT_EQ(this->blob_top_->height(), 2);
  EXPECT_EQ(this->blob_top_->width(), 1);
  EXPECT_EQ(this->blob_top_2_->num(), 2);
  EXPECT_EQ(this->blob_top_2_->channels(), 3);
  EXPECT_EQ(this->blob_top_2_->height(), 2);
  EXPECT_EQ(this->blob_top_2_->width(), 1);
}

TYPED_TEST(BinaryConvolutionLayerTest, TestSimpleBinaryConvolutionWithTestPhase) {
  typedef typename TypeParam::Dtype Dtype;
  this->blob_bottom_vec_.push_back(this->blob_bottom_2_);
  this->blob_top_vec_.push_back(this->blob_top_2_);
  LayerParameter layer_param;
	layer_param.set_phase(TEST);
  ConvolutionParameter* convolution_param =
      layer_param.mutable_convolution_param();
	convolution_param->set_bias_term(false);
  convolution_param->add_kernel_size(3);
  convolution_param->add_stride(2);
  convolution_param->set_num_output(4);
  convolution_param->mutable_weight_filler()->set_type("gaussian");
  convolution_param->mutable_bias_filler()->set_type("constant");
  convolution_param->mutable_bias_filler()->set_value(0.1);
  shared_ptr<Layer<Dtype> > layer(
      new BinaryConvolutionLayer<Dtype>(layer_param));
  layer->SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
	vector<shared_ptr<Blob<Dtype> > > layer_weights = layer->blobs();
	Blob<Dtype> weight_buffer;
	weight_buffer.CopyFrom(*layer_weights[0], false, true);

	layer->Forward(this->blob_bottom_vec_, this->blob_top_vec_);
  // Check against reference convolution.
  const Dtype* top_data;
  const Dtype* ref_top_data;
	caffe_bin_conv(this->blob_bottom_, convolution_param, &weight_buffer,
      this->MakeReferenceTop(this->blob_top_));
  top_data = this->blob_top_->cpu_data();
  ref_top_data = this->ref_blob_top_->cpu_data();
  for (int i = 0; i < this->blob_top_->count(); ++i) {
    EXPECT_NEAR(top_data[i], ref_top_data[i], 1e-4);
  }

	caffe_bin_conv(this->blob_bottom_2_, convolution_param, &weight_buffer,
      this->MakeReferenceTop(this->blob_top_2_));
  top_data = this->blob_top_2_->cpu_data();
  ref_top_data = this->ref_blob_top_->cpu_data();
  for (int i = 0; i < this->blob_top_->count(); ++i) {
    EXPECT_NEAR(top_data[i], ref_top_data[i], 1e-4);
  }
}

TYPED_TEST(BinaryConvolutionLayerTest, TestSimpleBinaryConvolutionWithTrainPhase) {
	typedef typename TypeParam::Dtype Dtype;
	this->blob_bottom_vec_.push_back(this->blob_bottom_2_);
	this->blob_top_vec_.push_back(this->blob_top_2_);
	LayerParameter layer_param;
	layer_param.set_phase(TRAIN);
	ConvolutionParameter* convolution_param =
		layer_param.mutable_convolution_param();
	convolution_param->set_bias_term(false);
	convolution_param->add_kernel_size(3);
	convolution_param->add_stride(2);
	convolution_param->set_num_output(4);
	convolution_param->mutable_weight_filler()->set_type("gaussian");
	convolution_param->mutable_bias_filler()->set_type("constant");
	convolution_param->mutable_bias_filler()->set_value(0.1);
	shared_ptr<Layer<Dtype> > layer(
		new BinaryConvolutionLayer<Dtype>(layer_param));
	layer->SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
	vector<shared_ptr<Blob<Dtype> > > layer_weights = layer->blobs();
	Blob<Dtype> weight_buffer;
	weight_buffer.CopyFrom(*layer_weights[0], false, true);

	layer->Forward(this->blob_bottom_vec_, this->blob_top_vec_);
	// Check against reference convolution.
	const Dtype* top_data;
	const Dtype* ref_top_data;
	caffe_bin_conv(this->blob_bottom_, convolution_param, &weight_buffer,
		this->MakeReferenceTop(this->blob_top_));
	top_data = this->blob_top_->cpu_data();
	ref_top_data = this->ref_blob_top_->cpu_data();
	for (int i = 0; i < this->blob_top_->count(); ++i) {
		EXPECT_NEAR(top_data[i], ref_top_data[i], 1e-4);
	}

	caffe_bin_conv(this->blob_bottom_2_, convolution_param, &weight_buffer,
		this->MakeReferenceTop(this->blob_top_2_));
	top_data = this->blob_top_2_->cpu_data();
	ref_top_data = this->ref_blob_top_->cpu_data();
	for (int i = 0; i < this->blob_top_->count(); ++i) {
		EXPECT_NEAR(top_data[i], ref_top_data[i], 1e-4);
	}
}

TYPED_TEST(BinaryConvolutionLayerTest, TestGradient) {
  typedef typename TypeParam::Dtype Dtype;
  LayerParameter layer_param;
  ConvolutionParameter* convolution_param =
      layer_param.mutable_convolution_param();
  this->blob_bottom_vec_.push_back(this->blob_bottom_2_);
  this->blob_top_vec_.push_back(this->blob_top_2_);
	convolution_param->set_bias_term(false);
  convolution_param->add_kernel_size(3);
  convolution_param->add_stride(2);
  convolution_param->set_num_output(2);
  convolution_param->mutable_weight_filler()->set_type("gaussian");
  convolution_param->mutable_bias_filler()->set_type("gaussian");
	BinaryConvolutionLayer<Dtype> layer(layer_param);
  GradientChecker<Dtype> checker(1e-2, 1e-3);
  checker.CheckGradientExhaustive(&layer, this->blob_bottom_vec_,
      this->blob_top_vec_);
}

}  // namespace caffe
