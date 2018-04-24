#include <vector>

#include "caffe/blob.hpp"
#include "caffe/common.hpp"
#include "caffe/filler.hpp"
#include "caffe/layers/binary_conv_layer2.hpp"
using namespace caffe;

#include <iostream>

template <typename Dtype>
class BinaryConvolutionV2LayerProfile {
public:
	BinaryConvolutionV2LayerProfile()
		: blob_bottom_(new Blob<Dtype>(1, 64, 1, 1)),
		blob_top_(new Blob<Dtype>()) {}

	virtual void SetUp() {
		// fill the values
		FillerParameter filler_param;
		filler_param.set_value(1.);
		GaussianFiller<Dtype> filler(filler_param);
		filler.Fill(this->blob_bottom_);
		caffe_cpu_sign(this->blob_bottom_->count(), this->blob_bottom_->cpu_data(), this->blob_bottom_->mutable_cpu_data());
		const Dtype* data = this->blob_bottom_->cpu_data();
		std::cout << "data:";
		for (int i = 0; i < this->blob_bottom_->count(); ++i) {
			std::cout << ", " << data[i];
		}
		std::cout << std::endl;
		blob_bottom_vec_.push_back(blob_bottom_);
		blob_top_vec_.push_back(blob_top_);
	}

	virtual ~BinaryConvolutionV2LayerProfile() {
		delete blob_bottom_;
		delete blob_top_;
	}

private:
	Blob<Dtype>* const blob_bottom_;
	Blob<Dtype>* const blob_top_;
	vector<Blob<Dtype>*> blob_bottom_vec_;
	vector<Blob<Dtype>*> blob_top_vec_;

public:
	void ProfileSimpleBinaryConvolutionWithTestPhase();
	void ProfileSimpleBinaryConvolutionWithTrainPhase();
};

template <typename Dtype>
void BinaryConvolutionV2LayerProfile<Dtype>::ProfileSimpleBinaryConvolutionWithTestPhase() {
	LayerParameter layer_param;
	layer_param.set_phase(TEST);
	BinaryConvolutionParameter* convolution_param =
		layer_param.mutable_binary_convolution_param();
	convolution_param->set_bias_term(false);
	convolution_param->add_kernel_size(6);
	convolution_param->add_stride(1);
	convolution_param->add_pad(0);
	convolution_param->set_num_output(4096);
	convolution_param->mutable_weight_filler()->set_type("gaussian");
	convolution_param->mutable_bias_filler()->set_type("constant");
	convolution_param->mutable_bias_filler()->set_value(0.1);
	shared_ptr<Layer<Dtype> > layer(
		new BinaryConvolutionV2Layer<Dtype>(layer_param));
	layer->SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
	layer->Forward(this->blob_bottom_vec_, this->blob_top_vec_);
	layer->Forward(this->blob_bottom_vec_, this->blob_top_vec_);
	vector<bool> propagate_down(1, true);
	layer->Backward(this->blob_top_vec_, propagate_down, this->blob_bottom_vec_);
	layer->Backward(this->blob_top_vec_, propagate_down, this->blob_bottom_vec_);
}

template <typename Dtype>
void BinaryConvolutionV2LayerProfile<Dtype>::ProfileSimpleBinaryConvolutionWithTrainPhase() {
	LayerParameter layer_param;
	layer_param.set_phase(TRAIN);
	BinaryConvolutionParameter* convolution_param =
		layer_param.mutable_binary_convolution_param();
	convolution_param->set_bias_term(false);
	convolution_param->add_kernel_size(1);
	convolution_param->add_stride(1);
	convolution_param->add_pad(0);
	convolution_param->set_num_output(2);
	convolution_param->mutable_weight_filler()->set_type("gaussian");
	convolution_param->mutable_bias_filler()->set_type("constant");
	convolution_param->mutable_bias_filler()->set_value(0.1);
	shared_ptr<Layer<Dtype> > layer(
		new BinaryConvolutionV2Layer<Dtype>(layer_param));
	layer->SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
	layer->Forward(this->blob_bottom_vec_, this->blob_top_vec_);
	//layer->Forward(this->blob_bottom_vec_, this->blob_top_vec_);
	//vector<bool> propagate_down(1, true);
	//layer->Backward(this->blob_top_vec_, propagate_down, this->blob_bottom_vec_);
	//layer->Backward(this->blob_top_vec_, propagate_down, this->blob_bottom_vec_);
}


int main(int argc, char *argv[]) {
	Caffe::SetDevice(0);
	Caffe::set_mode(Caffe::GPU);
	BinaryConvolutionV2LayerProfile<float> profiler;
	profiler.SetUp();
	profiler.ProfileSimpleBinaryConvolutionWithTrainPhase();
}