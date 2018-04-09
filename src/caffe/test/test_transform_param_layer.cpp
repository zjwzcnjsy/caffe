#include <algorithm>
#include <vector>

#include "gtest/gtest.h"

#include "caffe/blob.hpp"
#include "caffe/common.hpp"
#include "caffe/filler.hpp"
#include "caffe/layers/transform_param_layer.hpp"

#include "caffe/test/test_caffe_main.hpp"
#include "caffe/test/test_gradient_check_util.hpp"

namespace caffe {
template <typename TypeParam>
class TransformParamLayerTest : public MultiDeviceTest<TypeParam> {
	typedef typename TypeParam::Dtype Dtype;

protected:
	TransformParamLayerTest()
	: blob_bottom_a_(new Blob<Dtype>()),
		blob_bottom_b_(new Blob<Dtype>()),
		blob_top_(new Blob<Dtype>()) {}
	virtual void SetUp() {
		Caffe::set_random_seed(1701);
		vector<int> pred_shape(2);
		pred_shape[0] = 2;
		pred_shape[1] = 136;
		blob_bottom_a_->Reshape(pred_shape);
		pred_shape[0] = 1;
		blob_bottom_b_->Reshape(pred_shape);
		// fill the values
		FillerParameter filler_param;
		GaussianFiller<Dtype> filler(filler_param);
		filler.Fill(this->blob_bottom_a_);
		blob_bottom_vec_.push_back(blob_bottom_a_);
		filler.Fill(this->blob_bottom_b_);
		blob_bottom_vec_.push_back(blob_bottom_b_);
		blob_top_vec_.push_back(blob_top_);
	}

	virtual ~TransformParamLayerTest() {
		delete blob_bottom_a_;
		delete blob_bottom_b_;
		delete blob_top_;
	}

	Blob<Dtype>* const blob_bottom_a_;
	Blob<Dtype>* const blob_bottom_b_;
	Blob<Dtype>* const blob_top_;
	vector<Blob<Dtype>*> blob_bottom_vec_;
	vector<Blob<Dtype>*> blob_top_vec_;
};

TYPED_TEST_CASE(TransformParamLayerTest, TestDtypesAndDevices);

TYPED_TEST(TransformParamLayerTest, TestSetUp) {
	typedef typename TypeParam::Dtype Dtype;
	LayerParameter layer_param;
	TransformParamLayer<Dtype> layer(layer_param);
	layer.SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
	EXPECT_EQ(this->blob_top_->num(), 2);
	EXPECT_EQ(this->blob_top_->channels(), 6);
}

TYPED_TEST(TransformParamLayerTest, TestForward) {
	typedef typename TypeParam::Dtype Dtype;
	LayerParameter layer_param;
	TransformParamLayer<Dtype> layer(layer_param);
	layer.SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
	layer.Forward(this->blob_bottom_vec_, this->blob_top_vec_);
	const int num = this->blob_bottom_a_->num();
	const int landmark_dim = this->blob_bottom_a_->channels();
	const Dtype* data = this->blob_top_->cpu_data();
	const Dtype* in_data_a = this->blob_bottom_a_->cpu_data();
	const Dtype* in_data_b = this->blob_bottom_b_->cpu_data();
	Dtype T[6];
	for (int i = 0; i < num; ++i) {
		bestFit<Dtype>(landmark_dim / 2, in_data_b, in_data_a + i*landmark_dim, NULL, T);
		for (int j = 0; j < 6; ++j) {
			EXPECT_NEAR(data[i*6 + j], T[j], 1e-4);
		}
	}
}
}