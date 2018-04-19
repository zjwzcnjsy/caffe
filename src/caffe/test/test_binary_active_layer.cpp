#include <algorithm>
#include <vector>

#include "gtest/gtest.h"

#include "caffe/blob.hpp"
#include "caffe/common.hpp"
#include "caffe/filler.hpp"
#include "caffe/layers/binary_active_layer.hpp"

#include "caffe/test/test_caffe_main.hpp"
#include "caffe/test/test_gradient_check_util.hpp"

namespace caffe {

#define sign(x) ((x)>=0?1:-1)

template <typename TypeParam>
class BinaryActiveLayerTest : public MultiDeviceTest<TypeParam> {
  typedef typename TypeParam::Dtype Dtype;

 protected:
  BinaryActiveLayerTest()
      : blob_bottom_(new Blob<Dtype>(2, 3, 4, 5)),
        blob_top_(new Blob<Dtype>()) {
    Caffe::set_random_seed(1701);
    // fill the values
    FillerParameter filler_param;
    GaussianFiller<Dtype> filler(filler_param);
    filler.Fill(this->blob_bottom_);
    blob_bottom_vec_.push_back(blob_bottom_);
    blob_top_vec_.push_back(blob_top_);
  }
  virtual ~BinaryActiveLayerTest() { delete blob_bottom_; delete blob_top_; }

  void TestForward() {
    LayerParameter layer_param;
    BinaryActiveLayer<Dtype> layer(layer_param);
    layer.SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
    layer.Forward(this->blob_bottom_vec_, this->blob_top_vec_);
    // Now, check values
    const Dtype* bottom_data = this->blob_bottom_->cpu_data();
    const Dtype* top_data = this->blob_top_->cpu_data();
    for (int i = 0; i < this->blob_bottom_->count(); ++i) {
      EXPECT_NEAR(sign(bottom_data[i]), top_data[i], 1e-4);
    }
  }

  void TestBackward() {
    LayerParameter layer_param;
		BinaryActiveLayer<Dtype> layer(layer_param);
    GradientChecker<Dtype> checker(1e-3, 1, 1701, 0., 0.01);
    checker.CheckGradientEltwise(&layer, this->blob_bottom_vec_,
        this->blob_top_vec_);
  }

  Blob<Dtype>* const blob_bottom_;
  Blob<Dtype>* const blob_top_;
  vector<Blob<Dtype>*> blob_bottom_vec_;
  vector<Blob<Dtype>*> blob_top_vec_;
};

TYPED_TEST_CASE(BinaryActiveLayerTest, TestDtypesAndDevices);

TYPED_TEST(BinaryActiveLayerTest, TestForward) {
  typedef typename TypeParam::Dtype Dtype;
  this->TestForward();
}

TYPED_TEST(BinaryActiveLayerTest, TestBackward) {
  typedef typename TypeParam::Dtype Dtype;
  this->TestBackward();
}

}  // namespace caffe
