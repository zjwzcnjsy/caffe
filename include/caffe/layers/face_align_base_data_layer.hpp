#ifndef CAFFE_FACE_ALIGN_BASE_DATA_LAYERS_HPP_
#define CAFFE_FACE_ALIGN_BASE_DATA_LAYERS_HPP_

#include <vector>

#include "caffe/blob.hpp"
#include "caffe/data_transformer.hpp"
#include "caffe/internal_thread.hpp"
#include "caffe/layer.hpp"
#include "caffe/proto/caffe.pb.h"
#include "caffe/util/blocking_queue.hpp"
#include "caffe/layers/base_data_layer.hpp"

namespace caffe {

template <typename Dtype>
class FaceAlignBatch {
 public:
  Blob<Dtype> data_, label_, pose_;
  bool has_pose_;
};

template <typename Dtype>
class FaceAlignBasePrefetchingDataLayer :
    public BaseDataLayer<Dtype>, public InternalThread {
 public:
  explicit FaceAlignBasePrefetchingDataLayer(const LayerParameter& param);
  // LayerSetUp: implements common data layer setup functionality, and calls
  // DataLayerSetUp to do special data layer setup for individual layer types.
  // This method may not be overridden.
  void LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);

  virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Forward_gpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);

 protected:
  virtual void InternalThreadEntry();
  virtual void load_batch(FaceAlignBatch<Dtype>* batch) = 0;

  vector<shared_ptr<FaceAlignBatch<Dtype> > > prefetch_;
  BlockingQueue<FaceAlignBatch<Dtype>*> prefetch_free_;
  BlockingQueue<FaceAlignBatch<Dtype>*> prefetch_full_;
  FaceAlignBatch<Dtype>* prefetch_current_;

  Blob<Dtype> transformed_data_;
};

}  // namespace caffe

#endif  // CAFFE_FACE_ALIGN_BASE_DATA_LAYERS_HPP_
