#ifndef CAFFE_MULTI_LABEL_BASE_DATA_LAYERS_HPP_
#define CAFFE_MULTI_LABEL_BASE_DATA_LAYERS_HPP_

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
class MultiLabelBatch {
 public:
  Blob<Dtype> data_, label_;
};

template <typename Dtype>
class MultiLabelBasePrefetchingDataLayer :
    public BaseDataLayer<Dtype>, public InternalThread {
 public:
  explicit MultiLabelBasePrefetchingDataLayer(const LayerParameter& param);
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
  virtual void load_batch(MultiLabelBatch<Dtype>* batch) = 0;

  vector<shared_ptr<MultiLabelBatch<Dtype> > > prefetch_;
  BlockingQueue<MultiLabelBatch<Dtype>*> prefetch_free_;
  BlockingQueue<MultiLabelBatch<Dtype>*> prefetch_full_;
  MultiLabelBatch<Dtype>* prefetch_current_;

  Blob<Dtype> transformed_data_;
};

}  // namespace caffe

#endif  // CAFFE_MULTI_LABEL_BASE_DATA_LAYERS_HPP_
