#ifndef CAFFE_MULTI_LABEL_DATA_LAYER_HPP_
#define CAFFE_MULTI_LABEL_DATA_LAYER_HPP_

#include <vector>

#include "caffe/blob.hpp"
#include "caffe/data_transformer.hpp"
#include "caffe/internal_thread.hpp"
#include "caffe/layer.hpp"
#include "caffe/layers/multi_label_base_data_layer.hpp"
#include "caffe/proto/caffe.pb.h"
#include "caffe/util/db.hpp"

namespace caffe {

template <typename Dtype>
class MultiLabelDataLayer : public MultiLabelBasePrefetchingDataLayer<Dtype> {
 public:
  explicit MultiLabelDataLayer(const LayerParameter& param);
  virtual ~MultiLabelDataLayer();
  virtual void MultiLabelDataLayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual inline const char* type() const { return "MultiLabelData"; }
  virtual inline int ExactNumBottomBlobs() const { return 0; }
  virtual inline int MinTopBlobs() const { return 1; }
  virtual inline int MaxTopBlobs() const { return 2; }

 protected:
  void Next();
  bool Skip();
  virtual void load_batch(MultiLabelBatch<Dtype>* batch);

  shared_ptr<db::DB> db_;
  shared_ptr<db::Cursor> cursor_;
  uint64_t offset_;
};

}  // namespace caffe

#endif  // CAFFE_MULTI_LABEL_DATA_LAYER_HPP_
