#include <algorithm>
#include <vector>

#include "caffe/layers/binary_active_layer.hpp"

namespace caffe {

template <typename Dtype>
void BinaryActiveLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  const Dtype* bottom_data = bottom[0]->cpu_data();
  Dtype* top_data = top[0]->mutable_cpu_data();
  const int count = bottom[0]->count();
  for (int i = 0; i < count; ++i) {
    top_data[i] = sign(bottom_data[i]);
  }
}

template <typename Dtype>
void BinaryActiveLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down,
    const vector<Blob<Dtype>*>& bottom) {
  if (propagate_down[0]) {
    const Dtype* bottom_data = bottom[0]->cpu_data();
    const Dtype* top_diff = top[0]->cpu_diff();
    Dtype* bottom_diff = bottom[0]->mutable_cpu_diff();
    const int count = bottom[0]->count();
    for (int i = 0; i < count; ++i) {
      bottom_diff[i] = top_diff[i] * (bottom_data[i] > Dtype(-1.) && bottom_data[i] < Dtype(1.));
    }
  }
}


#ifdef CPU_ONLY
STUB_GPU(BinaryActiveLayer);
#endif

INSTANTIATE_CLASS(BinaryActiveLayer);
REGISTER_LAYER_CLASS(BinaryActive);

}  // namespace caffe
