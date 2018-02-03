#include <vector>

#include "caffe/blob.hpp"
#include "caffe/common.hpp"
#include "caffe/filler.hpp"
#include "caffe/layer.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/layers/additive_angular_margin_layer.hpp"

namespace caffe {

	template <typename Dtype>
	void AdditiveAngularMarginLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
		const vector<Blob<Dtype>*>& top) {
		const AdditiveAngularMarginParameter& param = this->layer_param_.additive_angular_margin_param();
		margin_ = param.margin();
		cos_margin_ = cosf(margin_);
		sin_margin_ = sinf(margin_);
	}

	template <typename Dtype>
	void AdditiveAngularMarginLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
		const vector<Blob<Dtype>*>& top) {
		if (top[0] != bottom[0]) top[0]->ReshapeLike(*bottom[0]);
	}

	template <typename Dtype>
	void AdditiveAngularMarginLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
		const vector<Blob<Dtype>*>& top) {
		const Dtype* bottom_data = bottom[0]->cpu_data();
		const Dtype* label_data = bottom[1]->cpu_data();
		Dtype* top_data = top[0]->mutable_cpu_data();

		int num = bottom[0]->num();
		int count = bottom[0]->count();
		int dim = count / num;

		if (top[0] != bottom[0]) caffe_copy(count, bottom_data, top_data);

		for (int i = 0; i < num; ++i) {
			int gt = static_cast<int>(label_data[i]);
			float cos_theta = top_data[i * dim + gt];
			top_data[i * dim + gt] = cos_theta * cos_margin_ - sqrtf(1- cos_theta*cos_theta)*sin_margin_;
		}
	}

	template <typename Dtype>
	void AdditiveAngularMarginLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
		const vector<bool>& propagate_down,
		const vector<Blob<Dtype>*>& bottom) {

		if (top[0] != bottom[0] && propagate_down[0]) {
			const Dtype* top_diff = top[0]->cpu_diff();
			Dtype* bottom_diff = bottom[0]->mutable_cpu_diff();
			const Dtype* top_data = top[0]->cpu_data();
			const Dtype* label_data = bottom[1]->cpu_data();

			int num = bottom[0]->num();
			int count = bottom[0]->count();
			int dim = count / num;
			caffe_copy(count, top_diff, bottom_diff);

			for (int i = 0; i < num; ++i) {
				int gt = static_cast<int>(label_data[i]);
				float cos_theta = top_data[i * dim + gt];
				float sin_theta = sqrtf(1 - cos_theta*cos_theta);
				bottom_diff[i * dim + gt] *= cos_margin_ + cos_theta*sin_margin_ / sin_theta;
			}
		}
	}

#ifdef CPU_ONLY
	STUB_GPU(AdditiveAngularMarginLayer);
#endif

	INSTANTIATE_CLASS(AdditiveAngularMarginLayer);
	REGISTER_LAYER_CLASS(AdditiveAngularMargin);

}  // namespace caffe
