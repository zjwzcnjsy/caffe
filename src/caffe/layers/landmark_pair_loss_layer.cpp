#include <vector>

#include "caffe/layers/landmark_pair_loss_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

	template <typename Dtype>
	Dtype calc_eye_dist_68(const Dtype*gtLandmark) {
		Dtype left_eye_x = 0, left_eye_y = 0;
		Dtype right_eye_x = 0, right_eye_y = 0;
		for (int i = 36; i < 42; ++i) {
			left_eye_x += gtLandmark[2 * i + 0];
			left_eye_y += gtLandmark[2 * i + 1];
		}
		left_eye_x /= 6;
		left_eye_y /= 6;
		for (int i = 42; i < 48; ++i) {
			right_eye_x += gtLandmark[2 * i + 0];
			right_eye_y += gtLandmark[2 * i + 1];
		}
		right_eye_x /= 6;
		right_eye_y /= 6;
		Dtype dx = left_eye_x - right_eye_x;
		Dtype dy = left_eye_y - right_eye_y;
		return sqrt(dx*dx + dy*dy);
	}

template <typename Dtype>
void LandmarkPairLossLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  LossLayer<Dtype>::LayerSetUp(bottom, top);
  CHECK_EQ(bottom[0]->num(), bottom[1]->num()) << "the number of shape miss match.";
  CHECK_EQ(bottom[0]->channels(), bottom[1]->channels()) << "the number of landmark miss match.";
  num_landmark_ = bottom[0]->channels() / 2;
  tmp_diff.ReshapeLike(*bottom[0]);
  vector<int> dist_shape(2);
  dist_shape[0] = bottom[0]->num();
  dist_shape[1] = num_landmark_;
  tmp_dist.Reshape(dist_shape);
  vector<int> eye_dist_shape(1, bottom[0]->num());
  tmp_eye_dist.Reshape(eye_dist_shape);
}

template <typename Dtype>
void LandmarkPairLossLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
	const vector<Blob<Dtype>*>& top) {
	LossLayer<Dtype>::Reshape(bottom, top);
}

template <typename Dtype>
void LandmarkPairLossLayer<Dtype>::Forward_cpu(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
	const int num = bottom[0]->num();
	const int landmark_dim = 2 * num_landmark_;
	const Dtype *pred_shape_data = bottom[0]->cpu_data();
	const Dtype *gt_shape_data = bottom[1]->cpu_data();
	Dtype *tmp_dist_data = tmp_dist.mutable_cpu_data();
	Dtype *tmp_eye_dist_data = tmp_eye_dist.mutable_cpu_data();
	caffe_sub(bottom[0]->count(), pred_shape_data, gt_shape_data, tmp_diff.mutable_cpu_data());
	caffe_mul(tmp_diff.count(), tmp_diff.cpu_data(), tmp_diff.cpu_data(), tmp_diff.mutable_cpu_diff());
	const Dtype *tmp_diff_squre_data = tmp_diff.cpu_diff();
	Dtype error = Dtype(0.0);
	for (int i = 0; i < bottom[0]->num(); ++i) {
		Dtype sum_dist = Dtype(0.0);
		for (int j = 0; j < num_landmark_; ++j) {
			Dtype dx_squre = tmp_diff_squre_data[2 * j];
			Dtype dy_squre = tmp_diff_squre_data[2 * j + 1];
			tmp_dist_data[i*num_landmark_ + j] = sqrt(dx_squre + dy_squre);
			sum_dist += tmp_dist_data[i*num_landmark_ + j];
		}
		sum_dist /= static_cast<Dtype>(num_landmark_);
		tmp_eye_dist_data[i] = calc_eye_dist_68(gt_shape_data);
		error += sum_dist / tmp_eye_dist_data[i];
		tmp_diff_squre_data += landmark_dim;
		pred_shape_data += landmark_dim;
		gt_shape_data += landmark_dim;
	}
	top[0]->mutable_cpu_data()[0] = error / static_cast<Dtype>(num);
}

template <typename Dtype>
void LandmarkPairLossLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
  if (propagate_down[0]) {
	  const int num = bottom[0]->num();
	  const int landmark_dim = 2 * num_landmark_;
	  const Dtype *tmp_diff_data = tmp_diff.cpu_data();
	  Dtype *tmp_dist_data = tmp_dist.mutable_cpu_data();
	  Dtype *tmp_eye_dist_data = tmp_eye_dist.mutable_cpu_data();
	  const Dtype *top_diff = top[0]->cpu_diff();
	  Dtype *bottom_diff = bottom[0]->mutable_cpu_diff();
	  for (int i = 0; i < bottom[0]->num(); ++i) {
		  Dtype alpha = top_diff[0] / (num*tmp_eye_dist_data[i] * num_landmark_);
		  caffe_div(num_landmark_, tmp_diff_data, tmp_dist_data, bottom_diff);
		  caffe_div(num_landmark_, tmp_diff_data + num_landmark_, tmp_dist_data, bottom_diff + num_landmark_);
		  caffe_scal(landmark_dim, alpha, bottom_diff);
		  tmp_diff_data += landmark_dim;
		  tmp_dist_data += num_landmark_;
		  bottom_diff += landmark_dim;
	  }
  }
}

#ifdef CPU_ONLY
STUB_GPU(LandmarkPairLossLayer);
#endif

INSTANTIATE_CLASS(LandmarkPairLossLayer);
REGISTER_LAYER_CLASS(LandmarkPairLoss);

}  // namespace caffe
