#include <vector>

#include "caffe/layers/landmark_image_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

template <typename Dtype>
Dtype clip(const Dtype x, const Dtype min_v, const Dtype max_v) {
	if (x < min_v) {
		return min_v;
	}
	else if (x > max_v) {
		return max_v;
	}
	else {
		return x;
	}
}

template <typename Dtype>
void gen_landmark_image(const int num_landmark, const Dtype* landmark_data,
	const int img_height, const int img_width, const int landmark_patch_size, Dtype *tmp_images, Dtype* out_img) {
	int half_size = landmark_patch_size / 2;
	Dtype clip_landmark_x, clip_landmark_y;
	for (int i = 0; i < num_landmark; ++i) {
		clip_landmark_x = clip(landmark_data[2 * i + 0], static_cast<Dtype>(half_size), static_cast<Dtype>(img_width - 1 - half_size));
		clip_landmark_y = clip(landmark_data[2 * i + 1], static_cast<Dtype>(half_size), static_cast<Dtype>(img_height - 1 - half_size));
		int intLandmark_x = static_cast<int>(clip_landmark_x);
		int intLandmark_y = static_cast<int>(clip_landmark_y);
		Dtype dx = clip_landmark_x - intLandmark_x;
		Dtype dy = clip_landmark_y - intLandmark_y;
		for (int ih = 0; ih <= 2 * half_size; ++ih) {
			for (int jh = 0; jh <= 2 * half_size; ++jh) {
				int offset_x = -half_size + ih;
				int offset_y = -half_size + jh;
				int location_x = offset_x + intLandmark_x;
				int location_y = offset_y + intLandmark_y;
				Dtype offsetsSubPix_x = static_cast<Dtype>(offset_x) - dx;
				Dtype offsetsSubPix_y = static_cast<Dtype>(offset_y) - dy;
				tmp_images[i*img_height*img_width + location_y*img_width + location_x] =
					Dtype(1.0) / (Dtype(1.0) + static_cast<Dtype>(sqrt(static_cast<double>(offsetsSubPix_x*offsetsSubPix_x + offsetsSubPix_y*offsetsSubPix_y) + 1e-6)));
			}
		}
	}
	for (int i = 0; i < num_landmark; ++i) {
		for (int j = 0; j < img_height; ++j) {
			for (int k = 0; k < img_width; ++k) {
				Dtype t = tmp_images[i*img_height*img_width + j*img_width + k];
				if (t >= out_img[j*img_width + k]) {
					out_img[j*img_width + k] = t;
				}
			}
		}
	}
}

template <typename Dtype>
void LandmarkImageLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  num_landmark_ = bottom[0]->channels() / 2;
  const LandmarkImageParameter& landmark_image_param = this->layer_param_.landmark_image_param();
  img_height_ = landmark_image_param.image_height();
  img_width_ = landmark_image_param.image_width();
  landmark_patch_size_ = landmark_image_param.landmark_patch_size();

  vector<int> landmark_images_shape(4);
  landmark_images_shape[0] = 1;
  landmark_images_shape[1] = num_landmark_;
  landmark_images_shape[2] = img_height_;
  landmark_images_shape[3] = img_width_;
  tmp_landmark_images_.Reshape(landmark_images_shape);
}

template <typename Dtype>
void LandmarkImageLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
	const vector<Blob<Dtype>*>& top) {
	vector<int> top_shape(4);
	top_shape[0] = bottom[0]->num();
	top_shape[1] = 1;
	top_shape[2] = img_height_;
	top_shape[3] = img_width_;
	top[0]->Reshape(top_shape);
}

template <typename Dtype>
void LandmarkImageLayer<Dtype>::Forward_cpu(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  const int spatial_dim = img_width_ * img_height_;
  const int landmark_dim = bottom[0]->count(1);
  const Dtype* bottom_data = bottom[0]->cpu_data();
  Dtype* top_data = top[0]->mutable_cpu_data();
  caffe_set(top[0]->count(), Dtype(0), top_data);
  for (int i = 0; i < bottom[0]->num(); ++i) {
	gen_landmark_image<Dtype>(num_landmark_, bottom_data, img_height_, img_width_, landmark_patch_size_, 
		tmp_landmark_images_.mutable_cpu_data(), top_data);
	bottom_data += landmark_dim;
	top_data += spatial_dim;
  }
}

template <typename Dtype>
void LandmarkImageLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
  if (propagate_down[0]) {
	  // nothing to do
  }
}

#ifdef CPU_ONLY
STUB_GPU(LandmarkImageLayer);
#endif

INSTANTIATE_CLASS(LandmarkImageLayer);
REGISTER_LAYER_CLASS(LandmarkImage);

}  // namespace caffe
