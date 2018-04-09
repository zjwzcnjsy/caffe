#include <vector>

#include "caffe/layers/affine_transform_layer.hpp"
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
void affine_transform(const int height, const int width, const Dtype* img, 
	const int out_height, const int out_width, Dtype* out_img, Dtype *T) {
	double T2[6];
	for (size_t i = 0; i < 6; i++)
	{
		T2[i] = static_cast<double>(T[i]);
	}
	caffe_2p2_matrix_inv<double>(T2, T2);
	double *A = T2;
	double *t = T2 + 4;
	double t0 = t[0];
	double t1 = t[1];
	t[0] = -(t0 * A[0] + t1 * A[2]);
	t[1] = -(t0 * A[1] + t1 * A[3]);

	double *pixels = new double[out_height*out_width * 2];
	double *outPixels = new double[out_height*out_width * 2];
	int idxCount = 0;
	for (int i = 0; i < out_width; ++i) {
		for (int j = 0; j < out_height; ++j) {
			pixels[idxCount] = static_cast<double>(i);
			pixels[idxCount + 1] = static_cast<double>(j);
			outPixels[idxCount] = t[0];
			outPixels[idxCount + 1] = t[1];
			idxCount += 2;
		}
	}
	caffe_cpu_gemm<double>(CblasNoTrans, CblasNoTrans,
		out_height*out_width, 2, 2, 1.,
		pixels, A, 1., outPixels);
	idxCount = 0;
	int outPixelsMinMin_x, outPixelsMinMin_y;
	int outPixelsMaxMin_x, outPixelsMaxMin_y;
	int outPixelsMinMax_x, outPixelsMinMax_y;
	int outPixelsMaxMax_x, outPixelsMaxMax_y;
	double dx, dy;
	for (int i = 0; i < out_width; ++i) {
		for (int j = 0; j < out_height; ++j) {
			outPixels[idxCount] = clip<double>(outPixels[idxCount], 0, width - 2);
			outPixels[idxCount + 1] = clip<double>(outPixels[idxCount + 1], 0, height - 2);
			
			outPixelsMinMin_x = static_cast<int>(outPixels[idxCount]);
			outPixelsMinMin_y = static_cast<int>(outPixels[idxCount + 1]);

			outPixelsMaxMin_x = outPixelsMinMin_x + 1;
			outPixelsMaxMin_y = outPixelsMinMin_y + 0;

			outPixelsMinMax_x = outPixelsMinMin_x + 0;
			outPixelsMinMax_y = outPixelsMinMin_y + 1;

			outPixelsMaxMax_x = outPixelsMinMin_x + 1;
			outPixelsMaxMax_y = outPixelsMinMin_y + 1;
			dx = outPixels[idxCount] - outPixelsMinMin_x;
			dy = outPixels[idxCount + 1] - outPixelsMinMin_y;

			int out_x = static_cast<int>(pixels[idxCount]);
			int out_y = static_cast<int>(pixels[idxCount + 1]);
			out_img[out_y*out_width + out_x] += (1 - dx)*(1 - dy)*img[outPixelsMinMin_y*width + outPixelsMinMin_x];
			out_img[out_y*out_width + out_x] += dx * (1 - dy) * img[outPixelsMaxMin_y*width + outPixelsMaxMin_x];
			out_img[out_y*out_width + out_x] += (1 - dx) * dy * img[outPixelsMinMax_y*width + outPixelsMinMax_x];
			out_img[out_y*out_width + out_x] += dx * dy * img[outPixelsMaxMax_y*width + outPixelsMaxMax_x];
			idxCount += 2;
		}
	}
	delete[]pixels;
	delete[]outPixels;
}

template <typename Dtype>
void AffineTransformLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  CHECK_EQ(bottom[0]->num(), bottom[1]->num()) << "the number of image miss match.";
  CHECK_EQ(bottom[1]->channels(), 6) << "transform param must have 6";
  num_ = bottom[0]->num();
  channels_ = bottom[0]->channels();
  height_ = bottom[0]->height();
  width_ = bottom[0]->width();
  // now only support same size affine transform
  out_img_height_ = height_;
  out_img_width_ = width_;
  tmp_transform_param_.ReshapeLike(*bottom[1]);
}

template <typename Dtype>
void AffineTransformLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
	const vector<Blob<Dtype>*>& top) {
	vector<int> top_shape = bottom[0]->shape();
	top[0]->Reshape(top_shape);
}

template <typename Dtype>
void AffineTransformLayer<Dtype>::Forward_cpu(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  const int spatial_dim = channels_ * height_ * width_;
  const Dtype* bottom_data = bottom[0]->cpu_data();
  caffe_copy(bottom[1]->count(), bottom[1]->cpu_data(), tmp_transform_param_.mutable_cpu_data());
  Dtype* transform_param_data = tmp_transform_param_.mutable_cpu_data();
  caffe_set(top[0]->count(), Dtype(0), top[0]->mutable_cpu_data());
  Dtype* top_data = top[0]->mutable_cpu_data();
  for (int i = 0; i < num_; ++i) {
		affine_transform<Dtype>(height_, width_, bottom_data, out_img_height_, out_img_width_, top_data, transform_param_data);
		transform_param_data += 6;
		bottom_data += spatial_dim;
		top_data += spatial_dim;
  }
}

template <typename Dtype>
void AffineTransformLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
  if (propagate_down[0]) {
	  // nothing to do
  }
}

#ifdef CPU_ONLY
STUB_GPU(AffineTransformLayer);
#endif

INSTANTIATE_CLASS(AffineTransformLayer);
REGISTER_LAYER_CLASS(AffineTransform);

}  // namespace caffe
