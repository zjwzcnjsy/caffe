#include <vector>

#include "caffe/layers/transform_param_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

template <typename Dtype>
void bestFit(const int num_landmark, const Dtype* dest, const Dtype* src, Dtype *transformed_src, Dtype *T) {
	Dtype dstMean_x = 0, dstMean_y = 0, srcMean_x = 0, srcMean_y = 0;
	for (size_t i = 0; i < num_landmark; i++)
	{
		dstMean_x += dest[2 * i + 0];
		dstMean_y += dest[2 * i + 1];
		srcMean_x += src[2 * i + 0];
		srcMean_y += src[2 * i + 1];
	}
	dstMean_x /= static_cast<Dtype>(num_landmark);
	dstMean_y /= static_cast<Dtype>(num_landmark);
	srcMean_x /= static_cast<Dtype>(num_landmark);
	srcMean_y /= static_cast<Dtype>(num_landmark);

	vector<Dtype> srcVec(2 * num_landmark);
	vector<Dtype> dstVec(2 * num_landmark);
	for (size_t i = 0; i < num_landmark; i++)
	{
		srcVec[2 * i + 0] = src[2 * i + 0] - srcMean_x;
		srcVec[2 * i + 1] = src[2 * i + 1] - srcMean_y;
		dstVec[2 * i + 0] = dest[2 * i + 0] - dstMean_x;
		dstVec[2 * i + 1] = dest[2 * i + 1] - dstMean_y;
	}
	Dtype srcVecL2Norm = caffe_l2_norm(2 * num_landmark, srcVec.data());
	Dtype a = caffe_cpu_dot(2 * num_landmark, srcVec.data(), dstVec.data()) / (srcVecL2Norm*srcVecL2Norm);
	Dtype b = Dtype(0);
	for (size_t i = 0; i < num_landmark; i++)
	{
		b += srcVec[2 * i] * dstVec[2 * i + 1] - srcVec[2 * i + 1] * dstVec[2 * i];
	}
	b /= srcVecL2Norm*srcVecL2Norm;
	T[0] = a;
	T[1] = b;
	T[2] = -b;
	T[3] = a;
	T[4] = dstMean_x - (srcMean_x*T[0] + srcMean_y*T[2]);
	T[5] = dstMean_y - (srcMean_x*T[1] + srcMean_y*T[3]);
	if (transformed_src != NULL) {
		for (size_t i = 0; i < num_landmark; i++) {
			transformed_src[2 * i + 0] = dstMean_x;
			transformed_src[2 * i + 1] = dstMean_y;
		}
		caffe_cpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans,
			num_landmark, 2, 2, (Dtype)1.,
			srcVec.data(), T, (Dtype)1., transformed_src);
	}
}

template <typename Dtype>
void TransformParamLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  CHECK_EQ(bottom[0]->channels(), bottom[1]->channels()) << "the number of landmark miss match.";
  CHECK_EQ(bottom[1]->num(), 1) << "init shape has only one";
  num_landmark_ = bottom[0]->channels() / 2;
  tmp_landmark_.ReshapeLike(*bottom[0]);
}

template <typename Dtype>
void TransformParamLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
	const vector<Blob<Dtype>*>& top) {
	vector<int> top_shape = bottom[0]->shape();
	top_shape[1] = 6;
	top[0]->Reshape(top_shape);
}

template <typename Dtype>
void TransformParamLayer<Dtype>::Forward_cpu(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  const int num = bottom[0]->num();
  const int dim = bottom[0]->count(1);
  const Dtype* bottom_data = bottom[0]->cpu_data();
  const Dtype* mean_shape_data = bottom[1]->cpu_data();
  Dtype* top_data = top[0]->mutable_cpu_data();
  for (int i = 0; i < num; ++i) {
	  bestFit<Dtype>(num_landmark_, mean_shape_data, bottom_data, NULL, top_data);
	  bottom_data += dim;
	  top_data += 6;
  }
}

template <typename Dtype>
void TransformParamLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
  if (propagate_down[0]) {
	  // nothing to do
  }
}

#ifdef CPU_ONLY
STUB_GPU(TransformParamLayer);
#endif

INSTANTIATE_CLASS(TransformParamLayer);
REGISTER_LAYER_CLASS(TransformParam);

}  // namespace caffe
