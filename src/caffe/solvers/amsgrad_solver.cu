#include "caffe/util/math_functions.hpp"


namespace caffe {

template <typename Dtype>
__global__ void AMSGradUpdate(int N, Dtype* w, Dtype* g, Dtype* m, Dtype* v, Dtype* v_hat,
    Dtype beta1, Dtype beta2, Dtype eps_hat, Dtype corrected_local_rate, Dtype local_decay) {
  CUDA_KERNEL_LOOP(i, N) {
    float gi = g[i];
    float mi = m[i] = m[i]*beta1 + gi*(1-beta1);
    float vi = v[i] = v[i]*beta2 + gi*gi*(1-beta2);
    v_hat[i] = max(v_hat[i], vi);
    g[i] = corrected_local_rate * mi / (sqrt(v_hat[i]) + eps_hat);// + local_decay*w[i];
  }
}
template <typename Dtype>
void amsgrad_update_gpu(int N, Dtype* w, Dtype* g, Dtype* m, Dtype* v, Dtype* v_hat, Dtype beta1,
    Dtype beta2, Dtype eps_hat, Dtype corrected_local_rate, Dtype local_decay) {
  AMSGradUpdate<Dtype>  // NOLINT_NEXT_LINE(whitespace/operators)
      <<<CAFFE_GET_BLOCKS(N), CAFFE_CUDA_NUM_THREADS>>>(
      N, w, g, m, v, v_hat, beta1, beta2, eps_hat, corrected_local_rate, local_decay);
  CUDA_POST_KERNEL_CHECK;
}
template void amsgrad_update_gpu<float>(int, float*, float*, float*, float*, float*,
    float, float, float, float, float);
template void amsgrad_update_gpu<double>(int, double*, double*, double*, double*, double*,
    double, double, double, double, double);

}  // namespace caffe
