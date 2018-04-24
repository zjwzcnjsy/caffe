#ifndef _CAFFE_UTIL_BINARY_KERNELS_HPP_
#define _CAFFE_UTIL_BINARY_KERNELS_HPP_

#include "caffe/common.hpp"

namespace caffe {

	// A is shape (m,n), B is shape (n,k) and C is shape (m,k)
	template <typename Dtype>
	void xnor_gemm(const Dtype* fw, const Dtype* fA, const Dtype* fd, Dtype* fC,
		unsigned int* uiA, unsigned int* uiB, int m, int n, int k);

}  // namespace caffe

#endif  // _CAFFE_UTIL_BINARY_KERNELS_HPP_
