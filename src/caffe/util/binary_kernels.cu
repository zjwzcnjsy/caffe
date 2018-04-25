#include <stdio.h>
#include "caffe/util/binary_kernels.hpp"

namespace caffe
{

#define BLOCK_SIZE 16

	// CUDA tutorial: http://www.nvidia.com/docs/IO/116711/sc11-cuda-c-basics.pdf
	// http://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#shared-memory
	// A is shape (m,n), B is shape (n,k) and C is shape (m,k)
	template <typename Dtype>
	__global__ void gemm(Dtype *A, Dtype *alpha, Dtype *B, Dtype *C, int m, int n, int k)
	{

		// Block row and column
		int blockRow = blockIdx.y;
		int blockCol = blockIdx.x;

		// Thread row and column within Csub
		int row = threadIdx.y;
		int col = threadIdx.x;

		// Each thread block computes one sub-matrix Csub of C
		Dtype *Csub = &C[BLOCK_SIZE * k * blockRow + BLOCK_SIZE * blockCol];

		// Shared memory used to store Asub and Bsub respectively
		__shared__ Dtype As[BLOCK_SIZE][BLOCK_SIZE];
		__shared__ Dtype Bs[BLOCK_SIZE][BLOCK_SIZE];

		// Each thread computes one element of Csub
		// by accumulating results into Cvalue
		// block_size = 16 -> 256 threads, one per Csub element
		Dtype Cvalue = 0.0;

		// Loop over all the sub-matrices of A and B that are
		// required to compute Csub
		// Multiply each pair of sub-matrices together
		// and accumulate the results
		for (int i = 0; i < (n / BLOCK_SIZE); ++i)
		{

			// Get sub-matrix Asub of A
			Dtype *Asub = &A[BLOCK_SIZE * blockRow * n + BLOCK_SIZE * i];

			// Get sub-matrix Bsub of B
			Dtype *Bsub = &B[BLOCK_SIZE * k * i + BLOCK_SIZE * blockCol];

			// Load Asub and Bsub from device memory to shared memory
			// Each thread loads one element of each sub-matrix
			As[row][col] = Asub[row * n + col];
			Bs[row][col] = Bsub[row * k + col];

			// Synchronize to make sure the sub-matrices are loaded
			// before starting the computation
			__syncthreads();

			// Multiply Asub and Bsub together
			for (int j = 0; j < BLOCK_SIZE; ++j)
				Cvalue += As[row][j] * Bs[j][col];

			// Synchronize to make sure that the preceding
			// computation is done before loading two new
			// sub-matrices of A and B in the next iteration
			__syncthreads();
		}

		// Write Csub to device memory
		// Each thread writes one element
		if (col + blockCol * BLOCK_SIZE < k && row + blockRow * BLOCK_SIZE < m)
			Csub[row * k + col] = alpha[row + blockRow * BLOCK_SIZE]*Cvalue;
	}


	// 32 single float array ->  32 bits unsigned int
	__device__ unsigned int concatenate(float *array)
	{
		unsigned int rvalue = 0;
		unsigned int sign;

		for (int i = 0; i < 32; i++)
		{
			sign = (array[i] >= 0);
			rvalue = rvalue | (sign << i);
		}

		return rvalue;
	}

	__global__ void concatenate_rows_kernel(float *a, unsigned int *b, int size)
	{
		int i = blockIdx.x * blockDim.x + threadIdx.x;
		if (i < size)
			b[i] = concatenate(&a[i * 32]);
	}

	__global__ void concatenate_cols_kernel(float *a, unsigned int *b, int m, int n)
	{

		int j = blockIdx.x * blockDim.x + threadIdx.x;

		if (j < n)
		{
			__shared__ float array[32];
			for (int i = 0; i < m; i += 32)
			{
				for (int k = 0; k < 32; k++)
					array[k] = a[j + n * (i + k)];
				b[j + n * i / 32] = concatenate(array);
			}
		}
	}

	// 32 bits unsigned int -> 32 single float array
	// TODO: the array allocation should not be done here
	__device__ float *deconcatenate(unsigned int x)
	{
		float *array = new float[32];

		for (int i = 0; i < 32; i++)
		{
			array[i] = (x & (1 << i)) >> i;
		}

		return array;
	}

	__global__ void deconcatenate_rows_kernel(unsigned int *a, float *b, int size)
	{
		float *array;

		for (int i = 0; i < size; i += 32)
		{
			array = deconcatenate(a[i / 32]);
			for (int k = 0; k < 32; k++)
				b[i + k] = array[k];
			delete[] array;
		}
	}

	// A is shape (m,n), B is shape (n,k) and C is shape (m,k)
	__global__ void xnor_gemm(unsigned int *A, float *fA, unsigned int *B, float *C, int m, int n, int k)
	{

		// Block row and column
		int blockRow = blockIdx.y;
		int blockCol = blockIdx.x;

		// Thread row and column within Csub
		int row = threadIdx.y;
		int col = threadIdx.x;

		// Each thread block computes one sub-matrix Csub of C
		float *Csub = &C[BLOCK_SIZE * k * blockRow + BLOCK_SIZE * blockCol];

		// Shared memory used to store Asub and Bsub respectively
		__shared__ unsigned int As[BLOCK_SIZE][BLOCK_SIZE];
		__shared__ unsigned int Bs[BLOCK_SIZE][BLOCK_SIZE];

		// Each thread computes one element of Csub
		// by accumulating results into Cvalue
		// block_size = 16 -> 256 threads, one per Csub element
		unsigned int Cvalue = 0;

		// Loop over all the sub-matrices of A and B that are
		// required to compute Csub
		// Multiply each pair of sub-matrices together
		// and accumulate the results
		for (int i = 0; i < (n / BLOCK_SIZE); ++i)
		{

			// Get sub-matrix Asub of A
			unsigned int *Asub = &A[BLOCK_SIZE * blockRow * n + BLOCK_SIZE * i];

			// Get sub-matrix Bsub of B
			unsigned int *Bsub = &B[BLOCK_SIZE * k * i + BLOCK_SIZE * blockCol];

			// Load Asub and Bsub from device memory to shared memory
			// Each thread loads one element of each sub-matrix
			As[row][col] = Asub[row * n + col];
			Bs[row][col] = Bsub[row * k + col];

			// Synchronize to make sure the sub-matrices are loaded
			// before starting the computation
			__syncthreads();

			// Multiply Asub and Bsub together
			// THIS IS THE MOST INTERESTING PART
			for (int j = 0; j < BLOCK_SIZE; ++j)
				Cvalue += __popc(As[row][j] ^ Bs[j][col]);

			// Synchronize to make sure that the preceding
			// computation is done before loading two new
			// sub-matrices of A and B in the next iteration
			__syncthreads();
		}

		// Write Csub to device memory
		// Each thread writes one element
		if (col + blockCol * BLOCK_SIZE < k && row + blockRow * BLOCK_SIZE < m)
			Csub[row * k + col] = fA[row + blockRow * BLOCK_SIZE] * (-(2 * (float)Cvalue - 32 * n));
	}

	// A is shape (m,n), B is shape (n,k) and C is shape (m,k)
	__global__ void xnor_gemm2(const int nThreads, unsigned int *A, float *fA, unsigned int *B, float *C, int m, int n, int k)
	{
		CUDA_KERNEL_LOOP(index, nThreads) {
			const int row = index / k;
			const int col = index % k;
			unsigned int Cvalue = 0;
			for (int t = 0; t < n; t++)
			{
				Cvalue += __popc(A[row*n+t] ^ B[t*k+col]);
			}
			C[index] = fA[row] * (-(2 * (float)Cvalue - 32 * n));
		}
	}

	// A is shape (m,n), B is shape (n,k) and C is shape (m,k)
	template <>
	void xnor_gemm(const float *fw, const float* fA, const float *fB, float *fC,
		unsigned int *uiA, unsigned int *uiB, int m, int n, int k)
	{
		//LOG(INFO) << "m=" << m << ", n=" << n << ", k=" << k;
		CHECK_EQ(n % 32, 0) << "n must be div by 32";
		int block = 64, grid = m * n / (block * 32) + 1;
		concatenate_rows_kernel << <grid, block >> >(const_cast<float*>(fw), uiA, m * n / 32);

		grid = k / block + 1;
		concatenate_cols_kernel << <grid, block >> >(const_cast<float*>(fB), uiB, n, k);

		/*dim3 blockDim(16, 16);
		dim3 gridDim(k / 16 + 1, m / 16 + 1);
		xnor_gemm << <gridDim, blockDim >> >(uiA, const_cast<float*>(fA), uiB, fC, m, n / 32, k);*/

		xnor_gemm2 << <CAFFE_GET_BLOCKS(m*k), CAFFE_CUDA_NUM_THREADS >> >(m*k, uiA, const_cast<float*>(fA), uiB, fC, m, n / 32, k);
	}

	template <>
	void xnor_gemm(const double *fw, const double *fA, const double *fB, double *fC,
		unsigned int *uiA, unsigned int *uiB, int m, int n, int k)
	{
		dim3 blockDim(16, 16);
		dim3 gridDim(k / 16 + 1, m / 16 + 1);
		gemm<double> << <gridDim, blockDim >> >(const_cast<double*>(fw), const_cast<double*>(fA), const_cast<double*>(fB), fC, m, n, k);
	}

}  // namespace caffe