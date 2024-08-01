#ifndef HADAMARD_CUH
#define HADAMARD_CUH

#include <iostream>

#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#include <cublas_v2.h>

#include "defs.h"

namespace hadamard 
{
	__host__
	bool hadamard_decode(int sample_count, int channel_count, int tx_count, const float* input, const float* hadamard, float** output);

	namespace _kernels
	{
		__global__ void
		generate_hadamard_kernel(float* hadamard, int prev_size, int final_size);

		__global__ void
		init_hadamard_matrix(float* matrix, int size);
	}

	namespace _host
	{
		__host__ cudaError_t
		generate_hadamard(uint size, float** dev_ptr);

		// Data dimensions are: samples X channels X transmissions
		// Hadamard dimension is transmission number
		__host__ cublasStatus_t
		decode(const float* data, const float* hadamard, float* output, int3 data_dims);

		__host__ void 
		print_array(float* out_array, uint size);
	}
}


#endif // !HADAMARD_CUH
