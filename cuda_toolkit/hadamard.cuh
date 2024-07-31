#ifndef HADAMARD_CUH
#define HADAMARD_CUH

#include <iostream>

#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#include "defs.h"

namespace hadamard 
{
	__host__
	bool hadamard_decode_cuda(int sample_count, int channel_count, int tx_count, const int* input, const float* hadamard, float** output);

	namespace kernels
	{
		__global__ void
		generate_hadamard_kernel(float* hadamard, int prev_size, int final_size);

		__global__ void
		init_hadamard_matrix(float* matrix, int size);
	}

	namespace host
	{
		__host__ cudaError_t
		generate_hadamard(uint size, float** dev_ptr);

		__host__
		void print_array(float* out_array, uint size);
	}
}


#endif // !HADAMARD_CUH
