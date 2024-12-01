#ifndef HADAMARD_CUH
#define HADAMARD_CUH

#include <iostream>

#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#include <cublas_v2.h>

#include "../defs.h"

namespace hadamard 
{
	__host__ bool 
	hadamard_decode(const float* d_input, float* d_output);

	__host__ bool
	generate_hadamard(uint size, float** dev_ptr);

	__host__ bool
	readi_decode(const float* d_input, float* d_output, uint group_number, uint group_size);

	namespace _kernels
	{
		__global__ void
		generate_hadamard(float* hadamard, int prev_size, int final_size);

		__global__ void
		init_hadamard_matrix(float* matrix, int size);
	}

	namespace _host
	{
		__host__ void 
		print_array(float* out_array, uint size);
	}
}


#endif // !HADAMARD_CUH
