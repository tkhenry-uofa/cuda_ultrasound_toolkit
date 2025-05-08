#ifndef HILBERT_TRANSFORM_CUH
#define HILBERT_TRANSFORM_CUH

#include <iostream>
#include <vector>
#include <complex>

#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#include "../defs.h"

namespace hilbert
{
	namespace kernels
	{
		__global__ void
		filter(cuComplex* data, uint sample_count, uint cutoff_sample);

		__global__ void
		scale_and_filter(cuComplex* signals, cuComplex* filter_kernel, uint sample_count);
	}

	__host__ bool
	f_domain_filter(cuComplex* input);

	__host__ bool
	plan_hilbert(int sample_count, int channel_count);

	__host__ bool
	setup_filter(int signal_length, int filter_length, const float* filter);

	__host__ bool 
	hilbert_transform(float* d_input, cuComplex* d_output);

	__host__ bool
	hilbert_transform_strided(float* d_input, cuComplex* d_output);

}




#endif // !HILBERT_TRANSFORM_CUH
