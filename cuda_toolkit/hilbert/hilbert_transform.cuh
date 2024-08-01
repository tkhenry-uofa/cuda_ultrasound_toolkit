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

	__host__ bool
	plan_hilbert(int sample_count, int channel_count, cufftHandle* fwd_handle, cufftHandle* inv_handle);

	__host__
	bool hilbert_transform(cufftHandle fwd_handle, cufftHandle inv_handle, float* d_input, cufftComplex* d_output);
}




#endif // !HILBERT_TRANSFORM_CUH
