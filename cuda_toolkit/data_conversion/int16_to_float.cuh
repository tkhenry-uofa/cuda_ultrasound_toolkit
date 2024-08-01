#ifndef INT16_TO_FLOAT_H
#define INT16_TO_FLOAT_H

#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#include "../defs.h"

namespace i16_to_f
{
	__host__ cudaError_t
	convert_data(const i16* input, float* output, uint2 input_dims, defs::RfDataDims output_dims, bool rx_rows);

	namespace _kernels
	{
		__global__ void
		short_convert(const i16* input, float* output, uint2 input_dims, defs::RfDataDims output_dims, bool rx_rows);
	}
}

#endif // !INT16_TO_FLOAT_H
