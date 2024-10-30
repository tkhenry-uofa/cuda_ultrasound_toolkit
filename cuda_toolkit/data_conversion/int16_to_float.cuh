#ifndef INT16_TO_FLOAT_H
#define INT16_TO_FLOAT_H

#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#include "../defs.h"



namespace i16_to_f
{
	__host__ bool
	convert_data(const i16* input, float* output);

	__host__ bool
	copy_channel_mapping(const u16 channel_mapping[TOTAL_TOBE_CHANNELS]);

	namespace _kernels
	{
		__global__ void
		short_to_float(const i16* input, float* output, uint2 input_dims, uint3 output_dims, uint channel_offset);
	}
}

#endif // !INT16_TO_FLOAT_H
