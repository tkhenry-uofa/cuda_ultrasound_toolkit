#ifndef BEAMFORMER_CUH
#define	BEAMFORMER_CUH
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include "../defs.h"

namespace old_beamformer
{
	namespace _kernels
	{
		__global__ void
			delay_and_sum(const cuComplex* rfData, float* volume, float samples_per_meter, const float2* location_array, uint64* times);

		__device__ inline float
			f_num_aprodization(float lateral_dist, float depth, float f_num);

		__global__ void
			double_loop(const cuComplex* rfData, float* volume, float samples_per_meter, uint64* times);
	}

	bool configure_volume(VolumeConfiguration* config);
	bool beamform(float* d_volume, const cuComplex* d_rf_data, const float2* d_loc_data, const float3 focus_pos, float samples_per_meter);
}

#endif // !BEAMFORMER_CUH
