#ifndef BEAMFORMER_CUH
#define	BEAMFORMER_CUH
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include "../defs.h"

namespace beamformer
{
	namespace _kernels
	{
		__device__ inline float
		calc_tx_distance(float3 vox_loc, float2* max_lateral_dists);

		__global__ void
		delay_and_sum(const cuComplex* rfData, float* volume, float samples_per_meter, const float2* location_array, uint64* times);

		__device__ inline float
		f_num_aprodization(float lateral_dist_ratio, float depth, float f_num);

		__global__ void
		double_loop(const cuComplex* rfData, cuComplex* volume, float samples_per_meter, uint64* times);

		__global__ void
		coherency_factor_beamform(const cuComplex* rfData, cuComplex* volume, float samples_per_meter, uint readi_group_id);
	}

	bool configure_volume(VolumeConfiguration* config);
	bool beamform(cuComplex* d_volume, const cuComplex* d_rf_data, const float3 focus_pos, float samples_per_meter);

	bool coherency_factor_beamform(cuComplex* d_volume, const cuComplex* d_rf_data, const float3 focus_pos, float samples_per_meter);
}

#endif // !BEAMFORMER_CUH
