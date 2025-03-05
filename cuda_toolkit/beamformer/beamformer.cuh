#ifndef BEAMFORMER_CUH
#define	BEAMFORMER_CUH
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include "../defs.h"

#define OUT_OF_TX_RANGE -1.0f

namespace beamformer
{
	namespace _kernels
	{
		__device__ inline float
		calc_tx_distance(float3 vox_loc, float2* max_lateral_dists);

		__device__ inline float
		f_num_aprodization(float lateral_dist_ratio, float depth, float f_num);

		__global__ void
		per_voxel_beamform(const cuComplex* rfData, cuComplex* volume, float samples_per_meter);

		__global__ void
		per_channel_beamform(const cuComplex* rfData, cuComplex* volume, float samples_per_meter, uint readi_group_id, float* hadamard);
	}

	bool configure_volume(VolumeConfiguration* config);

	bool beamform(cuComplex* d_volume, const cuComplex* d_rf_data, const float3 focus_pos, float samples_per_meter, float f_number);
}

#endif // !BEAMFORMER_CUH
