#ifndef BEAMFORMER_CUH
#define	BEAMFORMER_CUH
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include "../defs.h"

namespace beamformer
{
	namespace _kernels
	{
		__device__ inline float3
		calc_tx_distance(float3 vox_loc, float3 source_pos);

		__device__ inline float
		total_path_length(float3 tx_vec, float3 rx_vec, float focal_depth, float vox_depth);


		// Any voxels outside of the f# defined range of all channels are skipped
		__device__ inline bool
		check_ranges(float3 vox_loc, float f_number, float2 array_edges);

		__device__ inline float
		f_num_apodization(float lateral_dist_ratio, float depth, float f_num);

		__global__ void
		forces_beamformer(const cuComplex* rfData, cuComplex* volume, uint readi_group_id, float* hadamard);

		__global__ void
		per_voxel_beamform(const cuComplex* rfData, cuComplex* volume, uint readi_group_id, float* hadamard);

		__global__ void
		per_channel_beamform(const cuComplex* rfData, cuComplex* volume, uint readi_group_id, float* hadamard);

		__global__ void
		mixes_beamform(const cuComplex* rfData, cuComplex* volume);
	}

	bool beamform(cuComplex* d_volume, const cuComplex* d_rf_data, const float3 focus_pos, float samples_per_meter, float f_number, int delay_samples);

	bool forces_beamform(cuComplex* d_volume, const cuComplex* d_rf_data, const float3 focus_pos, float samples_per_meter, float f_number);
}

#endif // !BEAMFORMER_CUH
