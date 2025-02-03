#include <iostream>
#include <stdexcept>
#include <chrono>
#include <cuda/std/complex>
#include <math_constants.h>
#include <math_functions.h>

#include <cub/cub.cuh> 

#include "beamformer.cuh"

__constant__ KernelConstants Constants;

#define MAX_CHANNEL_COUNT 128
#define MAX_TX_COUNT 128

__device__ __inline__  float
beamformer::_kernels::f_num_aprodization(float lateral_dist, float depth, float f_num)
{
	float apro = f_num * lateral_dist / depth/2;
	apro = fminf(apro, 0.5);
	apro = cosf(CUDART_PI_F * apro);
	return apro * apro;
}

__device__ cuComplex 
reduce_shared_sum(const cuComplex* sharedVals, const uint channel_count) 
{
	// Each thread will compute a partial sum.
	int tid = threadIdx.x;

	// Compute partial sum over a stripe of the shared memory.
	cuComplex partial_sum = { 0.0f, 0.0f };
	// Each thread processes elements starting at its index and strides by the block size.
	for (int i = tid; i < channel_count; i += blockDim.x) {
		partial_sum = ADD_F2( sharedVals[i], partial_sum);
	}

	__shared__ cuComplex aux[MAX_TX_COUNT];
	aux[tid] = partial_sum;
	__syncthreads();

	// Perform iterative tree-based reduction.
	// Since blockDim.x is 128, we reduce until we have one value.
	for (int stride = blockDim.x / 2; stride > 0; stride /= 2) {
		if (tid < stride) {
			aux[tid] = ADD_F2(aux[tid], aux[tid + stride]);
		}
		__syncthreads();
	}
	return aux[0];
}

// Returns true if this element should be used for mixes
__device__ __inline__ bool
offset_mixes(int transmit, int element, int mixes_spacing, int offset, int pivot)
{
	int transmit_offset = 0;
	int element_offset = 0;


	if (transmit >= pivot) element_offset = offset;
	if (element >= pivot) transmit_offset = offset;
	
	if (element % mixes_spacing != element_offset && transmit % mixes_spacing != transmit_offset)
	{
		return false;
	}

	return true;
}

__device__ inline float
beamformer::_kernels::calc_tx_distance(float3 vox_loc, float2* max_lateral_dists)
{
	float tx_distance = 0.0f;
	if (Constants.tx_type == TX_X_FOCUS)
	{
		tx_distance = sqrt(powf(Constants.src_pos.z - vox_loc.z, 2) + powf(Constants.src_pos.x - vox_loc.x, 2)) + Constants.src_pos.z;

		float tx_angle = atan2f(Constants.xdc_maxes.x, -Constants.src_pos.z);

		max_lateral_dists->x = Constants.xdc_maxes.x + vox_loc.z * tanf(tx_angle);
		max_lateral_dists->y = Constants.xdc_maxes.y * 2;

		float2 lateral_ratios = { vox_loc.x / max_lateral_dists->x , vox_loc.y / max_lateral_dists->y };

		if (lateral_ratios.x >= 1.0f || lateral_ratios.y >= 1.0f) return;
		if (lateral_ratios.x <= -1.0f || lateral_ratios.y <= -1.0f) return;

	}
	else if (Constants.tx_type == TX_Y_FOCUS)
	{
		tx_distance = sqrt(powf(Constants.src_pos.z - vox_loc.z, 2) + powf(Constants.src_pos.y - vox_loc.y, 2)) + Constants.src_pos.z;

		float tx_angle = atan2f(Constants.xdc_maxes.y, -Constants.src_pos.z);


		max_lateral_dists->x = Constants.xdc_maxes.x * 2;
		max_lateral_dists->y = Constants.xdc_maxes.y + vox_loc.z * tanf(tx_angle);

		float2 lateral_ratios = { vox_loc.x / max_lateral_dists->x , vox_loc.y / max_lateral_dists->y };

		if (lateral_ratios.x >= 1.0f || lateral_ratios.y >= 1.0f) return;
		if (lateral_ratios.x <= -1.0f || lateral_ratios.y <= -1.0f) return;

	}
	else
	{
		tx_distance = vox_loc.z;
		max_lateral_dists->x = 2 * Constants.xdc_maxes.x;
		max_lateral_dists->y = 2 * Constants.xdc_maxes.y;
	}

	return tx_distance;
}

__global__ void
beamformer::_kernels::double_loop(const cuComplex* rfData, cuComplex* volume, float samples_per_meter, uint64* times)
{
	uint tid = threadIdx.x;
	uint64 start_time;
	if (tid == 0)
	{
		start_time = clock64();
	}

	uint xy_voxel = threadIdx.x + blockIdx.x * blockDim.x;

	if (xy_voxel > Constants.voxel_dims.x * Constants.voxel_dims.y)
	{
		return;
	}

	uint x_voxel = xy_voxel % Constants.voxel_dims.x;
	uint y_voxel = xy_voxel / Constants.voxel_dims.x;
	uint z_voxel = blockIdx.y;

	size_t volume_offset = z_voxel * Constants.voxel_dims.x * Constants.voxel_dims.y + y_voxel * Constants.voxel_dims.x + x_voxel;

	const float3 vox_loc =
	{
		Constants.volume_mins.x + x_voxel * Constants.resolutions.x,
		Constants.volume_mins.y + y_voxel * Constants.resolutions.y,
		Constants.volume_mins.z + z_voxel * Constants.resolutions.z,
	};

	float2 max_lateral_dists;
	float tx_distance = calc_tx_distance(vox_loc, &max_lateral_dists);
	float apro_depth = vox_loc.z / Constants.z_max;
	cuComplex total = {0.0f, 0.0f}, value;
	float incoherent_sum = 0.0f;

	float3 rx_vec = { Constants.xdc_mins.x - vox_loc.x + Constants.pitches.x/2, Constants.xdc_mins.y - vox_loc.y + Constants.pitches.y / 2, vox_loc.z };	
	//float3 rx_vec = { Constants.xdc_mins.x - vox_loc.x + Constants.pitches.x / 2, Constants.xdc_mins.y - vox_loc.y, vox_loc.z };
	//float3 rx_vec = { vox_loc.x - Constants.pitches.x / 2, Constants.xdc_mins.y - vox_loc.y + Constants.pitches.y / 2, vox_loc.z };
	//float3 rx_vec = { vox_loc.x - Constants.pitches.x / 2, vox_loc.y - Constants.pitches.y / 2, vox_loc.z };

	uint delay_samples = 12;


	float starting_x = rx_vec.x;
	float apro;
	size_t channel_offset = 0;
	uint sample_count = Constants.sample_count;
	uint scan_index;
	uint channel_count = Constants.channel_count;

	int mixes_number = 128;
	int mixes_spacing = 128/mixes_number;
	int mixes_offset = 0;
	int total_used_channels = 0;
	//int mixes_offset = mixes_spacing / 2;
	for (int t = 0; t < Constants.tx_count; t++)
	{
		for (int e = 0; e < channel_count; e++)
		{
			if (!offset_mixes(t, e, mixes_spacing, mixes_offset, 64))
			{
				rx_vec.x += Constants.pitches.x;
				continue;
			}

			channel_offset = channel_count * sample_count * t + sample_count * e;

			float2 lateral_ratios = { rx_vec.x / max_lateral_dists.x, rx_vec.y / max_lateral_dists.y };

			scan_index = (uint)((NORM_F3(rx_vec) + tx_distance) * samples_per_meter + 0);
			value = __ldg(&rfData[channel_offset + scan_index - 1]);

			float apro_argument = NORM_F2(lateral_ratios);
			apro = f_num_aprodization(apro_argument, apro_depth, 0.0);
			value = SCALE_F2(value, apro);

			total = ADD_F2(total, value);
			incoherent_sum += NORM_SQUARE_F2(value);

			rx_vec.x += Constants.pitches.x;
			total_used_channels++;

		}
		rx_vec.y += Constants.pitches.y;
		rx_vec.x = starting_x;
	}

	float coherent_sum = NORM_SQUARE_F2(total);

	float coherency_factor = coherent_sum / (incoherent_sum * total_used_channels);
	//volume[volume_offset] = SCALE_F2(total, coherency_factor);;

	volume[volume_offset] = total;

	if (tid == 0)
	{
		uint64 end_time = clock64();
		times[0] = end_time - start_time;
		times[1] = 0;
	}
}

__global__ void
beamformer::_kernels::per_channel_beamform(const cuComplex* rfData, cuComplex* volume, float samples_per_meter, uint readi_group_id)
{
	uint tid = threadIdx.x;

	uint channel_id = threadIdx.x;

	__shared__ cuComplex das_samples[MAX_CHANNEL_COUNT]; // 128

	uint x_voxel = blockIdx.x;
	uint y_voxel = blockIdx.y;
	uint z_voxel = blockIdx.z;

	size_t volume_offset = z_voxel * Constants.voxel_dims.x * Constants.voxel_dims.y + y_voxel * Constants.voxel_dims.x + x_voxel;

	const float3 vox_loc =
	{
		Constants.volume_mins.x + x_voxel * Constants.resolutions.x,
		Constants.volume_mins.y + y_voxel * Constants.resolutions.y,
		Constants.volume_mins.z + z_voxel * Constants.resolutions.z,
	};

	uint transmit_count = Constants.tx_count;

	float2 max_lateral_dists;
	float tx_distance = calc_tx_distance(vox_loc, &max_lateral_dists);
	float apro_depth = vox_loc.z / Constants.z_max;

	float3 rx_vec =	  { Constants.xdc_mins.x - vox_loc.x + channel_id * Constants.pitches.x + Constants.pitches.x / 2, 
						Constants.xdc_mins.y - vox_loc.y + Constants.pitches.y / 2, 
						vox_loc.z };

	uint delay_samples = 12;
	float apro;
	size_t channel_offset = 0;
	uint sample_count = Constants.sample_count;
	uint scan_index;
	uint channel_count = Constants.channel_count;
	uint sample_offset = channel_id;
	cuComplex value;
	cuComplex channel_total = { 0.0f,0.0f };
	for (int t = 0; t < transmit_count; t++)
	{
		channel_offset = channel_count * sample_count * t + sample_count * channel_id;

		float2 lateral_ratios = { rx_vec.x / max_lateral_dists.x, rx_vec.y / max_lateral_dists.y };

		scan_index = (uint)((NORM_F3(rx_vec) + tx_distance) * samples_per_meter + 0);
		value = __ldg(&rfData[channel_offset + scan_index - 1]);

		float apro_argument = NORM_F2(lateral_ratios);
		apro = f_num_aprodization(apro_argument, apro_depth, 0.0);
		value = SCALE_F2(value, apro);

		channel_total = ADD_F2(channel_total,value);

		rx_vec.y += Constants.pitches.y;
	}

	__syncthreads();

	das_samples[channel_id] = channel_total;

	cuComplex vox_total = reduce_shared_sum(das_samples, channel_count);

	if (channel_id == 0)
	{
		volume[volume_offset] = vox_total;
	}

}


bool
beamformer::per_channel_beamform(cuComplex* d_volume, const cuComplex* d_rf_data, float3 focus_pos, float samples_per_meter)
{

	TransmitType transmit_type;

	if (focus_pos.z == 0.0f || focus_pos.z == INFINITY)
	{
		transmit_type = TX_PLANE;
	}
	//else if (Session.channel_offset > 0)
	//{
	//	// TX on columns (x) axis so we have x focusing
	//	transmit_type = TX_X_FOCUS;
	//}
	else
	{
		transmit_type = TX_X_FOCUS;
	}

	VolumeConfiguration vol_config = Session.volume_configuration;

	KernelConstants consts =
	{
		Session.decoded_dims.x,
		Session.decoded_dims.y,
		Session.decoded_dims.z,
		vol_config.voxel_counts,
		vol_config.minimums,
		{vol_config.lateral_resolution, vol_config.lateral_resolution, vol_config.axial_resolution},
		focus_pos,
		transmit_type,
		Session.pitches,
		Session.pulse_delay,
		vol_config.maximums.z,
		Session.xdc_mins,
		Session.xdc_maxes
	};
	CUDA_RETURN_IF_ERROR(cudaMemcpyToSymbol(Constants, &consts, sizeof(KernelConstants)));

	uint64* times, * d_times;

	times = (uint64*)malloc(2 * sizeof(uint64));
	CUDA_RETURN_IF_ERROR(cudaMalloc(&d_times, 2 * sizeof(uint64)));

	uint3 vox_counts = vol_config.voxel_counts;

	dim3 grid_dim = { vox_counts.x, vox_counts.y, vox_counts.z};
	dim3 block_dim = { Session.decoded_dims.y, 1, 1 };

	auto start = std::chrono::high_resolution_clock::now();


	_kernels::per_channel_beamform << < grid_dim, block_dim >> > (d_rf_data, d_volume, samples_per_meter, Session.readi_group);

	CUDA_RETURN_IF_ERROR(cudaGetLastError());
	CUDA_RETURN_IF_ERROR(cudaDeviceSynchronize());

	auto end = std::chrono::high_resolution_clock::now();
	std::chrono::duration<double> elapsed = end - start;
	std::cout << "Kernel duration: " << elapsed.count() << " seconds" << std::endl;

	return true;
}




bool
beamformer::beamform(cuComplex* d_volume, const cuComplex* d_rf_data, float3 focus_pos, float samples_per_meter)
{

	TransmitType transmit_type;

	if (focus_pos.z == 0.0f || focus_pos.z == INFINITY)
	{
		transmit_type = TX_PLANE;
	}
	//else if (Session.channel_offset > 0)
	//{
	//	// TX on columns (x) axis so we have x focusing
	//	transmit_type = TX_X_FOCUS;
	//}
	else
	{
		transmit_type = TX_X_FOCUS;
	}

	VolumeConfiguration vol_config = Session.volume_configuration;

	KernelConstants consts =
	{
		Session.decoded_dims.x,
		Session.decoded_dims.y,
		Session.decoded_dims.z,
		vol_config.voxel_counts,
		vol_config.minimums,
		{vol_config.lateral_resolution, vol_config.lateral_resolution, vol_config.axial_resolution},
		focus_pos,
		transmit_type,
		Session.pitches,
		Session.pulse_delay,
		vol_config.maximums.z,
		Session.xdc_mins,
		Session.xdc_maxes
	};
	CUDA_RETURN_IF_ERROR(cudaMemcpyToSymbol(Constants, &consts, sizeof(KernelConstants)));

	uint64* times, * d_times;

	times = (uint64*)malloc(2 * sizeof(uint64));
	CUDA_RETURN_IF_ERROR(cudaMalloc(&d_times, 2 * sizeof(uint64)));

	uint3 vox_counts = vol_config.voxel_counts;
	uint xy_count = vox_counts.x * vox_counts.y;
	dim3 grid_dim = { (uint)ceilf((float)xy_count / MAX_THREADS_PER_BLOCK), (uint)vox_counts.z, 1 };
	dim3 block_dim = { MAX_THREADS_PER_BLOCK, 1, 1 };

	auto start = std::chrono::high_resolution_clock::now();

	_kernels::double_loop << < grid_dim, block_dim >> > (d_rf_data, d_volume, samples_per_meter, d_times);

	CUDA_RETURN_IF_ERROR(cudaGetLastError());
	CUDA_RETURN_IF_ERROR(cudaDeviceSynchronize());

	auto end = std::chrono::high_resolution_clock::now();
	std::chrono::duration<double> elapsed = end - start;
	std::cout << "Kernel duration: " << elapsed.count() << " seconds" << std::endl;

	return true;
}

