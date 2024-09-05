#include <iostream>
#include <stdexcept>
#include <chrono>
#include <cuda/std/complex>
#include <math_constants.h>
#include <math_functions.h>

#include <cub/cub.cuh> 

#include "beamformer.cuh"

__constant__ KernelConstants Constants;

__device__ __inline__  float
beamformer::_kernels::f_num_aprodization(float lateral_dist, float depth, float f_num)
{
	float apro = f_num * lateral_dist / depth;
	apro = fminf(apro, 0.5);
	apro = cosf(CUDART_PI_F * apro);
	return apro * apro;
}

__global__ void
beamformer::_kernels::delay_and_sum(const cuComplex* rfData, float* volume, float samples_per_meter, const float2* location_array, uint64* times)
{
	__shared__ cuComplex temp[MAX_THREADS_PER_BLOCK/WARP_SIZE];

	uint e = threadIdx.x;

	// Start timing the reduction portion
    uint64 start_time;
    if (e == 0) {
        start_time = clock64();
    }
	const float3 vox_loc =
	{
		Constants.volume_mins.x + blockIdx.x * Constants.resolutions.x,
		Constants.volume_mins.y + blockIdx.y * Constants.resolutions.y,
		Constants.volume_mins.z + blockIdx.z * Constants.resolutions.z,
	};

	float lateral_dist = sqrtf(vox_loc.x * vox_loc.x + vox_loc.y * vox_loc.y);

	float tx_distance;
	int dist_sign = ((vox_loc.z - Constants.src_pos.z) > 0) ? 1 : -1;
	switch (Constants.tx_type)
	{
	case TX_PLANE:
		tx_distance = vox_loc.z;
		break;

	case TX_Y_FOCUS:
		tx_distance = dist_sign * sqrt(powf(Constants.src_pos.z - vox_loc.z, 2) + powf(Constants.src_pos.y - vox_loc.y, 2)) + Constants.src_pos.z;
		break;

	case TX_X_FOCUS:
		tx_distance = dist_sign * sqrt(powf(Constants.src_pos.z - vox_loc.z, 2) + powf(Constants.src_pos.x - vox_loc.x, 2)) + Constants.src_pos.z;
		break;
	}

	cuComplex total, value;
	float3 rx_vec = { ((float)e - 63.5f) * Constants.element_pitch - vox_loc.x, (-63.5f) * Constants.element_pitch - vox_loc.y, vox_loc.z};

	uint64 loop_start;
	if (e == 0) {
		loop_start = clock64();
	}

	uint delay_samples = (uint)roundf(Constants.pulse_delay * Constants.sample_count);
	for (int t = 0; t < 128; t++)
	{
		uint scan_index = (uint)((NORM_F3(rx_vec) + tx_distance) * samples_per_meter + delay_samples);
		size_t channel_offset = (t * Constants.sample_count * Constants.channel_count) + (e * Constants.sample_count);
		value = __ldg(&rfData[channel_offset + scan_index - 1]);
		f_num_aprodization(lateral_dist, vox_loc.z, 1.5);

		total = ADD_F2(total, value);

		rx_vec.y += Constants.element_pitch;
	}

	__syncthreads();

	uint64 reduce_start_time = clock64();
	/* Each warp sums up their totals using intrinsics and stores the output
	 in that warp's index in temp*/
	for (int offset = WARP_SIZE / 2; offset > 0; offset /= 2)
	{
		double double_value = __shfl_xor_sync(0xFFFFFFFF, *(double*)&total, offset);
		total = ADD_F2(*(cuComplex*)&double_value, total);
	}
	if (e % WARP_SIZE == 0)
	{
		temp[e / WARP_SIZE] = total;
	}
	__syncthreads();
	
	if (e == 0) 
	{
		total = { 0.0f, 0.0f };
		for (int i = 1; i < MAX_THREADS_PER_BLOCK / WARP_SIZE; i++)
		{
			total = ADD_F2(total, temp[i]);
		}
		volume[blockIdx.z * gridDim.y * gridDim.x + blockIdx.y * gridDim.x + blockIdx.x] = NORM_F2(total);
		uint64 end_time = clock64();
		times[0] = end_time - start_time;
		times[1] = reduce_start_time - loop_start;
	}
}

__global__ void
beamformer::_kernels::double_loop(const cuComplex* rfData, float* volume, float samples_per_meter, uint64* times)
{
	int tid = threadIdx.x;
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

	float element_pitch = Constants.element_pitch;

	size_t volume_offset = z_voxel * Constants.voxel_dims.x * Constants.voxel_dims.y + y_voxel * Constants.voxel_dims.x + x_voxel;

	const float3 vox_loc =
	{
		Constants.volume_mins.x + x_voxel * Constants.resolutions.x,
		Constants.volume_mins.y + y_voxel * Constants.resolutions.y,
		Constants.volume_mins.z + z_voxel * Constants.resolutions.z,
	};

	float xdc_edge = 63.5f * element_pitch;

	float apro_argument = 0;
	float tx_distance = 0;
	bool diverging = (Constants.src_pos.z < 0.0f);
	if (diverging)
	{
		tx_distance = sqrt(powf(Constants.src_pos.z - vox_loc.z, 2) + powf(Constants.src_pos.y - vox_loc.y, 2)) + Constants.src_pos.z;
		float tx_angle = atan2f(xdc_edge, -Constants.src_pos.z);
		float max_lateral_dist = xdc_edge + vox_loc.z * tanf(tx_angle);
		float2 lateral_ratios = { vox_loc.y / max_lateral_dist, vox_loc.x / xdc_edge };
		lateral_ratios = { fmaxf(lateral_ratios.x, 1),fmaxf(lateral_ratios.y, 1) };

		apro_argument = NORM_F2(lateral_ratios);
		apro_argument = fmaxf(apro_argument, 1);
	}
	else
	{
		tx_distance = vox_loc.z;
		float max_lateral_dist = sqrtf(xdc_edge * xdc_edge);
		apro_argument = sqrt(vox_loc.x * vox_loc.x + vox_loc.y * vox_loc.y) / max_lateral_dist;
		apro_argument = fmaxf(apro_argument, 1);
	}

	float apro_depth = vox_loc.z / Constants.z_max;

	cuComplex total = {0.0f, 0.0f}, value;
	
	uint delay_samples = (uint)roundf(Constants.pulse_delay * 50e6f);

	float3 rx_vec = { -xdc_edge - vox_loc.x, -xdc_edge - vox_loc.y, vox_loc.z };
	float starting_y = rx_vec.y;
	float apro;
	size_t channel_offset = 0;
	uint sample_count = Constants.sample_count;
	uint scan_index;
	for (int t = 0; t < 128; t++)
	{
		for (int e = 0; e < 128; e++)
		{
			scan_index = (uint)((NORM_F3(rx_vec) + tx_distance) * samples_per_meter + delay_samples);
			
			value = __ldg(&rfData[channel_offset + scan_index - 1]);

			apro = f_num_aprodization(apro_argument, apro_depth, 0.5);

			value = SCALE_F2(value, apro);

			if (t == 0) value = SCALE_F2(value, I_SQRT_128);

			total = ADD_F2(total, value);

			rx_vec.y += element_pitch;
			channel_offset += sample_count;

		}
		rx_vec.x += element_pitch;
		rx_vec.y = starting_y;
	}

	float result = sqrtf(total.x * total.x + total.y * total.y);
	volume[volume_offset] = result;

	if (tid == 0)
	{
		uint64 end_time = clock64();
		times[0] = end_time - start_time;
		times[1] = 0;
	}
}

bool
beamformer::beamform(float* d_volume, const cuComplex* d_rf_data, float3 focus_pos, float samples_per_meter)
{

	TransmitType transmit_type;

	if (focus_pos.z == 0.0f)
	{
		transmit_type = TX_PLANE;
	}
	else if (Session.rx_cols)
	{
		// TX on rows (x) axis so we have x focusing
		transmit_type = TX_X_FOCUS;
	}
	else
	{
		transmit_type = TX_Y_FOCUS;
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
		Session.element_pitch,
		Session.pulse_delay,
		vol_config.maximums.z,
	};
	CUDA_RETURN_IF_ERROR(cudaMemcpyToSymbol(Constants, &consts, sizeof(KernelConstants)));


	uint64 *times, *d_times;

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

	std::cout << "First volume value: " << sample_value(d_volume) << std::endl;

	return true;

}

