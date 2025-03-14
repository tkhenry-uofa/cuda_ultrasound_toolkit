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
beamformer::_kernels::f_num_apodization(float lateral_dist, float depth, float f_num)
{
	// When lateral_dist > depth / f_num clamp the argument to pi/2 so that the cos is 0
	// Otherwise the ratio will map between 0 and pi/2 forming a hann window
	float apo = f_num * (lateral_dist / depth) /2;
	apo = fminf(apo, 0.5);
	apo = cosf(CUDART_PI_F * apo);
	return apo * apo; // cos^2
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

__device__ inline bool
beamformer::_kernels::check_ranges(float3 vox_loc, float f_number, float2 array_edges)
{
	// Get the max aperture size for this depth
	float lateral_extent = vox_loc.z / f_number;

	// Model 2 1D apertures to maintain square planes
	float x_extent = lateral_extent + array_edges.x;
	float y_extent = lateral_extent + array_edges.y;

	return (abs(vox_loc.x) < x_extent && abs(vox_loc.y) < y_extent);
}

__device__ inline float3
beamformer::_kernels::calc_tx_distance(float3 vox_loc, float3 source_pos)
{
	float3 tx_distance;
	if (Constants.tx_type == TX_X_FOCUS)
	{
		tx_distance = { source_pos.x - vox_loc.x, 0.0f, source_pos.z - vox_loc.z};
	}
	else if (Constants.tx_type == TX_Y_FOCUS)
	{
		tx_distance = { 0.0f, source_pos.y - vox_loc.y, source_pos.z - vox_loc.z };
	}
	else
	{
		tx_distance = {0.0f, 0.0f, vox_loc.z };
	}

	return tx_distance;
}

__device__ inline float
beamformer::_kernels::total_path_length(float3 tx_vec, float3 rx_vec, float focal_depth, float vox_depth)
{
	// If the voxel is shallower than the focus we need to subtract the tx vec
	int sign = vox_depth > focal_depth ? 1 : -1;
	return focal_depth + NORM_F3(rx_vec) + NORM_F3(tx_vec) * sign;
}

__global__ void
beamformer::_kernels::per_voxel_beamform(const cuComplex* rfData, cuComplex* volume, uint readi_group_id, float* hadamard)
{
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

	// If the voxel is out of the f_number defined range for all elements skip it
	if (!check_ranges(vox_loc, Constants.f_number, Constants.xdc_maxes)) return;
	
	float3 src_pos = Constants.src_pos;
	if (Constants.sequence == DAS_ID_FORCES)
	{
		src_pos.x = Constants.xdc_mins.x + Constants.pitches.x / 2;
		src_pos.z = 0; // Ignoring the elevational focus as it is out of plane
	}


	float3 tx_vec = calc_tx_distance(vox_loc, src_pos);

	float3 rx_vec = { Constants.xdc_mins.x - vox_loc.x + Constants.pitches.x / 2, Constants.xdc_mins.y - vox_loc.y + Constants.pitches.y / 2, vox_loc.z };
	
	if (Constants.sequence == DAS_ID_FORCES)
	{
		rx_vec.y = 0;
	}

	int readi_group_size = Constants.channel_count / Constants.tx_count;
	uint hadamard_offset = Constants.channel_count * readi_group_id;
	uint delay_samples = 12;

	cuComplex total = { 0.0f, 0.0f }, value;
	float incoherent_sum = 0.0f;

	float starting_x = rx_vec.x;
	float apo;
	size_t channel_offset = 0;
	uint sample_count = Constants.sample_count;
	uint scan_index;
	uint channel_count = Constants.channel_count;
	float samples_per_meter = Constants.samples_per_meter;

	int mixes_number = 128;
	int mixes_spacing = 128/mixes_number;
	int mixes_offset = 0;
	int total_used_channels = 0;
	//int mixes_offset = mixes_spacing / 2;

	float total_distance = 0.0f;
	for (int t = 0; t < Constants.tx_count; t++)
	{
		for (int e = 0; e < channel_count; e++)
		{
			channel_offset = channel_count * sample_count * t + sample_count * e;
			for (int g = 0; g < readi_group_size; g++)
			{
				if (!offset_mixes(t, e, mixes_spacing, mixes_offset, 64))
				{
					rx_vec.x += Constants.pitches.x;
					continue;
				}

				total_distance = total_path_length(tx_vec, rx_vec, src_pos.z, vox_loc.z);
				scan_index = (uint)(total_distance * samples_per_meter + delay_samples);
				value = __ldg(&rfData[channel_offset + scan_index - 1]);


				apo = f_num_apodization(NORM_F2(rx_vec), vox_loc.z, Constants.f_number);
				value = SCALE_F2(value, apo);

				// This acts as the final decoding step for the data within the readi group
				// If readi is turned off this will just scan the first row of the hadamard matrix (all 1s)
				value = SCALE_F2(value, hadamard[hadamard_offset + g]);

				total = ADD_F2(total, value);
				incoherent_sum += NORM_SQUARE_F2(value);

				rx_vec.x += Constants.pitches.x;
				total_used_channels++;

			}
		}
		rx_vec.x = starting_x;

		if (Constants.sequence == TransmitModes::DAS_ID_HERCULES)
		{
			rx_vec.y += Constants.pitches.x;
		}
		else if (Constants.sequence == TransmitModes::DAS_ID_FORCES)
		{
			tx_vec.x += Constants.pitches.x;
		}

		
	}

	float coherent_sum = NORM_SQUARE_F2(total);

	//float coherency_factor = coherent_sum / (incoherent_sum * total_used_channels);
	//volume[volume_offset] = SCALE_F2(total, coherency_factor);

	volume[volume_offset] = total;
}

__global__ void
beamformer::_kernels::per_channel_beamform(const cuComplex* rfData, cuComplex* volume, uint readi_group_id, float* hadamard)
{
	uint tid = threadIdx.x;

	uint channel_id = threadIdx.x;

	__shared__ cuComplex das_samples[MAX_CHANNEL_COUNT * 2]; // 128 * 2

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

	// If the voxel is out of the f_number defined range for all elements skip it
	if (!check_ranges(vox_loc, Constants.f_number, Constants.xdc_maxes)) return;

	float3 tx_vec = calc_tx_distance(vox_loc, Constants.src_pos);

	float3 rx_vec =	  { Constants.xdc_mins.x - vox_loc.x + channel_id * Constants.pitches.x + Constants.pitches.x / 2, 
						Constants.xdc_mins.y - vox_loc.y + Constants.pitches.y / 2, 
						vox_loc.z };



	uint readi_group_size = Constants.channel_count / Constants.tx_count;
	uint hadamard_offset = Constants.channel_count * readi_group_id;
	float f_number = Constants.f_number;
	uint delay_samples = 12;
	float apo;
	size_t channel_offset = 0;
	uint sample_count = Constants.sample_count;
	uint scan_index;
	uint channel_count = Constants.channel_count;
	cuComplex value;
	cuComplex channel_total = { 0.0f,0.0f };
	float incoherent_total = 0.0f;
	float total_distance = 0.0f;
	float samples_per_meter = Constants.samples_per_meter;
	for (int t = 0; t < transmit_count; t++)
	{
		channel_offset = channel_count * sample_count * t + sample_count * channel_id;

		for (int g = 0; g < readi_group_size; g++)
		{
			total_distance = total_path_length(tx_vec, rx_vec, Constants.src_pos.z, vox_loc.z);
			scan_index = (uint)(total_distance * samples_per_meter + delay_samples);
			value = __ldg(&rfData[channel_offset + scan_index - 1]);

			apo = f_num_apodization(NORM_F2(rx_vec), vox_loc.z, f_number);

			// This acts as the final decoding step for the data within the readi group
			// If readi is turned off this will just scan the first row of the hadamard matrix (all 1s)
			value = SCALE_F2(value, hadamard[hadamard_offset + g]);

			value = SCALE_F2(value, apo);

			incoherent_total += NORM_SQUARE_F2(value);
			channel_total = ADD_F2(channel_total, value);

			rx_vec.y += Constants.pitches.x;
		}

	}

	__syncthreads();

	das_samples[channel_id] = channel_total;
	das_samples[channel_id + MAX_CHANNEL_COUNT] = { incoherent_total, 0.0f };

	cuComplex vox_total = reduce_shared_sum(das_samples, channel_count);

	__syncthreads();
	//cuComplex incoherent_sum = reduce_shared_sum(das_samples + MAX_CHANNEL_COUNT, channel_count);

	__syncthreads();
	if (channel_id == 0)
	{
		//float coherence_factor = NORM_SQUARE_F2(vox_total) / (incoherent_sum.x * channel_count);
		volume[volume_offset] = vox_total;
		//volume[volume_offset] = SCALE_F2(vox_total, coherence_factor);
	}

}


bool
beamformer::beamform(cuComplex* d_volume, const cuComplex* d_rf_data, float3 focus_pos, float samples_per_meter, float f_number)
{

	TransmitType transmit_type;


	if (focus_pos.z == INFINITY)
	{
		focus_pos.z = 0.0f; // This lets us reuse focusing code for plane waves
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
		Session.xdc_maxes,
		f_number,
		samples_per_meter,
		Session.sequence,
	};
	CUDA_RETURN_IF_ERROR(cudaMemcpyToSymbol(Constants, &consts, sizeof(KernelConstants)));

	uint3 vox_counts = vol_config.voxel_counts;

	bool per_voxel = true;
	auto start = std::chrono::high_resolution_clock::now();

	if (per_voxel)
	{
		uint xy_count = vox_counts.x * vox_counts.y;
		dim3 grid_dim = { (uint)ceilf((float)xy_count / MAX_THREADS_PER_BLOCK), (uint)vox_counts.z, 1 };
		dim3 block_dim = { MAX_THREADS_PER_BLOCK, 1, 1 };

		_kernels::per_voxel_beamform << < grid_dim, block_dim >> > (d_rf_data, d_volume, Session.readi_group, Session.d_hadamard);
	}
	else
	{
		dim3 grid_dim = { vox_counts.x, vox_counts.y, vox_counts.z };
		dim3 block_dim = { Session.decoded_dims.y, 1, 1 };

		_kernels::per_channel_beamform << < grid_dim, block_dim >> > (d_rf_data, d_volume, Session.readi_group, Session.d_hadamard);
	}


	CUDA_RETURN_IF_ERROR(cudaGetLastError());
	CUDA_RETURN_IF_ERROR(cudaDeviceSynchronize());

	auto end = std::chrono::high_resolution_clock::now();
	std::chrono::duration<double> elapsed = end - start;
	std::cout << "Kernel duration: " << elapsed.count() << " seconds" << std::endl;

	return true;
}

