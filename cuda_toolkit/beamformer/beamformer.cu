#include <iostream>
#include <stdexcept>
#include <chrono>
#include <cuda/std/complex>
#include <math_constants.h>
#include <math_functions.h>

#include <cub/cub.cuh> 

#include "beamformer.cuh"

#define PULSE_DELAY 0
//#define SAMPLES_PER_METER 32467.5F // 50 MHz, 1540 m/s
#define SAMPLES_PER_METER 32467.5F // 50 MHz, 1540 m/s



__device__ inline float
old_beamformer::_kernels::f_num_aprodization(float2 lateral_dist, float depth, float f_num)
{
	float apro = f_num * NORM_F2(lateral_dist) / depth;
	apro = fminf(apro, 0.5);
	apro = cosf(CUDART_PI_F * apro);
	return apro * apro;
}

__global__ void
old_beamformer::_kernels::old_complexDelayAndSum(const cuComplex* rfData, const float2* locData, float* volume, cudaTextureObject_t textures[3], float samples_per_meter)
{



	//const float3 vox_loc = {
	//tex1D<float>(textures[0], blockIdx.x % Constants.voxel_dims.x),
	//tex1D<float>(textures[1], blockIdx.x / Constants.voxel_dims.x),
	//tex1D<float>(textures[2], blockIdx.y) };

	const float3 vox_loc =
	{
		Constants.volume_mins.x + blockIdx.x * Constants.resolutions.x,
		Constants.volume_mins.y + blockIdx.y * Constants.resolutions.y,
		Constants.volume_mins.z + blockIdx.z * Constants.resolutions.z,
	};


	uint e = threadIdx.x;
	//uint t = threadIdx.y + blockIdx.z * blockDim.y;
//	uint thread_id = threadIdx.x + threadIdx.y * blockDim.x;


	float tx_distance;
	{
		int dist_sign = ((vox_loc.z - Constants.src_pos.z) > 0) ? 1 : -1;

		
		switch (Constants.tx_type)
		{
		case defs::TX_PLANE:
			tx_distance = vox_loc.z;
			break;

		case defs::TX_Y_FOCUS:
			tx_distance = dist_sign * sqrt(powf(Constants.src_pos.z - vox_loc.z, 2) + powf(Constants.src_pos.y - vox_loc.y, 2)) + Constants.src_pos.z;
			break;

		case defs::TX_X_FOCUS:
			tx_distance = dist_sign * sqrt(powf(Constants.src_pos.z - vox_loc.z, 2) + powf(Constants.src_pos.x - vox_loc.x, 2)) + Constants.src_pos.z;
			break;
		}
	}
	// If the voxel is between the array and the focus this is -1, otherwise it is 1.

	// X is constant with the element number, Y will change with the transmit
	float2 element_loc = {((float)e - 64) * Constants.element_pitch, -64.0f * Constants.element_pitch };

	cuComplex value, total;
	// Beamform this voxel per element 
	for (int t = 0; t < Constants.tx_count; t++)
	{

		/*bool mixes_row = ((e % 4) == 1);
		bool mixes_col = ((t % 4) == 1);
		if (!mixes_row && !mixes_col)
		{
			continue;
		}*/

	//	float2 element_loc = __ldg(&locData[t * Constants.channel_count + e]);



		float2 rx_vec = { element_loc.x - vox_loc.x, element_loc.y - vox_loc.y };


		int scan_index = lroundf((norm3df(rx_vec.x, rx_vec.y, vox_loc.z) + tx_distance) * samples_per_meter + PULSE_DELAY);

		size_t channel_offset = (t * Constants.sample_count * Constants.channel_count) + (e * Constants.sample_count);

		value = __ldg(&rfData[ channel_offset + scan_index - 1]);

		//const float f_number = 1.f;
		//float apro = f_num_aprodization(rx_vec, vox_loc.z, f_number);
		float apro = 1.0f;
		//value = SCALE_F2(value, apro);
		total = cuCaddf(total, value);

		element_loc.y += Constants.element_pitch;

	}

	__shared__ cuComplex temp[MAX_THREADS_PER_BLOCK];

	temp[e] = total;

	__syncthreads();

	// Sum reduction
	int index = 0;
	for (int s = 1; s < Constants.channel_count; s *= 2)
	{
		index = 2 * s * e;

		if (index < (Constants.channel_count - s))
		{

			temp[index] = cuCaddf(temp[index], temp[index + s]);
		}

		__syncthreads();
	}



	if (e == 0)
	{
		atomicAdd(&(volume[blockIdx.z * gridDim.y * gridDim.x + blockIdx.y * gridDim.x + blockIdx.x]), cuCabsf(temp[0]));
	}
}


bool
old_beamformer::beamform(float* d_volume, const cuComplex* d_rf_data, const float2* d_loc_data, float3 src_pos, float samples_per_meter)
{

	defs::TransmitType transmit_type;

	if (src_pos.z == 0.0f)
	{
		transmit_type = defs::TX_PLANE;
	}
	else if (Session.rx_cols)
	{
		// TX on rows (x) axis so we have x focusing
		transmit_type = defs::TX_X_FOCUS;
	}
	else
	{
		transmit_type = defs::TX_Y_FOCUS;
	}

	VolumeConfiguration vol_config = Session.volume_configuration;


	defs::KernelConstants consts =
	{
		Session.decoded_dims.x,
		Session.decoded_dims.y,
		Session.decoded_dims.z,
		vol_config.voxel_counts,
		vol_config.minimums,
		{vol_config.lateral_resolution, vol_config.lateral_resolution, vol_config.axial_resolution},
		src_pos,
		transmit_type,
		Session.element_pitch,
	};


	cudaMemcpyToSymbol(Constants, &consts, sizeof(defs::KernelConstants));

	cudaTextureObject_t* d_textures;
	CUDA_RETURN_IF_ERROR(cudaMalloc(&d_textures, 3 * sizeof(cudaTextureObject_t)));
	CUDA_RETURN_IF_ERROR(cudaMemcpy(d_textures, Session.volume_configuration.textures, 3 * sizeof(cudaTextureObject_t), cudaMemcpyHostToDevice));

	uint3 vox_counts = vol_config.voxel_counts;



	// We fit as many transmits into the y block dimension as possible. 
	// Once the block is full more are added on the z grid dimension
	//uint tx_per_block = MAX_THREADS_PER_BLOCK / consts.channel_count;
	//uint tx_blocks = consts.tx_count / tx_per_block;

	//dim3 grid_dim = { vox_counts.x * vox_counts.y, vox_counts.z, tx_blocks };
	//dim3 block_dim = { (uint)consts.channel_count, tx_per_block, 1 };

	dim3 grid_dim = { vox_counts.x, vox_counts.y, vox_counts.z};
	dim3 block_dim = { (uint)consts.channel_count, 1, 1 };

	auto start = std::chrono::high_resolution_clock::now();

	_kernels::old_complexDelayAndSum << < grid_dim, block_dim >> > (d_rf_data, d_loc_data, d_volume, d_textures, samples_per_meter);

	CUDA_RETURN_IF_ERROR(cudaGetLastError());
	CUDA_RETURN_IF_ERROR(cudaDeviceSynchronize());

	auto end = std::chrono::high_resolution_clock::now();
	std::chrono::duration<double> elapsed = end - start;
	std::cout << "Kernel duration: " << elapsed.count() << " seconds" << std::endl;


	return true;

}

