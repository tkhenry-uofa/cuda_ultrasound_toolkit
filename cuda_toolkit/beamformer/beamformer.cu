#include <iostream>
#include <stdexcept>
#include <chrono>
#include <cuda/std/complex>
#include <math_constants.h>
#include <math_functions.h>

#include "beamformer.cuh"

#define PULSE_DELAY 0
//#define SAMPLES_PER_METER 32467.5F // 50 MHz, 1540 m/s
#define SAMPLES_PER_METER 32467.5F // 50 MHz, 1540 m/s



__device__ inline float
_kernels::f_num_aprodization(float2 lateral_dist, float depth, float f_num)
{
	float apro = f_num * NORM_F2(lateral_dist) / depth;
	apro = cosf(CUDART_PI_F * apro);
	return apro * apro;
}

__global__ void
_kernels::old_complexDelayAndSum(const cuComplex* rfData, const float2* locData, float* volume, cudaTextureObject_t textures[3], float samples_per_meter)
{
	__shared__ cuComplex temp[MAX_THREADS_PER_BLOCK];

	const float f_number = 1.0f;
	int e = threadIdx.x;

	if (e >= Constants.channel_count)
	{
		return;
	}
	temp[e] = { 0.0f, 0.0f };

	float3 src_pos = Constants.src_pos;

	const float3 vox_loc = {
		tex1D<float>(textures[0], blockIdx.x),
		tex1D<float>(textures[1], blockIdx.y),
		tex1D<float>(textures[2], blockIdx.z) };


	// If the voxel is between the array and the focus this is -1, otherwise it is 1.
	int dist_sign = ((vox_loc.z - src_pos.z) > 0 ) ? 1 : -1;

	float tx_distance;
	switch (Constants.tx_type)
	{
		case defs::TX_PLANE:
			tx_distance = vox_loc.z;
			break;

		case defs::TX_Y_FOCUS:
			tx_distance = dist_sign * sqrt(powf(src_pos.z - vox_loc.z, 2) + powf(src_pos.y - vox_loc.y, 2)) + src_pos.z;
			break;

		case defs::TX_X_FOCUS:
			tx_distance = dist_sign * sqrt(powf(src_pos.z - vox_loc.z, 2) + powf(src_pos.x - vox_loc.x, 2)) + src_pos.z;
			break;
	}

	
	
	float rx_distance;
	int scan_index;
	float2 rx_vec;
	
	cuComplex value;
	// Beamform this voxel per element 
	for (int t = 0; t < Constants.tx_count; t++)
	{
		/*bool mixes_row = ((e % 8) == 1);
		bool mixes_col = ((t % 8) == 1);
		if (!mixes_row && !mixes_col)
		{
			continue;
		}*/

		rx_vec = locData[t * Constants.channel_count + e];
		rx_vec = { rx_vec.x - vox_loc.x, rx_vec.y - vox_loc.y };

		rx_distance = norm3df(ABS(rx_vec.x), ABS(rx_vec.y), vox_loc.z);

		scan_index = lroundf((rx_distance + tx_distance) * samples_per_meter + PULSE_DELAY);

		value = rfData[(t * Constants.sample_count * Constants.channel_count) + (e * Constants.sample_count) + scan_index - 1];

		//float apro = f_num_aprodization(rx_vec, vox_loc.z, f_number);
		float apro = 1.0f;
		temp[e] = cuCaddf(temp[e], value);

	}

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
		volume[blockIdx.z * gridDim.x * gridDim.y + blockIdx.y * gridDim.x + blockIdx.x] = cuCabsf(temp[0]);
	}
}

bool
old_beamformer::configure_textures(VolumeConfiguration* config)
{
	std::vector<float> x_range;
	std::vector<float> y_range;
	std::vector<float> z_range;

	if (config->d_texture_arrays[0] != nullptr )
	{
		// Cleanup old data
		CUDA_THROW_IF_ERROR(cudaDestroyTextureObject(config->textures[0]));
		CUDA_THROW_IF_ERROR(cudaDestroyTextureObject(config->textures[1]));
		CUDA_THROW_IF_ERROR(cudaDestroyTextureObject(config->textures[2]));
		free(config->textures);

		CUDA_THROW_IF_ERROR(cudaFreeArray(config->d_texture_arrays[0]));
		CUDA_THROW_IF_ERROR(cudaFreeArray(config->d_texture_arrays[1]));
		CUDA_THROW_IF_ERROR(cudaFreeArray(config->d_texture_arrays[2]));
		free(config->d_texture_arrays);
	}

	uint x_count, y_count, z_count;
	x_count = y_count = z_count = 0;
	for (float x = config->minimums.x; x <= config->maximums.x; x += config->lateral_resolution) {
		x_range.push_back(x);
		x_count++;
	}
	for (float y = config->minimums.y; y <= config->maximums.y; y += config->lateral_resolution) {
		y_range.push_back(y);
		y_count++;
	}
	for (float z = config->minimums.z; z <= config->maximums.z; z += config->axial_resolution) {
		z_range.push_back(z);
		z_count++;
	}

	config->voxel_counts = { x_count, y_count, z_count };
	config->total_voxels = x_count * y_count * z_count;

	uint3 voxel_counts = config->voxel_counts;

	// TEXTURE SETUP
	// 32 bits in the channel 
	cudaChannelFormatDesc channel_desc = cudaCreateChannelDesc(sizeof(float) * 8, 0, 0, 0, cudaChannelFormatKindFloat);

	cudaTextureDesc tex_desc;
	memset(&tex_desc, 0, sizeof(cudaTextureDesc));
	tex_desc.addressMode[0] = cudaAddressModeClamp;
	tex_desc.filterMode = cudaFilterModePoint;
	tex_desc.readMode = cudaReadModeElementType;
	tex_desc.normalizedCoords = false;

	cudaResourceDesc tex_res_desc;
	memset(&tex_res_desc, 0, sizeof(cudaResourceDesc));
	tex_res_desc.resType = cudaResourceTypeArray;

	CUDA_THROW_IF_ERROR(cudaMallocArray(&(config->d_texture_arrays[0]), &channel_desc, voxel_counts.x));
	CUDA_THROW_IF_ERROR(cudaMemcpyToArray(config->d_texture_arrays[0], 0, 0, x_range.data(), voxel_counts.x * sizeof(float), cudaMemcpyHostToDevice));
	tex_res_desc.res.array.array = config->d_texture_arrays[0];
	CUDA_THROW_IF_ERROR(cudaCreateTextureObject(&(config->textures[0]), &tex_res_desc, &tex_desc, NULL));

	CUDA_THROW_IF_ERROR(cudaMallocArray(&(config->d_texture_arrays[1]), &channel_desc, voxel_counts.y));
	CUDA_THROW_IF_ERROR(cudaMemcpyToArray(config->d_texture_arrays[1], 0, 0, y_range.data(), voxel_counts.y * sizeof(float), cudaMemcpyHostToDevice));
	tex_res_desc.res.array.array = config->d_texture_arrays[1];
	CUDA_THROW_IF_ERROR(cudaCreateTextureObject(&(config->textures[1]), &tex_res_desc, &tex_desc, NULL));

	CUDA_THROW_IF_ERROR(cudaMallocArray(&(config->d_texture_arrays[2]), &channel_desc, voxel_counts.z));
	CUDA_THROW_IF_ERROR(cudaMemcpyToArray(config->d_texture_arrays[2], 0, 0, z_range.data(), voxel_counts.z * sizeof(float), cudaMemcpyHostToDevice));
	tex_res_desc.res.array.array = config->d_texture_arrays[2];
	CUDA_THROW_IF_ERROR(cudaCreateTextureObject(&(config->textures[2]), &tex_res_desc, &tex_desc, NULL));

	return true;
}


bool
old_beamformer::beamform(float* d_volume, const cuComplex* d_rf_data, const float2* d_loc_data, float3 src_pos, float samples_per_meter)
{

	defs::TransmitType transmit_type;

	if (src_pos.z == 0.0f)
	{
		transmit_type = defs::TX_PLANE;
	}
	else if (!Session.rx_cols)
	{
		// TX on rows (x) axis so we have x focusing
		transmit_type = defs::TX_X_FOCUS;
	}
	else
	{
		transmit_type = defs::TX_Y_FOCUS;
	}


	defs::KernelConstants consts =
	{
		Session.decoded_dims.x,
		Session.decoded_dims.y,
		Session.decoded_dims.z,
		src_pos,
		transmit_type,
		Session.volume_configuration.total_voxels
	};


	cudaMemcpyToSymbol(Constants, &consts, sizeof(defs::KernelConstants));

	cudaTextureObject_t* d_textures;
	CUDA_THROW_IF_ERROR(cudaMalloc(&d_textures, 3 * sizeof(cudaTextureObject_t)));
	CUDA_THROW_IF_ERROR(cudaMemcpy(d_textures, Session.volume_configuration.textures, 3 * sizeof(cudaTextureObject_t), cudaMemcpyHostToDevice));

	uint3 vox_counts = Session.volume_configuration.voxel_counts;
	size_t total_voxels = Session.volume_configuration.total_voxels;


	dim3 gridDim = vox_counts;
	auto start = std::chrono::high_resolution_clock::now();

	_kernels::old_complexDelayAndSum << < gridDim, consts.channel_count >> > (d_rf_data, d_loc_data, d_volume, d_textures, samples_per_meter);

	CUDA_THROW_IF_ERROR(cudaGetLastError());
	CUDA_THROW_IF_ERROR(cudaDeviceSynchronize());

	auto end = std::chrono::high_resolution_clock::now();
	std::chrono::duration<double> elapsed = end - start;
	std::cout << "Kernel duration: " << elapsed.count() << " seconds" << std::endl;


	return true;

}

