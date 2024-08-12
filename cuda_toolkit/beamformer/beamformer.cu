#include <iostream>
#include <stdexcept>
#include <chrono>
#include <cuda/std/complex>

#include "beamformer.cuh"

#define PULSE_DELAY 31
#define SAMPLES_PER_METER 32467.5F // 50 MHz, 1540 m/s



__device__ float
_kernels::f_num_aprodization(float3 vox_loc, float3 element_loc, float f_num)
{
	return 0.f;
}

__global__ void
_kernels::old_complexDelayAndSum(const cuComplex* rfData, const float* locData, float* volume, cudaTextureObject_t textures[3])
{
	__shared__ cuComplex temp[MAX_THREADS_PER_BLOCK];

	int e = threadIdx.x;

	if (e >= Constants.channel_count)
	{
		return;
	}
	temp[e] = { 0.0f, 0.0f };

	float3 src_pos = Constants.src_pos;

	const float3 voxPos = {
		tex1D<float>(textures[0], blockIdx.x),
		tex1D<float>(textures[1], blockIdx.y),
		tex1D<float>(textures[2], blockIdx.z) };


	int dist_sign = ((voxPos.z - src_pos.z) > 0 ) ? 1 : -1;

	float tx_distance;
	switch (Constants.tx_type)
	{
		case defs::TX_PLANE:
			tx_distance = voxPos.z;
			break;

		case defs::TX_X_LINE:
			tx_distance = dist_sign * sqrt(powf(src_pos.z - voxPos.z, 2) + powf(src_pos.y - voxPos.y, 2)) + src_pos.z;
			break;

		case defs::TX_Y_LINE:
			tx_distance = dist_sign * sqrt(powf(src_pos.z - voxPos.z, 2) + powf(src_pos.x - voxPos.x, 2)) + src_pos.z;
			break;
	}

	
	float rx_distance;
	int scanIndex;
	float exPos, eyPos;
	
	cuComplex value;
	// Beamform this voxel per element 
	for (int t = 0; t < Constants.tx_count; t++)
	{
		exPos = locData[2 * (t + e * Constants.tx_count)];
		eyPos = locData[2 * (t + e * Constants.tx_count) + 1];

		float apro = 1.0f;

		// voxel to rx element
		rx_distance = norm3df(voxPos.x - exPos, voxPos.y - eyPos, voxPos.z);

		// Plane wave
		scanIndex = lroundf((rx_distance + tx_distance) * SAMPLES_PER_METER + PULSE_DELAY);

		value = rfData[(t * Constants.sample_count * Constants.channel_count) + (e * Constants.sample_count) + scanIndex - 1];

		value.x = value.x * apro;
		value.y = value.y * apro;
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

	cudaTextureObject_t* textures = config->textures;
	cudaArray_t* d_arrays = config->d_texture_arrays;

	if (d_arrays[0] != nullptr )
	{
		// Cleanup old data
		CUDA_THROW_IF_ERROR(cudaDestroyTextureObject(textures[0]));
		CUDA_THROW_IF_ERROR(cudaDestroyTextureObject(textures[1]));
		CUDA_THROW_IF_ERROR(cudaDestroyTextureObject(textures[2]));
		free(textures);

		CUDA_THROW_IF_ERROR(cudaFreeArray(d_arrays[0]));
		CUDA_THROW_IF_ERROR(cudaFreeArray(d_arrays[1]));
		CUDA_THROW_IF_ERROR(cudaFreeArray(d_arrays[2]));
		free(d_arrays);
	}

	textures = (cudaTextureObject_t*)malloc(3 * sizeof(cudaTextureObject_t));
	d_arrays = (cudaArray_t*)malloc(3 * sizeof(cudaArray_t));

	for (float x = config->minimums.x; x <= config->maximums.x; x += config->lateral_resolution) {
		x_range.push_back(x);
	}
	for (float y = config->minimums.y; y <= config->maximums.y; y += config->lateral_resolution) {
		y_range.push_back(y);
	}
	for (float z = config->minimums.z; z <= config->maximums.z; z += config->axial_resolution) {
		z_range.push_back(z);
	}
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

	CUDA_THROW_IF_ERROR(cudaMallocArray(&(d_arrays[0]), &channel_desc, voxel_counts.x));
	CUDA_THROW_IF_ERROR(cudaMemcpyToArray(d_arrays[0], 0, 0, x_range.data(), voxel_counts.x * sizeof(float), cudaMemcpyHostToDevice));
	tex_res_desc.res.array.array = d_arrays[0];
	CUDA_THROW_IF_ERROR(cudaCreateTextureObject(&(textures[0]), &tex_res_desc, &tex_desc, NULL));

	CUDA_THROW_IF_ERROR(cudaMallocArray(&(d_arrays[1]), &channel_desc, voxel_counts.y));
	CUDA_THROW_IF_ERROR(cudaMemcpyToArray(d_arrays[1], 0, 0, y_range.data(), voxel_counts.y * sizeof(float), cudaMemcpyHostToDevice));
	tex_res_desc.res.array.array = d_arrays[1];
	CUDA_THROW_IF_ERROR(cudaCreateTextureObject(&(textures[1]), &tex_res_desc, &tex_desc, NULL));

	CUDA_THROW_IF_ERROR(cudaMallocArray(&(d_arrays[2]), &channel_desc, voxel_counts.z));
	CUDA_THROW_IF_ERROR(cudaMemcpyToArray(d_arrays[2], 0, 0, z_range.data(), voxel_counts.z * sizeof(float), cudaMemcpyHostToDevice));
	tex_res_desc.res.array.array = d_arrays[2];
	CUDA_THROW_IF_ERROR(cudaCreateTextureObject(&(textures[1]), &tex_res_desc, &tex_desc, NULL));

	return true;
}


bool
old_beamformer::beamform(float* d_volume, const cuComplex* d_rf_data, const float* d_loc_data, float3 src_pos)
{

	defs::KernelConstants consts =
	{
		Session.decoded_dims.x,
		Session.decoded_dims.y,
		Session.decoded_dims.z,
		src_pos,
		defs::TransmitType::TX_PLANE,
		Session.volume_configuration.total_voxels
	};


	cudaMemcpyToSymbol(Constants, &consts, sizeof(defs::KernelConstants));

	uint3 vox_counts = Session.volume_configuration.voxel_counts;
	size_t total_voxels = Session.volume_configuration.total_voxels;


	dim3 gridDim = vox_counts;
	auto start = std::chrono::high_resolution_clock::now();

	_kernels::old_complexDelayAndSum << <gridDim, consts.channel_count >> > (d_rf_data, d_loc_data, d_volume, Session.volume_configuration.textures);

	CUDA_THROW_IF_ERROR(cudaGetLastError());
	CUDA_THROW_IF_ERROR(cudaDeviceSynchronize());

	auto end = std::chrono::high_resolution_clock::now();
	std::chrono::duration<double> elapsed = end - start;
	std::cout << "Kernel duration: " << elapsed.count() << " seconds" << std::endl;


	return true;

}

