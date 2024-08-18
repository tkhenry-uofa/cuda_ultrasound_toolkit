#include <iostream>
#include <stdexcept>
#include <chrono>

#include "beamformer.cuh"

bool
old_beamformer::configure_textures(VolumeConfiguration* config)
{
	std::vector<float> x_range;
	std::vector<float> y_range;
	std::vector<float> z_range;

	if (config->d_texture_arrays[0] != nullptr)
	{
		// Cleanup old data
		CUDA_RETURN_IF_ERROR(cudaDestroyTextureObject(config->textures[0]));
		CUDA_RETURN_IF_ERROR(cudaDestroyTextureObject(config->textures[1]));
		CUDA_RETURN_IF_ERROR(cudaDestroyTextureObject(config->textures[2]));
		free(config->textures);

		CUDA_RETURN_IF_ERROR(cudaFreeArray(config->d_texture_arrays[0]));
		CUDA_RETURN_IF_ERROR(cudaFreeArray(config->d_texture_arrays[1]));
		CUDA_RETURN_IF_ERROR(cudaFreeArray(config->d_texture_arrays[2]));
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

	CUDA_RETURN_IF_ERROR(cudaMallocArray(&(config->d_texture_arrays[0]), &channel_desc, voxel_counts.x));
	CUDA_RETURN_IF_ERROR(cudaMemcpyToArray(config->d_texture_arrays[0], 0, 0, x_range.data(), voxel_counts.x * sizeof(float), cudaMemcpyHostToDevice));
	tex_res_desc.res.array.array = config->d_texture_arrays[0];
	CUDA_RETURN_IF_ERROR(cudaCreateTextureObject(&(config->textures[0]), &tex_res_desc, &tex_desc, NULL));

	CUDA_RETURN_IF_ERROR(cudaMallocArray(&(config->d_texture_arrays[1]), &channel_desc, voxel_counts.y));
	CUDA_RETURN_IF_ERROR(cudaMemcpyToArray(config->d_texture_arrays[1], 0, 0, y_range.data(), voxel_counts.y * sizeof(float), cudaMemcpyHostToDevice));
	tex_res_desc.res.array.array = config->d_texture_arrays[1];
	CUDA_RETURN_IF_ERROR(cudaCreateTextureObject(&(config->textures[1]), &tex_res_desc, &tex_desc, NULL));

	CUDA_RETURN_IF_ERROR(cudaMallocArray(&(config->d_texture_arrays[2]), &channel_desc, voxel_counts.z));
	CUDA_RETURN_IF_ERROR(cudaMemcpyToArray(config->d_texture_arrays[2], 0, 0, z_range.data(), voxel_counts.z * sizeof(float), cudaMemcpyHostToDevice));
	tex_res_desc.res.array.array = config->d_texture_arrays[2];
	CUDA_RETURN_IF_ERROR(cudaCreateTextureObject(&(config->textures[2]), &tex_res_desc, &tex_desc, NULL));

	return true;
}


