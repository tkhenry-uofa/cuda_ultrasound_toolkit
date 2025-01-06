#include <iostream>
#include <stdexcept>
#include <chrono>

#include "beamformer.cuh"

bool
beamformer::configure_volume(VolumeConfiguration* config)
{
	std::vector<float> x_range;
	std::vector<float> y_range;
	std::vector<float> z_range;

	uint x_count, y_count, z_count;
	x_count = y_count = z_count = 0;

	float x_diff = abs(config->maximums.x - config->minimums.x);
	float y_diff = abs(config->maximums.y - config->minimums.y);
	float z_diff = abs(config->maximums.z - config->minimums.z);

	x_count = (uint)floorf(x_diff/config->lateral_resolution);
	y_count = (uint)floorf(y_diff/config->lateral_resolution);
	z_count = (uint)floorf(z_diff/config->axial_resolution);

	float dx = x_diff / x_count;
	float dy = y_diff / y_count;
	float dz = z_diff / z_count;

	float x = config->minimums.x;
	for (int i = 0; i < x_count; i++) 
	{
		x_range.push_back(x);
		x += dx;
	}

	float y = config->minimums.y;
	for (int i = 0; i < y_count; i++) 
	{
		y_range.push_back(y);
		y += dy;
	}

	float z = config->minimums.z;
	for (int i = 0; i < z_count; i++) 
	{
		z_range.push_back(z);
		z += dz;
	}
	

	config->voxel_counts = { x_count, y_count, z_count };
	config->total_voxels = x_count * y_count * z_count;

	uint3 voxel_counts = config->voxel_counts;

	return true;
}


