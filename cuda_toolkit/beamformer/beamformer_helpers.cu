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
	for (float x = config->minimums.x; x < config->maximums.x; x += config->lateral_resolution) {
		x_range.push_back(x);
		x_count++;
	}
	for (float y = config->minimums.y; y < config->maximums.y; y += config->lateral_resolution) {
		y_range.push_back(y);
		y_count++;
	}
	for (float z = config->minimums.z; z < config->maximums.z; z += config->axial_resolution) {
		z_range.push_back(z);
		z_count++;
	}

	config->voxel_counts = { x_count, y_count, z_count };
	config->total_voxels = x_count * y_count * z_count;

	uint3 voxel_counts = config->voxel_counts;

	return true;
}


