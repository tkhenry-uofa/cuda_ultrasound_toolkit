#pragma once

#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include "../defs.h"
#include "../cuda_beamformer_parameters.h"
#include "beamformer_constants.cuh"

namespace beamform::kernels
{
    __global__ void
    per_voxel_beamform(const cuComplex* rfData, cuComplex* volume, uint readi_group_id, const float* hadamard);

    __global__ void
    per_channel_beamform(const cuComplex* rfData, cuComplex* volume, uint readi_group_id, const float* hadamard);

    __global__ void
    mixes_beamform(const cuComplex* rfData, cuComplex* volume, u8 mixes_rows[128]);

    __host__ bool
	copy_kernel_constants(const BeamformerConstants& constants);

}


