#pragma once
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include "../cuda_beamformer_parameters.h"
#include "../defs.h"

static constexpr float CUDART_PI_F = 3.141592654F;
static constexpr uint MAX_TX_COUNT = 128;
namespace beamform::kernels
{
    enum class FocalDirection
    {
        PLANE = 0,
        XZ_PLANE = 1,
        YZ_PLANE = 2,
        SPHERE = 3,
    };  
    struct BeamformerConstants
    {
        // Data Constants
        size_t sample_count;
        size_t channel_count;
        size_t tx_count;
        float2 xdc_mins;
        float2 xdc_maxes;
        float samples_per_meter;
        float3 focal_point;
        float2 pitches;
        int delay_samples;
        FocalDirection focal_direction;
        SequenceId sequence;

        // Render Constants
        uint3 voxel_dims;
        float3 volume_mins;
        float3 resolutions;
        float f_number;

        // Sequence Constants
        u8 mixes_count;
        u8 mixes_offset;

        u8 readi_group_count;
        u8 readi_group_id;
    };
}

__constant__ beamform::kernels::BeamformerConstants Beamformer_Constants;