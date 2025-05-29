#include <chrono>

#include "beamformer_kernels.cuh"
#include "beamformer.h"

namespace beamform
{
    kernels::BeamformerConstants
    params_to_constants(const CudaBeamformerParameters& bp)
    {
        kernels::BeamformerConstants constants;

        constants.sample_count = bp.dec_data_dim[0];
        constants.channel_count = bp.dec_data_dim[1];
        constants.tx_count = bp.dec_data_dim[2];

        constants.xdc_mins = {-bp.xdc_transform[12], -bp.xdc_transform[13]};
        constants.xdc_maxes = {bp.xdc_transform[12], bp.xdc_transform[13]};
        constants.samples_per_meter = bp.speed_of_sound / bp.sampling_frequency;

        constants.pitches = {bp.xdc_element_pitch[0], bp.xdc_element_pitch[1]};
        constants.delay_samples = (int)(bp.time_offset * bp.sampling_frequency);
        constants.sequence = bp.das_shader_id;

        constants.voxel_dims = {bp.output_points[0], bp.output_points[1], bp.output_points[2]};
        constants.volume_mins = {bp.output_min_coordinate[0], bp.output_min_coordinate[1], bp.output_min_coordinate[2]};

        float lateral_resolution = (bp.output_max_coordinate[0] - bp.output_min_coordinate[0])
                                   / (bp.output_points[0] - 1);

        float elevation_resolution = lateral_resolution;
        float axial_resolution = (bp.output_max_coordinate[2] - bp.output_min_coordinate[2])
                                   / (bp.output_points[2] - 1);

        constants.resolutions = {lateral_resolution, elevation_resolution, axial_resolution};
        constants.f_number = bp.f_number;

        constants.mixes_count = bp.mixes_count;
        constants.mixes_offset = bp.mixes_offset;

        float3 src_pos = {0.0f, 0.0f, bp.focal_depths[0]};
        constants.src_pos = src_pos;
        if(src_pos.z == INFINITY)
        {
            constants.focal_direction = kernels::FocalDirection::PLANE;
        }
        else if(bp.das_shader_id == SequenceId::HERCULES 
            || bp.das_shader_id == SequenceId::UHURCULES
            || bp.das_shader_id == SequenceId::EPIC_UHERCULES)
        {
            constants.focal_direction = kernels::FocalDirection::YZ_PLANE;
        }
        else
        {
            constants.focal_direction = kernels::FocalDirection::XZ_PLANE;
        }


        return constants;
    }

    bool
    Beamformer::per_voxel_beamform(cuComplex* d_input,
                         cuComplex* d_volume,
                         const CudaBeamformerParameters& bp,
                         const float* d_hadamard)
    {
        auto constants = params_to_constants(bp);
		kernels::copy_kernel_constants(constants);

        std::cout << "Starting beamform." << std::endl;

        uint3 vox_counts = constants.voxel_dims;
        uint xy_count = vox_counts.x * vox_counts.y;
		dim3 grid_dim = { (uint)ceilf((float)xy_count / MAX_THREADS_PER_BLOCK), (uint)vox_counts.z, 1 };
		dim3 block_dim = { MAX_THREADS_PER_BLOCK, 1, 1 };

        auto start = std::chrono::high_resolution_clock::now();
		kernels::per_voxel_beamform << < grid_dim, block_dim >> > (d_input, d_volume, bp.readi_group_id, d_hadamard);

        CUDA_RETURN_IF_ERROR(cudaGetLastError());
	    CUDA_RETURN_IF_ERROR(cudaDeviceSynchronize());

        auto end = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> elapsed = end - start;
        std::cout << "Kernel duration: " << elapsed.count() << " seconds" << std::endl;

        return true;
    }
}