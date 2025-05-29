
#include "beamformer_constants.cuh"
#include "beamformer_utils.cuh"
#include "beamformer_kernels.cuh"


namespace beamform::kernels
{
    __global__ void
    per_voxel_beamform(const cuComplex* rfData, cuComplex* volume, uint readi_group_id, const float* hadamard)
    {
        	uint xy_voxel = threadIdx.x + blockIdx.x * blockDim.x;

            if (xy_voxel > Beamformer_Constants.voxel_dims.x * Beamformer_Constants.voxel_dims.y)
            {
                return;
            }

            uint x_voxel = xy_voxel % Beamformer_Constants.voxel_dims.x;
            uint y_voxel = xy_voxel / Beamformer_Constants.voxel_dims.x;
            uint z_voxel = blockIdx.y;

            size_t volume_offset = z_voxel * Beamformer_Constants.voxel_dims.x * Beamformer_Constants.voxel_dims.y + y_voxel * Beamformer_Constants.voxel_dims.x + x_voxel;

            const float3 vox_loc =
            {
                Beamformer_Constants.volume_mins.x + x_voxel * Beamformer_Constants.resolutions.x,
                Beamformer_Constants.volume_mins.y + y_voxel * Beamformer_Constants.resolutions.y,
                Beamformer_Constants.volume_mins.z + z_voxel * Beamformer_Constants.resolutions.z,
            };

            // If the voxel is out of the f_number defined range for all elements skip it
           // if (!utils::check_ranges(vox_loc, Beamformer_Constants.f_number, Beamformer_Constants.xdc_maxes)) return;
            
            float3 src_pos = Beamformer_Constants.src_pos;
            if (Beamformer_Constants.sequence == SequenceId::FORCES)
            {
                src_pos.x = Beamformer_Constants.xdc_mins.x + Beamformer_Constants.pitches.x / 2;
                src_pos.z = 0; // Ignoring the elevational focus as it is out of plane
            }


            float3 tx_vec = utils::calc_tx_distance(vox_loc, src_pos, Beamformer_Constants.focal_direction);

            float3 rx_vec = { Beamformer_Constants.xdc_mins.x - vox_loc.x + Beamformer_Constants.pitches.x / 2, Beamformer_Constants.xdc_mins.y - vox_loc.y + Beamformer_Constants.pitches.y / 2, vox_loc.z };

            if (Beamformer_Constants.sequence == SequenceId::FORCES)
            {
                rx_vec.y = 0;
            }

            int readi_group_size = Beamformer_Constants.channel_count / Beamformer_Constants.tx_count;
            uint hadamard_offset = Beamformer_Constants.channel_count * readi_group_id;
            int delay_samples = Beamformer_Constants.delay_samples;


            cuComplex total = { 0.0f, 0.0f }, value;
            float incoherent_sum = 0.0f;

            float starting_x = rx_vec.x;
            float apo;
            size_t channel_offset = 0;
            uint sample_count = Beamformer_Constants.sample_count;
            float scan_index;
            uint channel_count = Beamformer_Constants.channel_count;
            float samples_per_meter = Beamformer_Constants.samples_per_meter;

            float focal_distance_sign = copysignf(1.0f, vox_loc.z - src_pos.z);

            int total_used_channels = 0;
            float total_distance = 0.0f;
            for (int t = 0; t < Beamformer_Constants.tx_count; t++)
            {
                for (int e = 0; e < channel_count; e++)
                {
                    channel_offset = channel_count * sample_count * t + sample_count * e;
                    //for (int g = 0; g < readi_group_size; g++)
                    {

                        total_distance = utils::total_path_length(tx_vec, rx_vec, src_pos.z, focal_distance_sign);
                        scan_index = total_distance * samples_per_meter + delay_samples;
						scan_index = utils::clampf(scan_index, 0.0f, (float)sample_count - 1.0f);

                        //value = utils::cubic_spline(channel_offset, scan_index, rfData);
                        value = __ldg(&rfData[channel_offset + (uint)scan_index - 1]);

                        if (t == 0)
                        {
                            //value = SCALE_F2(value, I_SQRT_128);
                            //value = SCALE_F2(value, I_SQRT_64);
                            value = { 0,0 };
                        }

                        apo = utils::f_num_apodization(NORM_F2(rx_vec), vox_loc.z, Beamformer_Constants.f_number);
                        value = SCALE_F2(value, apo);

                        // This acts as the final decoding step for the data within the readi group
                        // If readi is turned off this will just scan the first row of the hadamard matrix (all 1s)
                        //value = SCALE_F2(value, hadamard[hadamard_offset + g]);

                        total = ADD_F2(total, value);
                        incoherent_sum += NORM_SQUARE_F2(value);

                        rx_vec.x += Beamformer_Constants.pitches.x;
                        total_used_channels++;

                    }
                }
                rx_vec.x = starting_x;

                if (Beamformer_Constants.sequence == SequenceId::HERCULES)
                {
                    rx_vec.y += Beamformer_Constants.pitches.x;
                }
                else if (Beamformer_Constants.sequence == SequenceId::FORCES)
                {
                    tx_vec.x += Beamformer_Constants.pitches.x;
                }

            }

            float coherent_sum = NORM_SQUARE_F2(total);

            //float coherency_factor = coherent_sum / (incoherent_sum * total_used_channels);
            //volume[volume_offset] = SCALE_F2(total, coherency_factor);

            volume[volume_offset] = total;
    }

	__host__ bool
	copy_kernel_constants(const BeamformerConstants& constants)
	{
		CUDA_RETURN_IF_ERROR(cudaMemcpyToSymbol(Beamformer_Constants, &constants, sizeof(constants)));
		std::cout << "Beamformer constants copied to device." << std::endl;
	}

}

