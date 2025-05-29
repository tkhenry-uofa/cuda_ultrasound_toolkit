
#include "beamformer_constants.cuh"
#include "beamformer_utils.cuh"
#include "beamformer_kernels.cuh"


namespace beamform::kernels
{
    __global__ void
    mixes_beamform(const cuComplex* rfData, cuComplex* volume, u8 mixes_rows[128])
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
        if (!utils::check_ranges(vox_loc, Beamformer_Constants.f_number, Beamformer_Constants.xdc_maxes)) return;

        float3 src_pos = Beamformer_Constants.src_pos;

        float3 tx_vec = utils::calc_tx_distance(vox_loc, src_pos, Beamformer_Constants.focal_direction);

        float3 rx_vec = { Beamformer_Constants.xdc_mins.x - vox_loc.x + Beamformer_Constants.pitches.x / 2, Beamformer_Constants.xdc_mins.y - vox_loc.y + Beamformer_Constants.pitches.y / 2, vox_loc.z };

        int delay_samples = 0;

        cuComplex total = { 0.0f, 0.0f }, value;
        float incoherent_sum = 0.0f;

        uint middle_row = Beamformer_Constants.channel_count / 2;

        float3 starting_rx = rx_vec;
        uint sample_count = Beamformer_Constants.sample_count;
        int scan_index;
        uint transmit_count = Beamformer_Constants.tx_count;
        uint channel_count = Beamformer_Constants.channel_count;
        float samples_per_meter = Beamformer_Constants.samples_per_meter;

        uint mixes_count = Beamformer_Constants.mixes_count;
        int mixes_offset = Beamformer_Constants.mixes_offset;
        float total_distance = 0.0f;
        int total_used_channels = 0;


        float path_length_sign = PATH_LENGTH_SIGN(src_pos, vox_loc);

        for (int i = 0; i < mixes_count; i++)
        {
            int t = mixes_rows[i];

            if (t >= middle_row )
            {
                t += mixes_offset;
            }

            rx_vec.y = starting_rx.y + Beamformer_Constants.pitches.y * t;

            for (int e = 0; e < channel_count; e++)
            {
                rx_vec.x = starting_rx.x + Beamformer_Constants.pitches.x * e;

                size_t channel_offset = channel_count * sample_count * t + sample_count * e;
                total_distance = utils::total_path_length(tx_vec, rx_vec, src_pos.z, path_length_sign);
                scan_index = (int)(total_distance * samples_per_meter) + delay_samples;
                value = __ldg(&rfData[channel_offset + scan_index - 1]);

                if (t == 0)
                {
                    value = SCALE_F2(value, I_SQRT_128);
                }

                float apo = utils::f_num_apodization(NORM_F2(rx_vec), vox_loc.z, Beamformer_Constants.f_number);
                value = SCALE_F2(value, apo);


                total = ADD_F2(total, value);
                total_used_channels++;
                //incoherent_sum += NORM_SQUARE_F2(value);

            }
        }

        rx_vec = starting_rx;
        for (int t = 0; t < transmit_count; t++)
        {
            rx_vec.y = starting_rx.y + Beamformer_Constants.pitches.y * t;

            for (int j = 0; j < mixes_count; j++)
            {
                int e = mixes_rows[j];

                if (e >= middle_row)
                {
                    e += mixes_offset;
                }

                rx_vec.x = starting_rx.x + Beamformer_Constants.pitches.x * e;

                size_t channel_offset = channel_count * sample_count * t + sample_count * e;
                total_distance = utils::total_path_length(tx_vec, rx_vec, src_pos.z, path_length_sign);
                scan_index = (int)(total_distance * samples_per_meter) + delay_samples;
                value = __ldg(&rfData[channel_offset + scan_index - 1]);

                if (t == 0)
                {
                    value = SCALE_F2(value, I_SQRT_128);
                }

                float apo = utils::f_num_apodization(NORM_F2(rx_vec), vox_loc.z, Beamformer_Constants.f_number);
                value = SCALE_F2(value, apo);


                total = ADD_F2(total, value);
                total_used_channels++;
                //incoherent_sum += NORM_SQUARE_F2(value);

            }
        }

        float coherent_sum = NORM_SQUARE_F2(total);

        //float coherency_factor = coherent_sum / (incoherent_sum * total_used_channels);
        //volume[volume_offset] = SCALE_F2(total, coherency_factor);

        volume[volume_offset] = total;
    }

    __global__ void
    per_channel_beamform(const cuComplex* rfData, cuComplex* volume, uint readi_group_id, float* hadamard)
    {
        uint tid = threadIdx.x;

        uint channel_id = threadIdx.x;

        __shared__ cuComplex das_samples[MAX_CHANNEL_COUNT * 2]; // 128 * 2

        uint x_voxel = blockIdx.x;
        uint y_voxel = blockIdx.y;
        uint z_voxel = blockIdx.z;

        size_t volume_offset = z_voxel * Beamformer_Constants.voxel_dims.x * Beamformer_Constants.voxel_dims.y + y_voxel * Beamformer_Constants.voxel_dims.x + x_voxel;

        const float3 vox_loc =
        {
            Beamformer_Constants.volume_mins.x + x_voxel * Beamformer_Constants.resolutions.x,
            Beamformer_Constants.volume_mins.y + y_voxel * Beamformer_Constants.resolutions.y,
            Beamformer_Constants.volume_mins.z + z_voxel * Beamformer_Constants.resolutions.z,
        };

        uint transmit_count = Beamformer_Constants.tx_count;
        float3 src_pos = Beamformer_Constants.src_pos;

        // If the voxel is out of the f_number defined range for all elements skip it
        if (!utils::check_ranges(vox_loc, Beamformer_Constants.f_number, Beamformer_Constants.xdc_maxes)) return;

        float3 tx_vec = utils::calc_tx_distance(vox_loc, src_pos, Beamformer_Constants.focal_direction);

        float3 rx_vec =	  { Beamformer_Constants.xdc_mins.x - vox_loc.x + channel_id * Beamformer_Constants.pitches.x + Beamformer_Constants.pitches.x / 2, 
                            Beamformer_Constants.xdc_mins.y - vox_loc.y + Beamformer_Constants.pitches.y / 2, 
                            vox_loc.z };



        uint readi_group_size = Beamformer_Constants.channel_count / Beamformer_Constants.tx_count;
        uint hadamard_offset = Beamformer_Constants.channel_count * readi_group_id;
        float f_number = Beamformer_Constants.f_number;
        uint delay_samples = 12;
        float apo;
        size_t channel_offset = 0;
        uint sample_count = Beamformer_Constants.sample_count;
        uint scan_index;
        uint channel_count = Beamformer_Constants.channel_count;
        cuComplex value;
        cuComplex channel_total = { 0.0f,0.0f };
        float incoherent_total = 0.0f;
        float total_distance = 0.0f;
        float samples_per_meter = Beamformer_Constants.samples_per_meter;
        float path_length_sign = PATH_LENGTH_SIGN(src_pos, vox_loc);
        for (int t = 0; t < transmit_count; t++)
        {
            channel_offset = channel_count * sample_count * t + sample_count * channel_id;

            for (int g = 0; g < readi_group_size; g++)
            {
                total_distance = utils::total_path_length(tx_vec, rx_vec, Beamformer_Constants.src_pos.z, path_length_sign);
                scan_index = (uint)(total_distance * samples_per_meter + delay_samples);
                value = __ldg(&rfData[channel_offset + scan_index - 1]);

                apo = utils::f_num_apodization(NORM_F2(rx_vec), vox_loc.z, f_number);

                // This acts as the final decoding step for the data within the readi group
                // If readi is turned off this will just scan the first row of the hadamard matrix (all 1s)
                value = SCALE_F2(value, hadamard[hadamard_offset + g]);

                value = SCALE_F2(value, apo);

                incoherent_total += NORM_SQUARE_F2(value);
                channel_total = ADD_F2(channel_total, value);

                rx_vec.y += Beamformer_Constants.pitches.x;
            }

        }

        __syncthreads();

        das_samples[channel_id] = channel_total;
        das_samples[channel_id + MAX_CHANNEL_COUNT] = { incoherent_total, 0.0f };

        cuComplex vox_total = utils::reduce_shared_sum(das_samples, channel_count);

        __syncthreads();
        //cuComplex incoherent_sum = reduce_shared_sum(das_samples + MAX_CHANNEL_COUNT, channel_count);

        __syncthreads();
        if (channel_id == 0)
        {
            //float coherence_factor = NORM_SQUARE_F2(vox_total) / (incoherent_sum.x * channel_count);
            volume[volume_offset] = vox_total;
            //volume[volume_offset] = SCALE_F2(vox_total, coherence_factor);
        }

    }

    __global__ void
    per_voxel_beamform(const cuComplex* rfData, cuComplex* volume, uint readi_group_id, float* hadamard)
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
            if (!utils::check_ranges(vox_loc, Beamformer_Constants.f_number, Beamformer_Constants.xdc_maxes)) return;
            
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

            float path_length_sign = PATH_LENGTH_SIGN(src_pos, vox_loc);

            int total_used_channels = 0;
            float total_distance = 0.0f;
            for (int t = 0; t < Beamformer_Constants.tx_count; t++)
            {
                for (int e = 0; e < channel_count; e++)
                {
                    channel_offset = channel_count * sample_count * t + sample_count * e;
                    //for (int g = 0; g < readi_group_size; g++)
                    {

                        total_distance = utils::total_path_length(tx_vec, rx_vec, src_pos.z, path_length_sign);
                        scan_index = total_distance * samples_per_meter + delay_samples;

                        value = utils::cubic_spline(channel_offset, scan_index, rfData);
                        //value = __ldg(&rfData[channel_offset + (uint)scan_index - 1]);

                        if (t == 0)
                        {
                            //value = SCALE_F2(value, I_SQRT_128);
                            value = SCALE_F2(value, I_SQRT_64);
                            //value = { 0,0 };
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

}

