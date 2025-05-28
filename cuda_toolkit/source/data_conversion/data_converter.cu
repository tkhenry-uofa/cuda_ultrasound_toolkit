
#include "data_conversion_kernels.cuh"
#include "data_converter.h"


namespace data_conversion
{
    bool
    DataConverter::copy_channel_mapping(std::span<const int16_t> channel_mapping)
    {
        if (_d_channel_mapping)
        {
            cudaFree(_d_channel_mapping);
            _d_channel_mapping = nullptr;
        }

        size_t size = channel_mapping.size() * sizeof(short);
        CUDA_RETURN_IF_ERROR(cudaMalloc(&_d_channel_mapping, size));
        CUDA_RETURN_IF_ERROR(cudaMemcpy(_d_channel_mapping, channel_mapping.data(), size, cudaMemcpyHostToDevice));
        return true;
    }

    bool
    DataConverter::convert_i16(const int16_t* d_input, float* d_output, uint2 input_dims, uint3 output_dims)
    {
        if (!_d_channel_mapping)
        {
            return false; // Channel mapping not set
        }

        short first_channel_value = sample_value_i16(_d_channel_mapping);

        dim3 block_dim(MAX_THREADS_PER_BLOCK, 1, 1);
        uint grid_length = (uint)ceil((double)input_dims.x / MAX_THREADS_PER_BLOCK);
        dim3 grid_dim(grid_length, output_dims.y, 1);

        kernels::convert_to_f32<int16_t><<<grid_dim, block_dim >>>(d_input, d_output, input_dims, output_dims, _d_channel_mapping);
        CUDA_RETURN_IF_ERROR(cudaGetLastError());
        CUDA_RETURN_IF_ERROR(cudaDeviceSynchronize());

        return true;
    }

    bool
    DataConverter::convert_f32(const float* d_input, float* d_output, uint2 input_dims, uint3 output_dims)
    {
        if (!_d_channel_mapping)
        {
            return false; // Channel mapping not set
        }

        dim3 block_dim(input_dims.y, 1, 1);
        uint grid_length = (uint)ceil((double)input_dims.x / MAX_THREADS_PER_BLOCK);
        dim3 grid_dim(grid_length, output_dims.y, 1);

        kernels::convert_to_f32<float><<<grid_dim, block_dim >>>(d_input, d_output, input_dims, output_dims, _d_channel_mapping);
        CUDA_RETURN_IF_ERROR(cudaGetLastError());
        CUDA_RETURN_IF_ERROR(cudaDeviceSynchronize());

        return true;
    }
}