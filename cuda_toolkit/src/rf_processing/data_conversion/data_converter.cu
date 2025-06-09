
#include "data_conversion_kernels.cuh"
#include "data_converter.h"



namespace data_conversion
{
    bool
    DataConverter::copy_channel_mapping(std::span<const int16_t> channel_mapping)
    {
        CUDA_NULL_FREE(_d_channel_mapping);

        size_t size = channel_mapping.size() * sizeof(short);
        CUDA_RETURN_IF_ERROR(cudaMalloc(&_d_channel_mapping, size));
        CUDA_RETURN_IF_ERROR(cudaMemcpy(_d_channel_mapping, channel_mapping.data(), size, cudaMemcpyHostToDevice));
        return true;
    }

    bool
    DataConverter::convert(const void* d_input, void* d_output, InputDataTypes input_type, uint2 input_dims, uint3 output_dims)
    {
        if (!_d_channel_mapping)
        {
            std::cerr << "Channel mapping not set." << std::endl;
            return false; // Channel mapping not set
        }

        dim3 block_dim(input_dims.y, 1, 1);
        uint grid_length = (uint)ceil((double)input_dims.x / MAX_THREADS_PER_BLOCK);
        dim3 grid_dim(grid_length, output_dims.y, 1);


        switch(input_type)
        {
            case InputDataTypes::TYPE_I16:
            {
                using T = types::type_for_t<InputDataTypes::TYPE_I16>;
                static_assert(std::is_same_v<T, int16_t>, "Type mismatch for I16 conversion");
                kernels::convert_to_f32<T><<<grid_dim, block_dim>>>(static_cast<const T*>(d_input), static_cast<float*>(d_output), input_dims, output_dims, _d_channel_mapping);
                break;
            }
            case InputDataTypes::TYPE_F32:
            {
                using T = types::type_for_t<InputDataTypes::TYPE_F32>;
                static_assert(std::is_same_v<T, float>, "Type mismatch for F32 conversion");
                kernels::convert_to_f32<T><<<grid_dim, block_dim>>>(static_cast<const T*>(d_input), static_cast<float*>(d_output), input_dims, output_dims, _d_channel_mapping);
                break;
            }
            default:
            {
                std::cerr << "Data converter: Unsupported input data type." << std::endl;
                return false;
            } 
        }

        CUDA_RETURN_IF_ERROR(cudaGetLastError());
        CUDA_RETURN_IF_ERROR(cudaDeviceSynchronize());

        return true;
    }
}