#pragma once

#include <cstddef>
#include <concepts>
#include <type_traits>
#include <cuda_runtime.h>

#include "../defs.h"


namespace data_conversion::kernels 
{
    template<typename T>
    concept SupportedType = std::is_same_v<T, int16_t> || std::is_same_v<T, float>;
    
    template<SupportedType T> __global__ void
    convert_to_f32(const T* input, float* output, uint2 input_dims, uint3 output_dims, const short* d_channel_mapping)
    {
        uint raw_sample_idx = threadIdx.x + blockIdx.x * blockDim.x;
        uint output_channel_idx = blockIdx.y;
        uint tx_idx = raw_sample_idx / output_dims.x;
        uint output_sample_idx = raw_sample_idx % output_dims.x;

        if (tx_idx >= output_dims.z) return;

        uint raw_channel_idx = d_channel_mapping[output_channel_idx]; // This is 1 indexed from matlab
        
        uint input_idx = (raw_channel_idx * input_dims.x) + raw_sample_idx;
        uint output_idx = (tx_idx * output_dims.y * output_dims.x) + (output_channel_idx * output_dims.x) + output_sample_idx;

        output[output_idx] = static_cast<float>(input[input_idx]);
    }
}