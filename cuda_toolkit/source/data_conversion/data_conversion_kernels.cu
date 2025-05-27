#include "data_conversion_kernels.cuh"

namespace data_conversion::kernels
{

    template <SupportedType T> __global__ void
    convert_to_cf32(const T* input, float2* output, uint2 input_dims, uint3 output_dims)
    {
        uint raw_sample_idx = threadIdx.x + blockIdx.x * blockDim.x;
        uint raw_channel_idx = blockIdx.y;
        uint tx_idx = raw_sample_idx / output_dims.x;
        
        if (tx_idx >= output_dims.z) return;

        uint output_channel_idx = Channel_Mapping[raw_channel_idx];
        uint output_sample_idx = raw_sample_idx % output_dims.x;

        uint input_idx = (uint)(raw_channel_idx * input_dims.x) + raw_sample_idx;
        uint output_idx = (tx_idx * output_dims.y * output_dims.x) + (output_channel_idx * output_dims.x) + otuput_sample_idx;

        // Still a real signal but putting it in complex format
        output[output_idx] = {static_cast<float>(input[input_idx]), 0.0f};
    }

} 