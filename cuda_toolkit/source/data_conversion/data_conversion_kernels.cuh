#pragma once

#include <type_traits>
#include <cuda_runtime.h>


namespace data_conversion::kernels 
{
    template<typename T>
    concept SupportedType = std::is_same_v<T, int16_t> || std::is_same_v<T, float>;
    
    template<SupportedType T> __global__ void
    convert_to_cf32(const T* input, float2* output, uint2 input_dims, uint3 output_dims);
}