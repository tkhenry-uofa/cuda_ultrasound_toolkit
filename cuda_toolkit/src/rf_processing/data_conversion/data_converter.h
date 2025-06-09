#pragma once

#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <span>

#include "../../defs.h"

namespace data_conversion
{
    class DataConverter
    {
    public:
        DataConverter() : _d_channel_mapping(nullptr) {};
        ~DataConverter()
        {
            if (_d_channel_mapping)
            {
                cudaFree(_d_channel_mapping);
                _d_channel_mapping = nullptr;
            }
        }

        bool copy_channel_mapping(std::span<const int16_t> channel_mapping);

        bool convert(const void* d_input, void* d_output, InputDataTypes input_type, uint2 input_dims, uint3 output_dims);
        

    private:
        short* _d_channel_mapping;
    };
};