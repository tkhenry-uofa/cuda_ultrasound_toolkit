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
            CUDA_NULL_FREE(_d_channel_mapping);
        }

        bool copy_channel_mapping(std::span<const int16_t> channel_mapping);

        bool convert(const void* d_input, void* d_output, InputDataTypes input_type, uint2 input_dims, uint3 output_dims);
        
		short* get_channel_mapping() const
		{
			return _d_channel_mapping;
		}

    private:
        short* _d_channel_mapping;
    };
};