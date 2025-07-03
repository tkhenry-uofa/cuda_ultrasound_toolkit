#include "hadamard.cuh"


#include "hadamard_decoder.h"


namespace decoding
{

	bool HadamardDecoder::test_convert_and_decode(void* d_input, cuComplex* d_output, InputDataTypes input_type, uint2 input_dims, uint3 output_dims,
										   short* d_channel_mapping)
	{
		if (!_d_hadamard)
		{
			std::cerr << "Hadamard matrix not set." << std::endl;
			return false; // Hadamard matrix not set
		}

		dim3 block_dim(output_dims.z, 1, 1);
		dim3 grid_dim(output_dims.x, output_dims.y, 1);

		switch(input_type)
		{
			case InputDataTypes::TYPE_I16:
			{
				using T = types::type_for_t<InputDataTypes::TYPE_I16>;
				static_assert(std::is_same_v<T, int16_t>, "Type mismatch for I16 conversion");
				kernels::convert_and_decode<T><<<grid_dim, block_dim>>>(static_cast<const T*>(d_input), d_output, input_dims, output_dims, d_channel_mapping, _compact_hadamard_test);
				break;
			}
			case InputDataTypes::TYPE_F32:
			{
				using T = types::type_for_t<InputDataTypes::TYPE_F32>;
				static_assert(std::is_same_v<T, float>, "Type mismatch for F32 conversion");
				kernels::convert_and_decode<T><<<grid_dim, block_dim>>>(static_cast<const T*>(d_input), d_output, input_dims, output_dims, d_channel_mapping, _compact_hadamard_test);
				break;
			}
			default:
			{
				std::cerr << "Hadamard decoder: Unsupported input data type." << std::endl;
				return false;
			} 
		}

		CUDA_RETURN_IF_ERROR(cudaGetLastError());
		CUDA_RETURN_IF_ERROR(cudaDeviceSynchronize());

		return true;
	}
}

