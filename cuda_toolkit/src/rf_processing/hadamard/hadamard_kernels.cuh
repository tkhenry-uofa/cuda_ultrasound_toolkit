#pragma once

#include <cstddef>
#include <concepts>
#include <type_traits>
#include <cuda_runtime.h>


#include "../../defs.h"

namespace decoding::kernels
{
	template <typename T>
    concept SupportedConversionType = std::is_same_v<T, int16_t> || std::is_same_v<T, float>;

	/*
	 * Each block is one sample in one channel for all transmits. 
	 * They share a set of data to decode and each load their own hadamard row
	*/
	template<SupportedConversionType T> __global__ void
    convert_and_decode(const T* input, cuComplex* output, uint2 input_dims, uint3 output_dims, const short* d_channel_mapping, const u8* d_hadamard)
	{
		__shared__ float transmit_array[HADAMARD_MAX_SIZE];
		u8 hadamard_row_bytes[HADAMARD_MAX_ROW_BYTES];

		uint sample_idx = blockIdx.x;
		uint output_channel_idx = blockIdx.y;
		uint transmit_idx = threadIdx.x;

		uint input_channel_idx = d_channel_mapping[output_channel_idx];
		uint input_sample_idx = transmit_idx * output_dims.x + sample_idx;

		transmit_array[transmit_idx] = static_cast<float>(input[input_sample_idx + input_dims.x * input_channel_idx]);

		// NOTE: This will break if transmit count is not a multiple of 8, 
		// currently the only array we support that break this are 12 and 20.
		// In those cases the hadamard rows could start mid byte.
		u8 hadamard_byte_count = output_dims.z / BYTE_SIZE;

		for (uint i = 0; i < hadamard_byte_count; i++)
		{
			hadamard_row_bytes[i] = d_hadamard[i + transmit_idx * hadamard_byte_count];
		}

		__syncthreads();


		float result_real = 0.0f;

		for (uint tx = 0; tx < output_dims.z; ++tx)
		{
			uint byte = tx / BYTE_SIZE; 
			u8 bit = tx % BYTE_SIZE; 

			uint hadamard_bit = hadamard_row_bytes[byte];
			
			hadamard_bit = (hadamard_bit >> (BYTE_SIZE - 1 - bit)) << 31; // Get the bit we care about and shift it to the sign bit position

			float current_value = transmit_array[tx];
			
			float result_value = __int_as_float(__float_as_int(current_value) ^ hadamard_bit); // XOR sign bit with hadamard bit
			result_real += result_value;
		}

		//#pragma unroll
		// for (uint i = 0; i < 4; ++i)
		// {
		// 	uint word = *(((uint*)&hadamard_row) + i);
			
		// 	#pragma unroll
		// 	for (uint j = 0; j < 32; ++j)
		// 	{
		// 		uint current_tx = i * 32 + j;

		// 		float current_value = transmit_array[current_tx];

		// 		// If the bit is set, the hadamard value is +1, otherwise -1
		// 		float result_value = __int_as_float( __float_as_int(current_value) ^ ((word >> (31 - j)) << 31) ); // XOR sign bit with hadamard bit
		// 		result_real += result_value;
		// 	}
		// }

		result_real /= output_dims.z; // Hadamard decoding scales by the matrix size, undoing that
		output[sample_idx + output_channel_idx * output_dims.x + transmit_idx * output_dims.x * output_dims.y] = make_cuFloatComplex(result_real, 0.0f);
	};


}