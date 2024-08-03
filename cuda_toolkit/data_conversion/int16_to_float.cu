#include <chrono>
#include <iostream>

#include "int16_to_float.cuh"

__constant__ uint Channel_Mapping[TOTAL_TOBE_CHANNELS];

__host__ bool
i16_to_f::copy_channel_mapping(const uint channel_mapping[TOTAL_TOBE_CHANNELS])
{
	CUDA_THROW_IF_ERROR(cudaMemcpyToSymbol(Channel_Mapping, channel_mapping, TOTAL_TOBE_CHANNELS * sizeof(uint)));
	return true;
}

/**
* Input dims are (Sample Count * Transmission Count + Padding) X (Channel Count * 2)
* Output dims are Sample Count * Channel Count * Transmission Count
*/
__global__ void
i16_to_f::_kernels::short_to_float(const i16* input, float* output, uint2 input_dims, RfDataDims output_dims, bool rx_cols)
{
	uint raw_sample_idx = threadIdx.x + blockIdx.x * blockDim.x;
	uint channel_idx = blockIdx.y;

	uint tx_idx = raw_sample_idx / output_dims.sample_count;
	uint sample_idx = raw_sample_idx % output_dims.sample_count;

	if (tx_idx >= output_dims.tx_count)
	{
		return;
	}

	uint output_idx = (tx_idx * output_dims.channel_count * output_dims.sample_count) + (channel_idx * output_dims.sample_count) + sample_idx;
	
	// Shift the index by channel count if we are accessing the column data
	channel_idx = rx_cols ? (channel_idx + output_dims.channel_count) : channel_idx;
	channel_idx = Channel_Mapping[channel_idx];

	uint input_idx = (channel_idx * input_dims.x) + raw_sample_idx;

	i16 result = input[input_idx];

	output[output_idx] = (float)result;

}

__host__ bool
i16_to_f::convert_data(const i16* d_input, float* d_output, bool rx_cols)
{
	dim3 block_dim(MAX_THREADS_PER_BLOCK, 1, 1);

	uint2 input_dims = Session.input_dims;
	RfDataDims output_dims = Session.decoded_dims;
	uint grid_length = (uint)ceil((double)input_dims.x / MAX_THREADS_PER_BLOCK);
	dim3 grid_dim(grid_length, output_dims.channel_count, 1);

	size_t output_size = output_dims.tx_count * output_dims.channel_count * output_dims.sample_count * sizeof(float);
	CUDA_THROW_IF_ERROR(cudaMemset(d_output, 0x00, output_size));

	_kernels::short_to_float << <grid_dim, block_dim >> > (d_input, d_output, input_dims, output_dims, rx_cols);
	CUDA_THROW_IF_ERROR(cudaGetLastError());

	return true;
}