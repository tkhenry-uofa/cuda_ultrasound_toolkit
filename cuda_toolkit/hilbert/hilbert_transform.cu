﻿#include <cufft.h>
#include <thrust/device_vector.h>
#include <thrust/complex.h>

#include <chrono>

#include "hilbert_transform.cuh"

__global__ void
hilbert::kernels::filter(cuComplex* data, uint sample_count, uint cutoff_sample)
{
	uint sample_idx = threadIdx.x + blockIdx.x * blockDim.x;
	if (sample_idx < sample_count && sample_idx >= cutoff_sample)
	{
		data[blockIdx.y * sample_count + sample_idx] = { 0.0f, 0.0f };
	}
}

__host__ bool
hilbert::f_domain_filter(cuComplex* input, uint cutoff_sample)
{
	uint sample_count = Session.decoded_dims.x;
	uint channel_count = Session.decoded_dims.y * Session.decoded_dims.z;
	uint grid_length = (uint)ceil((double)sample_count / MAX_THREADS_PER_BLOCK);

	dim3 grid_dims = { grid_length, channel_count, 1 };
	dim3 block_dims = { MAX_THREADS_PER_BLOCK, 1, 1 };

	std::cout << "Filtering" << std::endl;
	kernels::filter << <grid_dims, block_dims >> > (input, sample_count, cutoff_sample);
	CUDA_THROW_IF_ERROR(cudaGetLastError());
	CUDA_THROW_IF_ERROR(cudaDeviceSynchronize());

	return true;
}

__host__ bool
hilbert::plan_hilbert(int sample_count, int channel_count)
{
	// The FFT fails with CUDA_INTERNAL_ERROR if we don't estimate first, this only happens when FFT size isnt a power of 2
	// No idea what the cause it, it isn't in the docs anywhere.
	size_t work_size = 0;

	int data_length = sample_count * channel_count;

	CUFFT_THROW_IF_ERR(cufftEstimateMany(1, &sample_count, &sample_count, 1, sample_count, &sample_count, 1, sample_count, CUFFT_R2C, channel_count, &work_size));

	CUFFT_THROW_IF_ERR(cufftPlanMany(&(Session.forward_plan), 1, &sample_count, &data_length, 1, sample_count, &data_length, 1, sample_count, CUFFT_R2C, channel_count));
	CUFFT_THROW_IF_ERR(cufftPlanMany(&(Session.inverse_plan), 1, &sample_count, NULL, 1, 0, NULL, 1, 0, CUFFT_C2C, channel_count));
	return true;
}

__host__ bool 
hilbert::hilbert_transform(float* d_input, cuComplex* d_output)
{
	size_t output_size = Session.decoded_dims.x * Session.decoded_dims.y * Session.decoded_dims.z * sizeof(cuComplex);

	CUDA_THROW_IF_ERROR(cudaMemset(d_output, 0x00, output_size));

	CUFFT_THROW_IF_ERR(cufftExecR2C(Session.forward_plan, d_input, d_output));
	//hilbert::f_domain_filter(d_output, 1350);
	CUFFT_THROW_IF_ERR(cufftExecC2C(Session.inverse_plan, d_output, d_output, CUFFT_INVERSE));
	return true;
}

__host__ bool
hilbert::hilbert_transform2(float* d_input, cuComplex* d_output, cuComplex* d_intermediate)
{
	uint sample_count = Session.decoded_dims.x;


	//CUDA_THROW_IF_ERROR(cudaMemset(d_output, 0x00, output_size));

	CUFFT_THROW_IF_ERR(cufftExecR2C(Session.forward_plan, d_input, d_intermediate));
	
	//hilbert::f_domain_filter(d_intermediate, 1350);
	
	CUFFT_THROW_IF_ERR(cufftExecC2C(Session.inverse_plan, d_intermediate, d_output, CUFFT_INVERSE));

	
	return true;
}