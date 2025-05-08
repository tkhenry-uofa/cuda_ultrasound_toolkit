#include <cufft.h>
#include <thrust/device_vector.h>
#include <thrust/complex.h>

#include <chrono>

#include "hilbert_transform.cuh"

__device__ inline float2 complex_multiply_f2(float2 a, float2 b) {
	return make_float2(a.x * b.x - a.y * b.y,
		a.x * b.y + a.y * b.x);
}


__global__ void
hilbert::kernels::filter(cuComplex* data, uint sample_count, uint cutoff_sample)
{
	uint sample_idx = threadIdx.x + blockIdx.x * blockDim.x;
	if (sample_idx < sample_count && sample_idx > cutoff_sample)
	{
		data[blockIdx.y * sample_count + sample_idx] = { 0.0f, 0.0f };
	}
	else if (sample_idx == 0 || sample_idx == cutoff_sample)
	{
		cuComplex value = data[blockIdx.y * sample_count + sample_idx];
		data[blockIdx.y * sample_count + sample_idx] = SCALE_F2(value, 0.5f);
	}
}

__global__ void
hilbert::kernels::scale_and_filter(cuComplex* spectrums, cuComplex* filter_kernel, uint sample_count)
{
	uint sample_idx = threadIdx.x + blockIdx.x * blockDim.x;
	if (sample_idx > (sample_count >> 1)) return; // We only need to process the first half of the spectrum.

	float scale_factor = 1.0f / (float)sample_count;
	// Scale the DC and Nyquist components by 0.5 compared to the rest of the spectrum (analytic signal)
	if (sample_idx == 0 || sample_idx == (sample_count >> 1)) scale_factor *= 0.5f; 

	uint channel_offset = blockIdx.y * sample_count;
	spectrums[channel_offset + sample_idx] = SCALE_F2(spectrums[channel_offset + sample_idx], scale_factor);

	if (!filter_kernel) return; // No filter kernel, just scale the spectrum

	spectrums[channel_offset + sample_idx] =
		complex_multiply_f2(spectrums[channel_offset + sample_idx], filter_kernel[sample_idx]);

}

__host__ bool
hilbert::setup_filter(int signal_length, int filter_length, const float* filter)
{
	size_t final_size = signal_length * sizeof(cuComplex);
	cuComplex* d_filter;
	CUDA_RETURN_IF_ERROR(cudaMalloc((void**)&d_filter, final_size));
	CUDA_RETURN_IF_ERROR(cudaMemset(d_filter, 0x00, final_size)); // Padded with zeros to signal_length
	CUDA_FLOAT_TO_COMPLEX(filter, d_filter, filter_length);

	cufftHandle plan;
	CUFFT_RETURN_IF_ERR(cufftPlan1d(&plan, signal_length, CUFFT_C2C, 1));
	CUFFT_RETURN_IF_ERR(cufftExecC2C(plan, d_filter, d_filter, CUFFT_INVERSE));
	Session.d_match_filter = d_filter;
	cufftDestroy(plan);

	return true;
}

__host__ bool
hilbert::f_domain_filter(cuComplex* input)
{
	uint sample_count = Session.decoded_dims.x;
	uint cutoff_sample = (uint)floor((float)sample_count / 2.0f) + 1; // We know the second half of each spectrum is all 0's

	uint channel_count = Session.decoded_dims.y * Session.decoded_dims.z;
	uint grid_length = (uint)ceil((double)cutoff_sample / MAX_THREADS_PER_BLOCK);

	dim3 grid_dims = { grid_length, channel_count, 1 };
	dim3 block_dims = { MAX_THREADS_PER_BLOCK, 1, 1 };

	kernels::scale_and_filter << <grid_dims, block_dims >> > (input, Session.d_match_filter, sample_count);
	CUDA_RETURN_IF_ERROR(cudaGetLastError());
	CUDA_RETURN_IF_ERROR(cudaDeviceSynchronize());

	return true;
}

__host__ bool
hilbert::plan_hilbert(int sample_count, int channel_count)
{
	// The FFT fails with CUDA_INTERNAL_ERROR if we don't estimate first, this only happens when FFT size isnt a power of 2
	// No idea what the cause it, it isn't in the docs anywhere.
	size_t work_size = 0;

	int data_length = sample_count * channel_count;
	int double_l = data_length * 2;

	//CUFFT_THROW_IF_ERR(cufftEstimateMany(1, &sample_count, &sample_count, 1, sample_count, &sample_count, 1, sample_count, CUFFT_R2C, channel_count, &work_size));

	CUFFT_RETURN_IF_ERR(cufftPlanMany(&(Session.forward_plan), 1, &sample_count, &data_length, 1, sample_count, &data_length, 1, sample_count, CUFFT_R2C, channel_count));
	CUFFT_RETURN_IF_ERR(cufftPlanMany(&(Session.inverse_plan), 1, &sample_count, NULL, 1, 0, NULL, 1, 0, CUFFT_C2C, channel_count));

	CUFFT_RETURN_IF_ERR(cufftPlanMany(&(Session.strided_plan), 1, &sample_count, &data_length, 2, sample_count * 2, &data_length, 1, sample_count, CUFFT_R2C, channel_count));
	return true;
}

__host__ bool 
hilbert::hilbert_transform(float* d_input, cuComplex* d_output)
{
	size_t output_size = Session.decoded_dims.x * Session.decoded_dims.y * Session.decoded_dims.z * sizeof(cuComplex);

	CUDA_RETURN_IF_ERROR(cudaMemset(d_output, 0x00, output_size));

	CUFFT_RETURN_IF_ERR(cufftExecR2C(Session.forward_plan, d_input, d_output));

	CUDA_RETURN_IF_ERROR(cudaGetLastError());
	CUDA_RETURN_IF_ERROR(cudaDeviceSynchronize());
	hilbert::f_domain_filter(d_output);
	CUFFT_RETURN_IF_ERR(cufftExecC2C(Session.inverse_plan, d_output, d_output, CUFFT_INVERSE));

	CUDA_RETURN_IF_ERROR(cudaGetLastError());
	CUDA_RETURN_IF_ERROR(cudaDeviceSynchronize());
	return true;
}

__host__ bool
hilbert::hilbert_transform_strided(float* d_input, cuComplex* d_output)
{
	size_t output_size = (size_t)Session.decoded_dims.x * Session.decoded_dims.y * Session.decoded_dims.z * sizeof(cuComplex);

	CUDA_RETURN_IF_ERROR(cudaMemset(d_output, 0x00, output_size));

	CUFFT_RETURN_IF_ERR(cufftExecR2C(Session.strided_plan, d_input, d_output));

	float scale = 1 / ((float)Session.decoded_dims.x / 2);

	float* sample = (float*)d_output;
	///std::cout << "First input value: Re:" << sample_value(d_input+ Session.decoded_dims.x) << " Im: " << sample_value(d_input+1+ Session.decoded_dims.x) << std::endl;
	//std::cout << "First output value: Re:" << sample_value(sample + Session.decoded_dims.x) * scale << " Im: " << sample_value(sample + 1 + Session.decoded_dims.x) * scale << std::endl;
	//hilbert::f_domain_filter(d_output, Session.decoded_dims.x/2);
	//std::cout << "First output value: Re:" << sample_value(sample + Session.decoded_dims.x) * scale << " Im: " << sample_value(sample + 1 + Session.decoded_dims.x) * scale << std::endl;
	//hilbert::f_domain_filter(d_output, 1350);
	CUFFT_RETURN_IF_ERR(cufftExecC2C(Session.inverse_plan, d_output, d_output, CUFFT_INVERSE));
	return true;
}


