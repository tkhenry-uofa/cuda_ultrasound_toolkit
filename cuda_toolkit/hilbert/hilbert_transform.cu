#include <cufft.h>
#include <thrust/device_vector.h>
#include <thrust/complex.h>

#include <chrono>

#include "hilbert_transform.cuh"

__host__ bool
hilbert::plan_hilbert(int sample_count, int channel_count, cufftHandle* fwd_handle, cufftHandle* inv_handle)
{
	cufftResult_t fft_result;

	fft_result = cufftCreate(fwd_handle);
	FFT_RETURN_IF_ERROR(fft_result, "Failed to create forward plan.\n")
	fft_result = cufftCreate(fwd_handle);
	FFT_RETURN_IF_ERROR(fft_result, "Failed to create inverse plan.\n")

	// The FFT fails with CUDA_INTERNAL_ERROR if we don't estimate first, this only happens when FFT size isnt a power of 2
	// No idea what the cause it, it isn't in the docs anywhere.
	size_t work_size = 0;
	int dimensions[] = { sample_count };

	fft_result = cufftEstimateMany(1, &sample_count, dimensions, 1, sample_count, dimensions, 1, sample_count, CUFFT_R2C, channel_count, &work_size);
	FFT_RETURN_IF_ERROR(fft_result, "Failed to estimate forward plan.")

	fft_result = cufftPlanMany(fwd_handle, 1, &sample_count, dimensions, 1, sample_count, dimensions, 1, sample_count, CUFFT_R2C, channel_count);
	FFT_RETURN_IF_ERROR(fft_result, "Failed to create forward plan.")

	fft_result = cufftPlanMany(inv_handle, 1, &sample_count, NULL, 1, 0, NULL, 1, 0, CUFFT_C2C, channel_count);
	FFT_RETURN_IF_ERROR(fft_result, "Failed to configure inverse plan.")

	return true;
}

__host__ bool 
hilbert::hilbert_transform(cufftHandle fwd_handle, cufftHandle inv_handle, float* d_input, cufftComplex* d_output)
{
	cufftResult_t fft_result;

	// Exec forward transform
	fft_result = cufftExecR2C(fwd_handle, d_input, d_output);
	FFT_RETURN_IF_ERROR(fft_result, "Failed to execute forward plan.")

	// Exec reverse transform
	fft_result = cufftExecC2C(inv_handle, d_output, d_output, CUFFT_INVERSE);
	FFT_RETURN_IF_ERROR(fft_result, "Failed to execute inverse plan.")

	return true;
}
